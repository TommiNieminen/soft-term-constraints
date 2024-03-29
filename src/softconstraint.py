import argparse
import ast
import os.path
import re
import sys
import gzip
import stanza
import sentencepiece as spm
import random
import json
from contextlib import nullcontext
from datetime import datetime
from stanza import DownloadMethod
from sgm_generator import generate_sgm

class JoinedToken:
    def __init__(self,token,sp_indices):
        self.token = token
        self.sp_indices = sp_indices

def join_sp_tokens(sp_tokens):
    joined_tokens = {}
    token_in_progress = []
    token_sp_indices = []
    joined_index = 0
    for sp_index, sp_token in enumerate(sp_tokens):
        if sp_token.startswith('▁'):
            if token_in_progress:
                joined_tokens[joined_index] = JoinedToken(token_in_progress, token_sp_indices)
                joined_index += 1
            token_in_progress = [sp_token.replace('▁', '')]
            token_sp_indices = [sp_index]
        else:
            token_in_progress.append(sp_token)
            token_sp_indices.append(sp_index)
    joined_tokens[joined_index] = JoinedToken(token_in_progress, token_sp_indices)
    return joined_tokens

def get_stanza_noun_chunks(sent_words):
    #remove the end punctuation, since Stanza dependencies consider it part of the noun chunk
    if sent_words[-1].pos == "PUNCT":
        sent_words = sent_words[:-1]

    nouns = [x for x in sent_words if x.pos == "NOUN"]
    noun_chunks = [[x] for x in nouns]


    for chunk in noun_chunks[:]:
        can_extend = True
        extendable_chunk = chunk
        while can_extend:
            chunk_ids = [x.id for x in extendable_chunk]
            extensions = [x for x in sent_words if x.head in chunk_ids and x.id not in chunk_ids]
            if extensions:
                noun_chunks.append(extendable_chunk+extensions)
                extendable_chunk = extendable_chunk+extensions
            else:
                can_extend = False

    #remove chunks with verbs or punctuation, which might indicate incorrect parses
    noun_chunks = [
        x for x in noun_chunks if not [
            y for y in x if y.pos in ["AUX","VERB","PUNCT"]]]

    #remove chunks with gaps
    noun_chunks = [x for x in noun_chunks if max([y.id for y in x])-min([z.id for z in x]) < len(x)]


    #sort the chunks according to id
    noun_chunks = [sorted(x,key=lambda y: y.id) for x in noun_chunks]

    return noun_chunks


def get_sp_indices_for_chunks(chunks, stanza_to_sp):
    sp_indices_to_chunk = {}
    for chunk in chunks:
        # Stanza word ids start at 1, alignment word ids at 0, so deduct 1
        chunk_indices = [x.id - 1 for x in chunk]
        sp_indices = set()

        for index in chunk_indices:
            sp_indices.update(stanza_to_sp[index])
        sp_indices_to_chunk[frozenset(sp_indices)] = chunk
    return sp_indices_to_chunk

def get_aligned_chunks(
        source_noun_chunks,
        target_noun_chunks,
        source_verbs,
        target_verbs,
        line_alignment,
        source_stanza_to_sp,
        target_stanza_to_sp,
        alignment_dict):

    #Pool verbs and noun chunks here
    sp_indices_to_source_chunks = get_sp_indices_for_chunks(source_noun_chunks+source_verbs, source_stanza_to_sp)
    sp_indices_to_target_chunks = get_sp_indices_for_chunks(target_noun_chunks+target_verbs, target_stanza_to_sp)

    aligned_chunks = []

    for source_index_set,source_chunk in sp_indices_to_source_chunks.items():
        source_alignment = set()
        for source_index in source_index_set:
            if source_index in alignment_dict:
                source_alignment.update(alignment_dict[source_index])

        #do not include chunks with no alignment
        if (source_alignment):
            source_alignment = frozenset(source_alignment)
            if source_alignment in sp_indices_to_target_chunks:
                aligned_chunks.append(
                    (source_alignment,source_chunk,sp_indices_to_target_chunks[source_alignment],source_index_set))

    #remove noun/verb alignments
    aligned_chunks = [x for x in aligned_chunks if not(
        (len(x[1]) == 1 and x[1][0].pos == "VERB" and "NOUN" in [y.pos for y in x[2]]) or
        (len(x[2]) == 1 and x[2][0].pos == "VERB" and "NOUN" in [y.pos for y in x[1]]))]

    #remove overlapped aligned chunks
    occupied_source_indices = {}
    for chunk in aligned_chunks:
        for source_index in chunk[0]:
            if source_index in occupied_source_indices:
                occupied_source_indices[source_index].append(chunk)
            else:
                occupied_source_indices[source_index] = [chunk]

    overlapping_chunks_by_index = [x for x in occupied_source_indices.values() if len(x) > 1]
    for overlapping_chunks in overlapping_chunks_by_index:
        sorted_nonremoved_chunks = [x for x in sorted(overlapping_chunks, key=lambda x: len(x[0])) if x in aligned_chunks]
        if len(sorted_nonremoved_chunks) > 1:
            for short_chunk in sorted_nonremoved_chunks[1:]:
                aligned_chunks.remove(short_chunk)

    return aligned_chunks

def get_stanza_token_sp_indices(stanza_tokens,sp_tokens):
    stanza_index_to_sp = {}
    sp_tokens_without_wordstart = [x.replace('▁','') for x in sp_tokens]

    stanza_texts = [x.text for x in stanza_tokens]
    sp_index = 0
    for (stanza_index,stanza_text) in enumerate(stanza_texts):
        stanza_index_to_sp[stanza_index] = []
        # strip spaces from stanza text, since there can be intralemma spaces but not intra-sp token spaces
        while sp_index+1 != len(sp_tokens_without_wordstart) and stanza_text.strip().startswith(sp_tokens_without_wordstart[sp_index]):
            stanza_index_to_sp[stanza_index].append(sp_index)
            #consume the next sp bit from remaining stanza text
            stanza_text = stanza_text.strip().replace(sp_tokens_without_wordstart[sp_index],"",1)
            sp_index += 1
        if stanza_text:
            if sp_tokens_without_wordstart[sp_index].startswith(stanza_text):
                stanza_index_to_sp[stanza_index].append(sp_index)
                sp_tokens_without_wordstart[sp_index] = sp_tokens_without_wordstart[sp_index].replace(
                    stanza_text,'',1)
            else:
                return None

    return stanza_index_to_sp

def get_bare_stanza_lemma(stanza_token):
    # Hash symbol is used as compound divider in the lemmas, problem is that
    # if the compound was hyphenated, the hash overwrites the hyphen, so restore hyphen
    # note that this will probably fail in some cases, where there are more than two parts to the
    # compound and some are hyphenated and others not (but the effect will be small).
    if stanza_token and stanza_token.lemma and stanza_token.text:
        if "#" in stanza_token.lemma and "#" not in stanza_token.text:
            if "-" in stanza_token.text:
                return stanza_token.lemma.replace('#','-')
            else:
                return stanza_token.lemma.replace('#','')
        else:
            return stanza_token.lemma
    else:
        sys.stderr.write(f"Invalid stanza token {stanza_token}\n")
        return None


def sp_to_sent(sp_line):
    sp_tokens = sp_line.split()
    joined_tokens = join_sp_tokens(sp_tokens)
    sentence = " ".join(["".join(v.token) for k, v in joined_tokens.items()])
    return sentence
 
def get_next_stanza_sent(stanza_batch):
    # the batch can be empty for some reason, not sure why
    if stanza_batch:
        stanza_sent = stanza_batch.pop(0)
    else:
        return None
    # return sent if next batch line is either sentencebreak or batch end
    if not stanza_batch or stanza_batch[0].text == "SENTENCEBREAK":
        # pop SENTENCEBREAK from the top of the batch
        if stanza_batch:
            stanza_batch.pop(0)
        return stanza_sent
    else:
        # if we get here, there are multiple sentences in the batch between sentencebreaks
        # pop all of those sentences out and return None
        while stanza_batch and stanza_batch[0].text != "SENTENCEBREAK":
            stanza_batch.pop(0)
        if stanza_batch:
            # pop SENTENCEBREAK from the top of the batch
            stanza_batch.pop(0)
        return None

def process_batch(batch,source_stanza_nlp,target_stanza_nlp):
    batch_aligned_chunks = []

    source_stanza_prebatch = "\n\nSENTENCEBREAK\n\n".join([sp_to_sent(x[0]) for x in batch])
    target_stanza_prebatch = "\n\nSENTENCEBREAK\n\n".join([sp_to_sent(x[1]) for x in batch])
    #sys.stderr.write(target_stanza_prebatch) 
    batch_start_time = datetime.now()
    source_stanza_batch = source_stanza_nlp(source_stanza_prebatch).sentences
    sys.stderr.write(f"Source stanza processing duration {datetime.now()-batch_start_time}.\n")
    batch_start_time = datetime.now()
    target_stanza_batch = target_stanza_nlp(target_stanza_prebatch).sentences
    sys.stderr.write(f"Target stanza processing duration {datetime.now()-batch_start_time}.\n")
    

    for (source_line_sp,target_line_sp,line_alignment, orig_alignment_string) in batch:
        sp_source_tokens = source_line_sp.split()
        sp_target_tokens = target_line_sp.split()

        joined_source_tokens = join_sp_tokens(sp_source_tokens)
        joined_target_tokens = join_sp_tokens(sp_target_tokens)
        # print(joined_target_tokens)
        source_sentence = " ".join(["".join(v.token) for k, v in joined_source_tokens.items()])
        target_sentence = " ".join(["".join(v.token) for k, v in joined_target_tokens.items()])
        source_stanza_sent = get_next_stanza_sent(source_stanza_batch)
        target_stanza_sent = get_next_stanza_sent(target_stanza_batch)

        #This occurs if there are multiple sentences on the line according to Stanza, skip those
        if not source_stanza_sent or not target_stanza_sent:
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment, orig_alignment_string),[]))
            continue
        if source_stanza_sent.text == "OVERLONG_SENTENCE":
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment, orig_alignment_string),[]))
            continue

        source_sent_words = source_stanza_sent.words
        source_noun_chunks = get_stanza_noun_chunks(source_sent_words)
        source_verbs = [[x] for x in source_sent_words if x.pos == "VERB"]

        target_sent_words = target_stanza_sent.words
        target_noun_chunks = get_stanza_noun_chunks(target_sent_words)
        target_verbs = [[x] for x in target_sent_words if x.pos == "VERB"]

        source_stanza_to_sp = get_stanza_token_sp_indices(source_sent_words, sp_source_tokens)
        target_stanza_to_sp = get_stanza_token_sp_indices(target_sent_words, sp_target_tokens)

        # if syncing stanza and sp fails, move on to next sentence
        if source_stanza_to_sp is None:
            #sys.stderr.write("Problem with mapping stanza tokens to sp subwords.\n")
            #sys.stderr.write(f"Stanza words: {source_sent_words}\n")
            #sys.stderr.write(f"SP subwords: {sp_source_tokens}\n")
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment, orig_alignment_string),[]))
            continue
        if target_stanza_to_sp is None:
            #sys.stderr.write("Problem with mapping stanza tokens to sp subwords.\n")
            #sys.stderr.write(f"Stanza words: {target_sent_words}\n")
            #sys.stderr.write(f"SP subwords: {sp_target_tokens}\n")
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment, orig_alignment_string),[]))
            continue

        aligned_chunks = get_aligned_chunks(
            source_noun_chunks,
            target_noun_chunks,
            source_verbs,
            target_verbs,
            line_alignment,
            source_stanza_to_sp,
            target_stanza_to_sp,
            line_alignment)

        if (aligned_chunks):

            # make sure that the chunks are in left to right order
            aligned_chunks.sort(key=lambda x: max(x[3]))

            plain_aligned_chunks = [
                    (list(a),[get_bare_stanza_lemma(e) for e in b],
                    [get_bare_stanza_lemma(f) for f in c],list(d),
                    [g.text for g in b], [h.text for h in c]) for (a,b,c,d) in aligned_chunks]
            #if getting a bare lemma fails, None will be returned. Remove those chunks from the results.
            plain_aligned_chunks = [x for x in plain_aligned_chunks if not None in x[1] and not None in x[2]]
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment, orig_alignment_string),
                plain_aligned_chunks))
        else:
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment, orig_alignment_string),[]))

    #if len(source_stanza_batch) != 0 or len(target_stanza_batch) != 0:
    #    print("should not happen")

    return batch_aligned_chunks

def filter_chunks(aligned_chunks,term_buckets):
    # use random seed based on aligned chunks to make sure that same data generates same
    # results with same arguments
    random.seed(len(aligned_chunks))
    # truncate term list to max term per sentence value
    if len(aligned_chunks) > args.max_terms_per_sent:
        aligned_chunks = random.sample(aligned_chunks, args.max_terms_per_sent)

    # look for a bucket with room
    bucket_found = False
    for bucket_index in range(0, len(aligned_chunks) - 1):
        if (term_buckets[bucket_index] <=
                term_buckets[bucket_index + 1] * args.terms_per_sent_ratio):
            aligned_chunks = random.sample(aligned_chunks, bucket_index + 1)
            term_buckets[bucket_index] += 1
            bucket_found = True
            break

    # if no room found in lower buckets, place sentence in the maximal bucket for its term count
    if not bucket_found:
        term_buckets[len(aligned_chunks) - 1] += 1
    return (aligned_chunks,term_buckets)

def annotate(args,source_line_sp,target_line_sp,aligned_chunks,
        target_sp_model,alignment_dict):

    #sort aligned chunks according to source position
    aligned_chunks.sort(key=lambda x: min(x[3]))

    #alignment is a list of lists, position in the list indicates source index, inner list
    #contains target indices
    alignment_items = alignment_dict.items()

    output_source_line_split = source_line_sp.split()

    alignment = [[]] * (len(output_source_line_split))  # initialize empty list of lists
    for source_index, target_indices in alignment_items:  # populate with alignment
        alignment[source_index] = target_indices

    output_source_line_split_with_align = list(zip(output_source_line_split,alignment))

    # int-append means interleaving terms in the sentence, appending the term to the source
    if "nonfac-int-append" in args.annotation_method:
        insertion_offset = 1
        for target_alignment, source_lemmas, target_lemmas, source_alignment, source_surfs, target_surfs in aligned_chunks:
            term_start_insertion_point = min(source_alignment) - 1 + insertion_offset
            #insert term start tag into split list
            output_source_line_split[term_start_insertion_point:term_start_insertion_point] = [
                args.term_start_tag
            ]

            term_start_alignment = min(target_alignment)

            #this shifts all alignments on the right side of the insertion point
            alignment[term_start_insertion_point:term_start_insertion_point] = [[term_start_alignment]]

            insertion_offset += 1
            insertion_point = max(source_alignment) + insertion_offset

            if "lemma" in args.annotation_method:
                term_target_sp = target_sp_model.encode_as_pieces(
                    " ".join(target_lemmas))
            elif "surface" in args.annotation_method:
                #this might produce some problems, e.g. with contracted words being split with spaces
                #TODO: get the actual target text span of the term when getting chunks
                term_target_sp = target_sp_model.encode_as_pieces(
                    " ".join(target_surfs))

            sp_lemma_count = len(term_target_sp)

            term_end_alignment = max(target_alignment)
            #Add two to lemma count to account for the two tags added
            alignment[insertion_point:insertion_point] = [[term_end_alignment]]*(sp_lemma_count+2)

            output_source_line_split[insertion_point:insertion_point] = \
                [args.term_end_tag] + term_target_sp + [args.trans_end_tag]
            insertion_offset += sp_lemma_count+2

        # Optionally mask the source terms to make following the constraint more likely
        # (see Encouraging Neural Machine Translation to Satisfy Terminology Constraints,
        # Melissa Ailem, Jingshu Liu, and Raheel Qader. 2021.
        # Encouraging neural machine translation to satisfy terminology constraints.)
        if "+mask" in args.annotation_method:
            #this can only become negative, since the only operation affecting it is
            #the removal of non-word-initial subwords
            alignment_offset = 0
            inside_source_term = False
            for sp_token_index in range(0,len(output_source_line_split)):
                sp_token = output_source_line_split[sp_token_index]
                if sp_token == args.term_start_tag:
                    inside_source_term = True
                    first_term_token = True
                    continue
                if sp_token.startswith(args.term_end_tag):
                    inside_source_term = False
                    continue
                if inside_source_term:
                    #only add one mask token per term
                    #alternatively, use sp_token.startswith('▁'): if one mask token per term token is required
                    if first_term_token:
                        output_source_line_split[sp_token_index] = args.mask_tag
                        first_term_token = False
                    else:
                        #Remove empty tokens after loop
                        output_source_line_split[sp_token_index] = ""
                        #Add alignment for this token to previous token, since the mask should be aligned to the same tokens
                        #as the unmasked source term
                        alignment[sp_token_index+alignment_offset-1].extend(alignment[sp_token_index+alignment_offset])
                        #Remove this token's alignment from the alignment list
                        alignment = alignment[0:sp_token_index+alignment_offset] + alignment[sp_token_index+alignment_offset+1:]
                        alignment_offset -= 1
            #remove empties
            output_source_line_split = [x for x in output_source_line_split if x]

    # int-replace means interleaving terms in the sentence, replacing source term with target term
    if "nonfac-int-replace" in args.annotation_method:
        insertion_offset = 1
        for target_alignment, source_lemmas, target_lemmas, source_alignment, source_surfs, target_surfs in aligned_chunks:

            term_start_insertion_point = min(source_alignment) - 1 + insertion_offset
            #insert term start tag before source term
            output_source_line_split_with_align[term_start_insertion_point:term_start_insertion_point] = [
                (args.term_start_tag,[min(target_alignment)])
            ]

            insertion_offset += 1

            if "lemma" in args.annotation_method:
                term_target_sp = target_sp_model.encode_as_pieces(
                    " ".join(target_lemmas))
            elif "surface" in args.annotation_method:
                #this might produce some problems, e.g. with contracted words being split with spaces
                #TODO: get the actual target text span of the term when getting chunks
                term_target_sp = target_sp_model.encode_as_pieces(
                    " ".join(target_surfs))

            #remove source term and add term lemma
            output_source_line_split_with_align[min(source_alignment)+insertion_offset-1:max(source_alignment)+insertion_offset] = \
                [(x,[]) for x in term_target_sp]
            insertion_offset += len(term_target_sp)-len(source_alignment)

            # insert term end tag after lemma
            term_end_insertion_point = max(source_alignment) + insertion_offset
            output_source_line_split_with_align[term_end_insertion_point:term_end_insertion_point] = [
                (args.term_end_tag, [max(target_alignment)])
            ]
            insertion_offset += 1

        alignment = [x[1] for x in output_source_line_split_with_align]
        output_source_line_split = [x[0] for x in output_source_line_split_with_align]

    output_alignment_string = ""
    for source_index, target_indices in enumerate(alignment):
        for target_index in target_indices:
            output_alignment_string += f"{source_index}-{target_index} "

    source_with_terms = " ".join(output_source_line_split)



    return (source_with_terms, target_line_sp.strip(), output_alignment_string)

def simple_sp_decode(sp_line):
    return sp_line.replace(' ','').replace('▁',' ').strip() 

#returns 1 if term sent added, 0 otherwise
def process_parallel_sentence(
        aligned_chunks, term_buckets, source_line_sp, target_line_sp,
        alignment_dict, orig_alignment_string, output_source, output_target, output_alignments,
        omit_unannotated, keep_original, args, do_not_augment, new_term_annotations=None, jsonl_terms=None, sgm_terms=None):


    if aligned_chunks:
        if new_term_annotations:
            new_term_annotations.write(str(aligned_chunks)+'\n')
        #TODO: same sentences could be annotated with different chunk selections, would be 
        #especially useful for evalsets to increase variety 
        (aligned_chunks, term_buckets) = filter_chunks(aligned_chunks, term_buckets)
        (term_source, term_target, term_alignment) = annotate(
            args, source_line_sp, target_line_sp, aligned_chunks, target_sp_model, alignment_dict)
        if jsonl_terms:
            term_pairs = [{args.source_lang: " ".join(x[4]),args.target_lang: " ".join(x[5])} for x in aligned_chunks]
            json.dump(term_pairs,jsonl_terms,ensure_ascii=False)
            jsonl_terms.write("\n")
        if sgm_terms:
            sgm_terms.write(str(aligned_chunks)+'\n')
        
        #do_not_augment is for cases where you just want the annotations, but don't want to apply them to the source
        if not do_not_augment:
            if args.sp_output:
                output_source.write(term_source.strip() + "\n")
                output_target.write(term_target.strip() + "\n")
            else:
                output_source.write(simple_sp_decode(term_source) + "\n")
                output_target.write(simple_sp_decode(term_target) + "\n")
            output_alignments.write(term_alignment + "\n")
        
        if keep_original or do_not_augment:
            if args.sp_output:
                output_source.write(source_line_sp.strip() + "\n")
                output_target.write(target_line_sp.strip() + "\n")
            else:
                output_source.write(simple_sp_decode(source_line_sp) + "\n")
                output_target.write(simple_sp_decode(target_line_sp) + "\n")
            output_alignments.write(orig_alignment_string + "\n")
        return 1
    else:
        if new_term_annotations:
            new_term_annotations.write("[]\n")
        if not omit_unannotated:
            if args.sp_output:
                output_source.write(source_line_sp.strip() + "\n")
                output_target.write(target_line_sp.strip() + "\n")
            else:
                output_source.write(simple_sp_decode(source_line_sp) + "\n")
                output_target.write(simple_sp_decode(target_line_sp) + "\n")
            output_alignments.write(orig_alignment_string + "\n")

        if sgm_terms:
            sgm_terms.write('\n')

        return 0

 
if __name__ == "__main__":

    #If this is set, GPU won't be used on LUMI
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("CUDA_VISIBLE_DEVICES")

    parser = argparse.ArgumentParser(
        description="Adds translations for specific words or phrases to source sentences. " +
                    "These added translations train the model to handle soft term constraints when decoding. " +
                    "The corpora are expected to be segmented with sentencepiece.")
    parser.add_argument("--sp_input", default=False, action='store_true',
                        help="Is the input sentence piece segmented.")
    parser.add_argument("--sp_output", default=False, action='store_true',
                        help="Should the output be sentence piece segmented?")
    parser.add_argument("--omit_unannotated", default=False, action='store_true',
                        help="Include unannotated sentence pairs in output.")
    parser.add_argument("--do_not_augment", default=False, action='store_true',
                        help="Do not augment data, used in cases where you only want the annotations.")
    parser.add_argument("--source_corpus", type=str,
                        help="Corpus containing the source sentences.")
    parser.add_argument("--source_output_path", type=str,
                        help="Path where the annotated source will be stored.")
    parser.add_argument("--source_sgm_path", type=str,
                        help="Path where the annotated source sgm file (for terminology evaluation) will be stored.")
    parser.add_argument("--target_corpus", type=str,
                        help="Corpus containing the target sentences.")
    parser.add_argument("--target_output_path", type=str,
                        help="Path where the annotated target will be stored.")
    parser.add_argument("--target_sgm_path", type=str,
                        help="Path where the annotated target sgm file (for terminology evaluation) will be stored.")
    parser.add_argument("--source_lang", type=str,
                        help="Source language for lemmatization.")
    parser.add_argument("--target_lang", type=str,
                        help="Target language for lemmatization.")
    parser.add_argument("--alignment_file", type=str,
                        help="File containing the alignments between the source and target corpora.")
    parser.add_argument("--alignment_output_path", type=str,
                        help="Path where the annotated source and target alignment will be stored.")
    parser.add_argument("--term_jsonl_output_path", type=str,
                        help="Path where the jsonl term info is stored.")
    parser.add_argument("--source_spm", type=str,
                        help="Source sentencepiece model.")
    parser.add_argument("--target_spm", type=str,
                        help="Target sentencepiece model.")
    parser.add_argument("--output_suffix", type=str,
                        help="Output suffix added to original file names to generate file names" +
                             "for the files with added translations.")
    parser.add_argument("--termbase", type=str,
                        help="(NOT IMPLEMENTED YET) If this is defined, the corpus in annotated using" +
                             "a termbase, if not defined (the current system), aligned noun phrases and verbs" +
                             "are used as term proxies.")
    parser.add_argument("--annotation_method", type=str, default="lemma-nonfac-int-append",
                        help="Method to use when annotating target terms to source text." +
                             "There are several dimensions: lemma vs surface form (lemma/surf)," +
                             "factored or non-factored (fac/nonfac), interleaved vs suffixed (int/suf)," +
                             "append/replace/mask+append. See WMT21 terminology task papers for details." +
                             "Currently the approach used is lemma-nonfac-int-append, since it seems" +
                             "most sensible.")
    parser.add_argument("--term_start_tag", type=str, default="<term_start>",
                        help="Tag that is inserted before the source term")
    parser.add_argument("--term_end_tag", type=str, default="<term_end>",
                        help="Tag that is inserted after the source term and before translation lemma")
    parser.add_argument("--trans_end_tag", type=str, default="<trans_end>",
                        help="Tag that is inserted after the translation lemma")
    parser.add_argument("--mask_tag", type=str, default="<term_mask>",
                        help="Tag that is used to mask the source tokens")
    parser.add_argument("--terms_per_sent_ratio", type=int, default=2,
                        help="If default value is used, for each 100 sentences with one term, " +
                        "output 50 sentences with two terms, 25 sentences with three terms, 13 " +
                        "sentences with four terms, 7 sentences with five terms etc." +
                        "If any term count bucket is over its allotment, drop terms from " +
                        "sentence to make it conform with count")
    parser.add_argument("--max_terms_per_sent", type=int, default=10,
                        help="Max amount of terms to annotate per sentence.")
    parser.add_argument("--sents_per_term_sent", type=int,
                        help="The ratio of unannotated sentences to annotated sentences." +
                        "With the default 10, there will be 10 unannotated sentences per annptated sentence.")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Batch size for stanza processing.")
    parser.add_argument("--max_sents", type=str,
                        help="Max amount of sentences with terms to generate. If not defined, generate all. " +
                        "k can be used for 1000s and M for millions")


    args = parser.parse_args()

    do_not_augment = args.do_not_augment

    if args.max_sents:
        if args.max_sents.isnumeric():
            int_max_sents = int(args.max_sents)
        elif args.max_sents[-1] in ['k','M'] and args.max_sents[0:-1].isnumeric():
            max_sents = args.max_sents.replace('k',"000").replace('M',"000000")
            int_max_sents = int(max_sents)
        else:
            sys.stderr("Invalid max sent count arg")
            sys.exit()
    else:
        int_max_sents = -1

    if not args.max_sents and args.sents_per_term_sent:
        #get the amount of lines in the corpus
        with gzip.open(args.alignment_file, 'rb') as f:
            for i, l in enumerate(f):
                pass
            terms_sent_max = i / args.sents_per_term_sent
            if int_max_sents == -1 or i < int_max_sents:
                int_max_sents = int(terms_sent_max)


    sys.stderr.write(f"Annotating a maximum of {int_max_sents} sentences.\n")

    term_buckets = [0] * args.max_terms_per_sent
    #all input files should be gzipped
    if not (args.source_corpus.endswith(".gz") and
            args.target_corpus.endswith(".gz") and
            args.alignment_file.endswith(".gz")):
        sys.stderr.write("All input files should have .gz extension\n")
        sys.exit()

    if args.alignment_output_path:
        output_alignments_path = args.alignment_output_path
    else:
        output_alignments_path = re.sub(r".gz$", f".{args.output_suffix}.gz", args.alignment_file)
    if args.source_output_path:
        output_source_path = args.source_output_path
    else:
        output_source_path = re.sub(r".gz$", f".{args.output_suffix}.gz", args.source_corpus)
    if args.target_output_path:
        output_target_path = args.target_output_path
    else:
        output_target_path = re.sub(r".gz$", f".{args.output_suffix}.gz", args.target_corpus)

    keep_original = "-keep" in args.annotation_method

    existing_term_annotations_path = re.sub(r".gz$", f".annotations.gz", args.alignment_file)
    new_term_annotations_path = re.sub(r".gz$", f".new_annotations.txt", args.alignment_file)

    #If no existing term annotations, create an empty file to keep later file reading simple
    if not os.path.exists(existing_term_annotations_path):
        open(existing_term_annotations_path, 'a').close()

    source_stanza_nlp = stanza.Pipeline(args.source_lang, tokenize_batch_size=5000, pos_batch_size=5000, lemma_batch_size=5000, depparse_batch_size=5000, processors='tokenize,pos,lemma,depparse',download_method=DownloadMethod.REUSE_RESOURCES)
    target_stanza_nlp = stanza.Pipeline(args.target_lang, tokenize_batch_size=5000, pos_batch_size=5000, lemma_batch_size=5000, depparse_batch_size=5000, processors='tokenize,pos,lemma,depparse',download_method=DownloadMethod.REUSE_RESOURCES)

    source_sp_model = spm.SentencePieceProcessor(args.source_spm)
    target_sp_model = spm.SentencePieceProcessor(args.target_spm)
    

    with \
        gzip.open(args.source_corpus,'rt', encoding="utf8") as orig_source,\
        gzip.open(args.target_corpus,'rt', encoding="utf8") as target,\
        gzip.open(args.alignment_file,'rt', encoding="utf8") as orig_alignments,\
        gzip.open(output_source_path,'wt', encoding="utf8") as output_source,\
        gzip.open(output_target_path,'wt', encoding="utf8") as output_target,\
        gzip.open(output_alignments_path,'wt', encoding="utf8") as output_alignments,\
        gzip.open(existing_term_annotations_path,'rt', encoding="utf8") as existing_term_annotations, \
        open(new_term_annotations_path, 'wt', encoding="utf8") as new_term_annotations, \
        open(args.term_jsonl_output_path, 'wt', encoding="utf8") if args.term_jsonl_output_path else nullcontext() as jsonl_terms, \
        open(args.source_sgm_path + ".terms", 'wt', encoding="utf8") if args.source_sgm_path else nullcontext() as sgm_terms:

        sent_count = 0
        batch_counter = 0
        sents_with_terms_count = 0

        batch = []

        #source, target and alignment have identical amount of lines
        batch_start_time = datetime.now()
        sys.stderr.write("Starting processing\n")
        for source_line in orig_source:
            sent_count += 1
            target_line = target.readline()

            #parse line word alignment
            current_line_alignment = orig_alignments.readline().strip()
 
            #if enough sentences have been annotated, just output the original lines
            if (int_max_sents != -1 and sents_with_terms_count >= int_max_sents):
                if args.omit_unannotated:
                    break
                else:
                    if args.sp_input and not args.sp_output:
                        output_source.write(simple_sp_decode(source_line) + "\n")
                        output_target.write(simple_sp_decode(target_line) + "\n")
                    else:
                        output_source.write(source_line)
                        output_target.write(target_line)
                    output_alignments.write(current_line_alignment + "\n")
                    continue 

            # mark long sentences that break stanza
            if len(source_line) > 2000 or len(target_line) > 2000:
                source_line_sp = "OVERLONG_SENTENCE"
                target_line_sp = "OVERLONG_SENTENCE"
                current_alignment_dict = {}
                current_line_alignment = ""
            else:
                if not args.sp_input:
                    source_line_sp = " ".join(source_sp_model.encode(source_line,out_type=str))
                    target_line_sp = " ".join(target_sp_model.encode(target_line,out_type=str))
                else:
                    source_line_sp = source_line
                    target_line_sp = target_line

                current_alignment_dict = {}
                for token_alignment in current_line_alignment.split():
                    source_index, target_index = [int(x) for x in token_alignment.split('-') if '-' in token_alignment]
                    if source_index in current_alignment_dict:
                        current_alignment_dict[source_index].append(target_index)
                    else:
                        current_alignment_dict[source_index] = [target_index]

            #check if an annotation of the line exists already in the annotation file
            existing_term_annotation = existing_term_annotations.readline()

            #if there are pre-saved term annotations, use them
            if existing_term_annotation:
                #Parse the pre-saved chunks
                aligned_chunks = ast.literal_eval(existing_term_annotation)
                sents_with_terms_count += process_parallel_sentence(
                        aligned_chunks,term_buckets,source_line_sp,target_line_sp,current_alignment_dict,current_line_alignment,
                        output_source, output_target, output_alignments, args.omit_unannotated,
                        keep_original, args, do_not_augment, jsonl_terms=jsonl_terms, sgm_terms=sgm_terms)
                if int_max_sents != -1 and sents_with_terms_count >= int_max_sents:
                    continue
            else:
                #start batching forV stanza
                batch_counter += 1
                batch.append((source_line_sp, target_line_sp, current_alignment_dict, current_line_alignment))
                if batch_counter % args.batch_size == 0:
                    batch_aligned_chunks = process_batch(batch,source_stanza_nlp,target_stanza_nlp)
                    #if batch and result lengths don't match, nullify whole batch by adding empty results
                    if len(batch_aligned_chunks) != len(batch):
                        sys.stderr.write(f"Batch and result counts do not match, skipping batch.")
                        batch = []
                        continue
                    else:
                        #sents_with_terms_count += len(sents_with_terms)
                        for ((source_line_sp,target_line_sp,line_alignment,orig_alignment_string),aligned_chunks) in batch_aligned_chunks:
                            sents_with_terms_count += process_parallel_sentence(
                                aligned_chunks,term_buckets,source_line_sp,target_line_sp,line_alignment,orig_alignment_string,
                                output_source, output_target, output_alignments, args.omit_unannotated,
                                keep_original, args, do_not_augment, new_term_annotations, jsonl_terms=jsonl_terms, sgm_terms=sgm_terms)
                            if int_max_sents != -1 and sents_with_terms_count >= int_max_sents:
                                break
                    batch = []
                    if int_max_sents != -1 and sents_with_terms_count >= int_max_sents:
                        continue
                    else:
                        sys.stderr.write(f"Processed {sent_count} sentences. "+
                            f"Batch duration {datetime.now()-batch_start_time}. Starting new batch.\n")
                        batch_start_time = datetime.now()
                        # this is for slurm jobs, otherwise the output will be buffered for a long time
                        sys.stderr.flush()

        #handle possible unfinished batch (should not fire usually, since batch will be empty
        # when max sents is reached, only occurs if whole corpus is analyzed)
        if batch and not (int_max_sents != -1 and sents_with_terms_count >= int_max_sents):
            batch_aligned_chunks = process_batch(batch,source_stanza_nlp,target_stanza_nlp)
            if len(batch_aligned_chunks) != len(batch):
                sys.stderr.write(f"Batch and result counts do not match, skipping batch.")
                batch = []
            else:
                for ((source_line_sp, target_line_sp, line_alignment, orig_alignment_string), aligned_chunks) in batch_aligned_chunks:
                    sents_with_terms_count += process_parallel_sentence(
                        aligned_chunks,term_buckets,source_line_sp,target_line_sp,line_alignment,orig_alignment_string,
                        output_source, output_target, output_alignments, args.omit_unannotated,
                        keep_original, args, do_not_augment, new_term_annotations, jsonl_terms=jsonl_terms, sgm_terms=sgm_terms)
                    if int_max_sents != -1 and sents_with_terms_count >= int_max_sents:
                        continue


    with gzip.open(existing_term_annotations_path,'at', encoding="utf8") as existing_term_annotations, \
         open(new_term_annotations_path, 'rt', encoding="utf8") as new_term_annotations:
        for line in new_term_annotations:
            existing_term_annotations.write(line)
    #remove the new annotation file
    os.remove(new_term_annotations_path)
    sys.stderr.write(f"Sentences processed {sent_count}, term sentences generated {sents_with_terms_count}\n")

    #generate sgm files to be used with the terminologyevaluation tool (https://github.com/mahfuzibnalam/terminology_evaluation)
    if args.source_sgm_path and args.target_sgm_path:
        sys.stderr.write(f"Generating sgm with arguments {args.source_corpus}, {args.target_corpus}, {args.source_sgm_path+'.terms'}, {args.source_lang}, {args.target_lang}, evalsets, {args.source_sgm_path}, {args.target_sgm_path}\n")
        generate_sgm(args.source_corpus, args.target_corpus, args.source_sgm_path+".terms", args.source_lang, args.target_lang, "evalsets", args.source_sgm_path, args.target_sgm_path)
