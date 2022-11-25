import argparse
import ast
import os.path
import re
import sys
import gzip
import stanza
import sentencepiece as spm
import random
from datetime import datetime
from stanza import DownloadMethod

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
    stanza_sent = stanza_batch.pop(0)
    if not stanza_batch or stanza_batch[0].text == "SENTENCEBREAK":
        if stanza_batch:
            stanza_batch.pop(0)
        return stanza_sent
    else:
        while stanza_batch and stanza_batch[0].text != "SENTENCEBREAK":
            stanza_batch.pop(0)
        if stanza_batch:
            #remove SENTENCEBREAK
            stanza_batch.pop(0)
        return None

def process_batch(batch,source_stanza_nlp,target_stanza_nlp):
    batch_aligned_chunks = []

    source_stanza_prebatch = "\n\nSENTENCEBREAK\n\n".join([sp_to_sent(x[0]) for x in batch])
    target_stanza_prebatch = "\n\nSENTENCEBREAK\n\n".join([sp_to_sent(x[1]) for x in batch])

    source_stanza_batch = source_stanza_nlp(source_stanza_prebatch).sentences
    target_stanza_batch = target_stanza_nlp(target_stanza_prebatch).sentences

    for (source_line_sp,target_line_sp,line_alignment) in batch:
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
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment),[]))
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
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment),[]))
            continue
        if target_stanza_to_sp is None:
            #sys.stderr.write("Problem with mapping stanza tokens to sp subwords.\n")
            #sys.stderr.write(f"Stanza words: {target_sent_words}\n")
            #sys.stderr.write(f"SP subwords: {sp_target_tokens}\n")
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment),[]))
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

            plain_aligned_chunks = [(list(a),[get_bare_stanza_lemma(e) for e in b],
                                     [get_bare_stanza_lemma(f) for f in c],list(d)) for (a,b,c,d) in aligned_chunks]
            #if getting a bare lemma fails, None will be returned. Remove those chunks from the results.
            plain_aligned_chunks = [x for x in plain_aligned_chunks if not None in x[1] and not None in x[2]]
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment),
                plain_aligned_chunks))
        else:
            batch_aligned_chunks.append(((source_line_sp,target_line_sp,line_alignment),[]))

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

    if args.annotation_method.startswith("lemma-nonfac-int-append"):
        insertion_offset = 1
        for target_alignment, source_lemmas, target_lemmas, source_alignment in aligned_chunks:
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

            chunk_bare_lemmas_sp = target_sp_model.encode_as_pieces(
                " ".join(target_lemmas))

            sp_lemma_count = len(chunk_bare_lemmas_sp)

            term_end_alignment = max(target_alignment)
            #Add two to lemma count to account for the two tags added
            alignment[insertion_point:insertion_point] = [[term_end_alignment]]*(sp_lemma_count+2)

            output_source_line_split[insertion_point:insertion_point] = \
                [args.term_end_tag] + chunk_bare_lemmas_sp + [args.trans_end_tag]
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

    if args.annotation_method.startswith("lemma-nonfac-int-replace"):
        insertion_offset = 1
        for target_alignment, source_lemmas, target_lemmas, source_alignment in aligned_chunks:

            term_start_insertion_point = min(source_alignment) - 1 + insertion_offset
            #insert term start tag before source term
            output_source_line_split_with_align[term_start_insertion_point:term_start_insertion_point] = [
                (args.term_start_tag,[min(target_alignment)])
            ]

            insertion_offset += 1

            chunk_bare_lemmas_sp = target_sp_model.encode_as_pieces(
                " ".join(target_lemmas))

            #remove source term and add term lemma
            output_source_line_split_with_align[min(source_alignment)+insertion_offset-1:max(source_alignment)+insertion_offset] = \
                [(x,[]) for x in chunk_bare_lemmas_sp]
            insertion_offset += len(chunk_bare_lemmas_sp)-len(source_alignment)

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


def process_term_line(
        aligned_chunks, term_buckets, source_line_sp, target_line_sp, alignment_dict):
    (aligned_chunks, term_buckets) = filter_chunks(aligned_chunks, term_buckets)
    (term_source, term_target, term_alignment) = annotate(
        args, source_line_sp, target_line_sp, aligned_chunks, target_sp_model, alignment_dict)
    output_source.write(term_source + "\n")
    output_target.write(term_target + "\n")
    output_alignments.write(term_alignment + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adds translations for specific words or phrases to source sentences. " +
                    "These added translations train the model to handle soft term constraints when decoding. " +
                    "The corpora are expected to be segmented with sentencepiece.")
    parser.add_argument("--source_corpus", type=str,
                        help="Corpus containing the source sentences.")
    parser.add_argument("--source_output_path", type=str,
                        help="Path where the annotated source will be stored.")
    parser.add_argument("--target_corpus", type=str,
                        help="Corpus containing the target sentences.")
    parser.add_argument("--target_output_path", type=str,
                        help="Path where the annotated target will be stored.")
    parser.add_argument("--source_lang", type=str,
                        help="Source language for lemmatization.")
    parser.add_argument("--target_lang", type=str,
                        help="Target language for lemmatization.")
    parser.add_argument("--alignment_file", type=str,
                        help="File containing the alignments between the source and target corpora.")
    parser.add_argument("--alignment_output_path", type=str,
                        help="Path where the annotated source and target alignment will be stored.")
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
    parser.add_argument("--annotation_method", type=str,
                        help="Method to use when annotating target terms to source text." +
                             "There are several dimensions: lemma vs surface form (lemma/surf)," +
                             "factored or non-factored (fac/nonfac), interleaved vs suffixed (int/suf)," +
                             "append/replace/mask+append. See WMT21 terminology task papers for details." +
                             "Currently the approach used is lemma-nonfac-int-replace, since it seems" +
                             "simplest.")
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
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for stanza processing.")
    parser.add_argument("--max_sents", type=str,
                        help="Max amount of sentences with terms to generate. If not defined, generate all. " +
                        "k can be used for 1000s and M for millions")

    args = parser.parse_args()

    if args.max_sents:
        if args.max_sents.isnumeric():
            int_max_sents = int(args.max_sents)
        elif args.max_sents[-1] in ['k','M'] and args.max_sents[0:-1].isnumeric():
            max_sents = args.max_sents.replace('k',"000").replace('M',"000000")
            int_max_sents = int(max_sents)
        else:
            sys.stderr("Invalid max sent count arg")
            sys.exit()

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


    existing_term_annotations_path = re.sub(r".gz$", f".annotations.gz", args.source_corpus)
    new_term_annotations_path = re.sub(r".gz$", f".new_annotations.txt", args.source_corpus)

    #If no existing term annotations, create an empty file to keep later file reading simple
    if not os.path.exists(existing_term_annotations_path):
        open(existing_term_annotations_path, 'a').close()

    source_stanza_nlp = stanza.Pipeline(args.source_lang, processors='tokenize,pos,lemma,depparse')#,download_method=DownloadMethod.NONE)
    target_stanza_nlp = stanza.Pipeline(args.target_lang, processors='tokenize,pos,lemma,depparse')#,download_method=DownloadMethod.NONE)

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
        open(new_term_annotations_path, 'wt', encoding="utf8") as new_term_annotations:

        sent_count = 0
        batch_counter = 0
        sents_with_terms_count = 0

        batch = []

        #source, target and alignment have identical amount of lines
        batch_start_time = datetime.now()
        sys.stderr.write("Starting processing\n")
        for source_line_sp in orig_source:
            sent_count += 1
            target_line_sp = target.readline()

            #parse line word alignment
            current_line_alignment = orig_alignments.readline()
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
                if aligned_chunks:
                    process_term_line(aligned_chunks,term_buckets,source_line_sp,target_line_sp,current_alignment_dict)
                    sents_with_terms_count += 1
                    if args.max_sents and sents_with_terms_count >= int_max_sents:
                        break
            else:
                #start batching for stanza
                batch_counter += 1
                batch.append((source_line_sp, target_line_sp, current_alignment_dict))
                if batch_counter % args.batch_size == 0:
                    batch_aligned_chunks = process_batch(batch,source_stanza_nlp,target_stanza_nlp)
                    #if batch and result lengths don't match, nullify whole batch by adding empty results
                    if len(batch_aligned_chunks) != len(batch):
                        sys.stderr.write(f"Batch and result counts do not match, skipping batch.")
                        batch = []
                        for batch_sent in batch:
                            new_term_annotations.write(str(aligned_chunks) + '\n')
                        continue
                    #sents_with_terms_count += len(sents_with_terms)
                    for ((source_line_sp,target_line_sp,line_alignment),aligned_chunks) in batch_aligned_chunks:
                        if aligned_chunks:
                            new_term_annotations.write(str(aligned_chunks)+'\n')
                            process_term_line(aligned_chunks,term_buckets,source_line_sp,target_line_sp,line_alignment)
                            sents_with_terms_count += 1
                            if args.max_sents and sents_with_terms_count >= int_max_sents:
                                break
                        else:
                            new_term_annotations.write(str(aligned_chunks)+'\n')
                    batch = []
                    if args.max_sents and sents_with_terms_count >= int_max_sents:
                        break
                    else:
                        sys.stderr.write(f"Processed {sent_count} sentences. "+
                            f"Batch duration {datetime.now()-batch_start_time}. Starting new batch.\n")
                        batch_start_time = datetime.now()
                        # this is for slurm jobs, otherwise the output will be buffered for a long time
                        sys.stderr.flush()

        #handle possible unfinished batch (should not fire usually, since batch will be empty
        # when max sents is reached, only occurs if whole corpus is analyzed)
        if batch and not (args.max_sents and sents_with_terms_count >= int_max_sents):
            for ((source_line_sp, target_line_sp, line_alignment), aligned_chunks) in batch_aligned_chunks:
                if aligned_chunks:
                    new_term_annotations.write(str(aligned_chunks) + '\n')
                    process_term_line(aligned_chunks, term_buckets, source_line_sp, target_line_sp, line_alignment)
                    sents_with_terms_count += 1
                else:
                    new_term_annotations.write(f"[]\n")

    with gzip.open(existing_term_annotations_path,'at', encoding="utf8") as existing_term_annotations, \
         open(new_term_annotations_path, 'rt', encoding="utf8") as new_term_annotations:
        for line in new_term_annotations:
            existing_term_annotations.write(line)
    #remove the new annotation file
    os.remove(new_term_annotations_path)
    sys.stderr.write(f"Sentences processed {sent_count}, term sentences generated {sents_with_terms_count}\n")

