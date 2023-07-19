import argparse
import json
import os.path
import re
import sys
import ast
import gzip
#import sentencepiece as spm
import random
from datetime import datetime

 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Annotates source text with term information according to a specific scheme " +
                    "(the same annotation scheme that was used to annotate the training corpus).")
    parser.add_argument("--source_file", type=str,
                        help="File containing the source sentences.")
    parser.add_argument("--source_output_path", type=str,
                        help="Path where the annotated source will be stored.")
    parser.add_argument("--source_spm", type=str,
                        help="Source sentencepiece model.")
    parser.add_argument("--target_output_path", type=str,
                        help="Path where the term prefix for the target sentence is stored (to be used with prefix decoding).")
    parser.add_argument("--source_lang", type=str,
                        help="Source language for lemmatization.")
    parser.add_argument("--target_lang", type=str,
                        help="Target language for lemmatization.")
    parser.add_argument("--termbase", type=str,
                        help="The termbase containing the terms to annotate")
    parser.add_argument("--terms_per_sentence", type=str,
                        help="Term list where each line contains the terms for the corresponding line in the source sentence")
    parser.add_argument("--annotation_method", type=str,
                        help="Method to use when annotating target terms to source text." +
                             "There are several dimensions: lemma vs surface form (lemma/surf)," +
                             "factored or non-factored (fac/nonfac), interleaved vs suffixed (int/suf)," +
                             "append/replace/mask+append. See WMT21 terminology task papers for details.")
    parser.add_argument("--term_start_tag", type=str, default="<term_start>",
                        help="Tag that is inserted before the source term")
    parser.add_argument("--term_end_tag", type=str, default="<term_end>",
                        help="Tag that is inserted after the source term and before translation lemma")
    parser.add_argument("--trans_end_tag", type=str, default="<trans_end>",
                        help="Tag that is inserted after the translation lemma")
    parser.add_argument("--mask_tag", type=str, default="<term_mask>",
                        help="Tag that is used to mask the source tokens")

    args = parser.parse_args()

    #source_sp_model = spm.SentencePieceProcessor(args.source_spm)

    with \
        open(args.source_file,'rt', encoding="utf8") as orig_source,\
        open(args.source_output_path,'wt', encoding="utf8") as output_source:

        jsonl_terms = args.terms_per_sentence.endswith(".jsonl")
        #wmt23 term format is jsonl, while soft-constraint script stores terms as python object strings
        with open(args.terms_per_sentence,'rt') if jsonl_terms else gzip.open(args.terms_per_sentence,'r') as terms:
            for line in orig_source:
                term_line = terms.readline()
                if not term_line or term_line.isspace():
                    output_source.write(line)
                    continue
                if jsonl_terms:
                    line_terms = [x for x in json.loads(term_line) if x[args.source_lang] and x[args.target_lang]]
                else:
                    line_terms = [{args.source_lang: " ".join(x[4]),args.target_lang: " ".join(x[5])} for x in ast.literal_eval(term_line)]
                
                #First mark matches for all terms
                termindex=0
                for line_term in line_terms:
                    if args.source_lang in ["zh"]:
                        term_without_spaces = line_term[args.source_lang].replace(" ","")
                        line = re.sub(rf'{re.escape(term_without_spaces)}',f'TERM_MATCH_{termindex}',line)
                    else:
                        line = re.sub(rf"\b{re.escape(line_term[args.source_lang])}\b",f'TERM_MATCH_{termindex}',line)
                    termindex += 1
                #Replace term match placeholders with the term annotation
                termindex=0
                for line_term in line_terms:
                    if "append" in args.annotation_method:
                        line = re.sub(f" ?TERM_MATCH_{termindex}",f"{args.term_start_tag} {line_term[args.source_lang]}{args.term_end_tag} {line_term[args.target_lang]}{args.trans_end_tag}", line)
                    elif "replace" in args.annotation_method:
                        line = re.sub(f" ?TERM_MATCH_{termindex}",f"{args.term_start_tag} {line_term[args.target_lang]}{args.term_end_tag}", line)

                    termindex += 1

                output_source.write(line)            
                #print(source_sp_model.encode(line,out_type=str))

