import argparse
import ast

def generate_sgm(input_src_path, input_trg_path, terminology_path, source_lang_code, target_lang_code, set_id, output_src_path, output_trg_path):
    with \
        open(input_src_path, 'r', encoding='utf-8') as input_src_file, \
        open(input_trg_path, 'r', encoding='utf-8') as input_trg_file, \
        open(terminology_path, 'r', encoding='utf-8') as term_file, \
        open(output_src_path, 'w', encoding='utf-8') as output_src_file, \
        open(output_trg_path, 'w', encoding='utf-8') as output_trg_file:
        
        output_src_file.write(f'<srcset setid="{set_id}" srclang="{source_lang_code}">\n')
        output_trg_file.write(f'<refset setid="{set_id}" srclang="{source_lang_code} trglang="{target_lang_code}">\n')
        
        output_src_file.write(f'<doc sysid="ref" docid="evalset" genre="terminology" origlang="{source_lang_code}">\n')
        output_src_file.write('<p>\n')

        output_src_file.write(f'<doc sysid="ref" docid="evalset" genre="terminology" origlang="{source_lang_code}">\n')
        output_src_file.write('<p>\n')

        seg_id = 0
        term_id = 0

        for src_line, trg_line, term_line in zip(input_src_file, input_trg_file, term_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            if not term_line.strip():
                continue
            terms = ast.literal_eval(term_line.strip())
            src_words = src_line.split()
            trg_words = trg_line.split()

            output_src_file.write('<seg id="{}"> '.format(seg_id))
            output_trg_file.write('<seg id="{}"> '.format(seg_id))

            src_term_insertions = {}
            trg_term_insertions = {}
            
            for (target_indices, source_lemmas, target_lemmas, source_indices, source_surfs, target_surfs) in terms:
                src = " ".join(source_lemmas)
                trg = " ".join(target_lemmas)
                term_tag = f'<term id="{term_id}" type="src_original_and_tgt_original" src="{src}" tgt="{trg}">'
                src_start_pos = sorted(source_indices)[0]
                src_end_pos = sorted(source_indices)[-1]+1
                trg_start_pos = sorted(target_indices)[0]
                trg_end_pos = sorted(target_indices)[-1]+1

                def insert_term(start_pos, end_pos, term_tag, insertion_dict):
                    if start_pos in insertion_dict:
                        insertion_dict[start_pos] = "</term>" + term_tag
                    else:
                        insertion_dict[start_pos] = term_tag
                    if end_pos in insertion_dict:
                        insertion_dict[end_pos] = "</term>" + insertion_dict[end_pos]
                    else:
                        insertion_dict[end_pos] = "</term>"

                insert_term(src_start_pos, src_end_pos, term_tag, src_term_insertions)
                insert_term(trg_start_pos, trg_end_pos, term_tag, trg_term_insertions)

                term_id += 1

            print(terms)
            for pos in reversed(sorted(src_term_insertions.keys())):
                src_words.insert(pos, src_term_insertions[pos])
            for pos in reversed(sorted(trg_term_insertions.keys())):
                print(pos)
                trg_words.insert(pos, trg_term_insertions[pos])

            output_src_file.write(" ".join(src_words))
            output_trg_file.write(" ".join(trg_words))
            output_src_file.write('</seg>\n')
            output_trg_file.write('</seg>\n')
            seg_id += 1

        output_src_file.write('</p>\n')
        output_src_file.write('</doc>\n')
        output_src_file.write('</srcset>\n')
        output_trg_file.write('</p>\n')
        output_trg_file.write('</doc>\n')
        output_trg_file.write('</refset>\n')


def main():
    parser = argparse.ArgumentParser(description="Generate output SGM file with terminology tagging.")
    parser.add_argument("input_src_path", help="Path to the input src file containing lines of text.")
    parser.add_argument("input_trg_path", help="Path to the input trg file containing lines of text.")
    parser.add_argument("terminology_path", help="Path to the terminology file (each line consist of Python literals)")
    parser.add_argument("source_lang_code", help="Source language code.")
    parser.add_argument("target_lang_code", help="Target language code.")
    parser.add_argument("set_id", help="Set ID to be inserted into the setid attribute.")
    parser.add_argument("output_src_path", help="Path to the source SGM file.")
    parser.add_argument("output_trg_path", help="Path to the target SGM file.")
    args = parser.parse_args()

    generate_sgm(args.input_src_path, args.input_trg_path, args.terminology_path, args.source_lang_code, args.target_lang_code, args.set_id, args.output_src_path, args.output_trg_path)


if __name__ == "__main__":
    main()

