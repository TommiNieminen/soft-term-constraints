# soft-term-constraints

This repository contains a script for annotating NMT training data with terminology information. The purpose of this annotation is to implement *soft terminology constraints* in the resulting NMT model. With soft terminology constraints, the NMT model will be more likely to produce the target term specified during decoding (as opposed to *hard terminology constraints*, where the term translation is enforced unconditionally). The advantage of soft constraints over hard constraints is that they allow the NMT model to diverge from the exact specified term translation, which is useful in contexts where the term translation is unsuitable, or in cases where the term translation needs to be inflected.

In the research literature, there are multiple different methods of annotation data for soft terminology constraint training (see [Dinu et al., 2019](https://aclanthology.org/P19-1294/) for the original method, and the [terminology translation task papers for WMT21 for an overview](https://www.statmt.org/wmt21/papers.html) for variants). These methods vary along several different dimensions:
- lemma vs surface form (whether the target terms are annotated in lemma or surface form)
- factored vs tagged (whether the annotations are indicated by using factors or by tags in the source text)
- interleaved vs suffixed (whether the annotations are interleaved in the source text after the source terms or simply concatenated to the end of the whole sentence)
- append vs replace vs mask+append (are term annotations appended to source terms, do they replace the source terms, or is the source term masked and an annotation added)
- aligned synthetic terms vs termbase (generate synthetic terms from aligned noun and verb phrases, or annotate based on an existing termbase)

Currently the only method supported in the script is a lemma-tagged-interleaved-append-aligned, but other methods will be added. The script assumes the parallel corpus has already been segmented with SentencePiece, and that alignments have been generated for the SentencePiece subwords (this is to conform with the [OPUS-MT-train](https://github.com/Helsinki-NLP/OPUS-MT-train) training pipeline). Note that the SentencePiece models need to be generated with same terms tags that are used in script (<term_start>, <term_end>, <trans_end> by default).

Lemmatization is performed with [Stanza](https://stanfordnlp.github.io/stanza/), since it has wide language support. SpaCy was also tested initially, but Stanza performed much better with Finnish lemmatization (note that ([Tilde](https://github.com/tilde-nlp/terminology_translation)) reports that Stanza lemmatization is inaccurate for Latvian, so there's no guarantee that Stanza will work with all language pairs).

The training uses both annotated and unannotated data. The ratios of annotated and unannotated data vary, [Dinu et al., 2019](https://aclanthology.org/P19-1294/) uses 10 percent annotated data, while [Bergmanis and Pinnis, 2021](https://aclanthology.org/2021.eacl-main.271/) uses a ratio of 1:1. As Stanza is quite resource-intensive, it might be best to use a smaller proportion of annotated data.

Script help (note that some arguments have not yet been implemented):

```
  --source_corpus SOURCE_CORPUS
                        Corpus containing the source sentences.
  --target_corpus TARGET_CORPUS
                        Corpus containing the target sentences.
  --source_lang SOURCE_LANG
                        Source language for lemmatization.
  --target_lang TARGET_LANG
                        Target language for lemmatization.
  --alignment_file ALIGNMENT_FILE
                        File containing the alignments between the source and target corpora.
  --source_spm SOURCE_SPM
                        Source sentencepiece model.
  --target_spm TARGET_SPM
                        Target sentencepiece model.
  --output_suffix OUTPUT_SUFFIX
                        Output suffix added to original file names to generate file namesfor the files with added translations.
  --termbase TERMBASE   (NOT IMPLEMENTED YET) If this is defined, the corpus in annotated usinga termbase, if not defined (the
                        current system), aligned noun phrases and verbsare used as term proxies.
  --annotation_method ANNOTATION_METHOD
                        Method to use when annotating target terms to source text.There are several dimensions: lemma vs surface form
                        (lemma/surf),factored or non-factored (fac/nonfac), interleaved vs suffixed
                        (int/suf),append/replace/mask+append. See WMT21 terminology task papers for details.Currently the approach
                        used is lemma-nonfac-int-append, since it seemsto have worked best in WMT21 (and is simple).
  --term_start_tag TERM_START_TAG
                        Tag that is inserted before the source term
  --term_end_tag TERM_END_TAG
                        Tag that is inserted after the source term and before translation lemma
  --trans_end_tag TRANS_END_TAG
                        Tag that is inserted after the translation lemma
  --terms_per_sent_ratio TERMS_PER_SENT_RATIO
                        If default value is used, for each 100 sentences with one term, output 50 sentences with two terms, 25
                        sentences with three terms, 13 sentences with four terms, 7 sentences with five terms etc.If any term count
                        bucket is over its allotment, drop terms from sentence to make it conform with count
  --max_terms_per_sent MAX_TERMS_PER_SENT
                        Max amount of terms to annotate per sentence.
  --batch_size BATCH_SIZE
                        Batch size for stanza processing.
  --max_sents MAX_SENTS
                        Max amount of sentences with terms to generate. If not defined, generate all.
```
