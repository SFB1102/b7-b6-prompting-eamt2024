The materials for this project are available for download from https://doi.org/10.5281/zenodo.11127626:

**EN-DE-Bidirectional-Europarl-UdS corpus** 

1. The initial input (0_align/xml_translationese/) to the parallel corpus-building pipeline

    -- xml_translationese.zip

2. Raw-text sentence-aligned documents for both translation directions (columns=['sdoc_id', 'sseg_id', 'sseg', 'tseg', 'hunalign_qua']) (data/raw_aligned/):
     
     -- deen_wide2018_cap0_score0.3.tsv.gz

     -- ende_wide2018_cap0_score0.5.tsv.gz

     -- meta.zip contains four files with XML tags with metadata to each document in the corpus.

3. The same documents annotated with Stanza (with the conllu-style vertical format for each segment and document wrapped in XML tags containing metadata) (data/conllu/):
     
     -- ORG_WR_DE_EN.conllu.xml.gz (original German)

     -- ORG_WR_EN_DE.conllu.xml.gz (original English)

     -- TR_DE_EN.conllu.xml.gz (translated English)

     -- TR_EN_DE.conllu.xml.gz (translated German)

4. The sentence-level subset with extracted morphsyntactic (or lexicogrammatical) features as described in the paper  (documents longer than 450 tokens, 1500 documents per translation direction) (data/feats_tabled/):

     -- seg-450-1500.feats.tsv.gz

5. A multi-parallel subset of 200 most_translated documents re-written by GPT-4 under various prompting conditions as described in the paper (data/rewritten/):

   -- ratio2.5_de_7aligned_2056segs.tsv

   -- ratio2.5_en_7aligned_2109segs.tsv


If you find this data useful, consider referencing the paper where it is described:

@inproceedings{Kunilovskaya2024prompting,
author = {Kunilovskaya, Maria and Chowdhury, Koel Dutta and Przybyl, Heike and {Espa{\~{n}}a i Bonet}, Cristina and {Van Genabith}, Josef},
title = {{Mitigating Translationese with GPT-4: Strategies and Performance}},
booktitle = {Proceedings of the 25th Annual conference of the European Association for Machine Translation},
month = {24--27 June},
pages = {411--430},
publisher = {Association for Computational Linguistics},
address = {Sheffield, UK},
year = {2024}