#!/bin/bash

# using results of feature analysis including best features and their thresholds (=the expected TL norm, see 4_prompting/cutoffs/),
# pre-generate custom-made instructions to be feed to API for each of the segments in data/input/de_most_translated_seg_aligned.tsv.gz for each of 5 modes like this:
python3 4_prompting/custom_instructions.py --tables 4_prompting/input/ --level seg --mode min --vratio 2.5 --approach feature-based --thresholds 3_feats_analysis/res/ --lang en
python3 4_prompting/custom_instructions.py --tables 4_prompting/input/ --level seg --mode min --vratio 2.5 --approach feature-based --thresholds 3_feats_analysis/res/ --lang de
# it makes sense to send segments to  API in batches to minimize failures
python3 prompt/chunking.py --tables prompt/ol_prompts/ --setup seg_translated_min

# this is a wrapper around python3 prompt/en_api_prompting.py, the outputs go to chunked_outputs/
# the float number is the GPT temperature: 0.7
sh prompt/feeding_chunks.sh self-guided_min 0.7 seg gpt-4

# populate _megaout/input/ and _reworks/input/
python3 prompt/get_megaouts_and_reworks.py --temp 0.7 --setup seg_translated_min --model gpt-4
# populate _reworks/output/
python3 prompt/add_reworked.py --temp 0.7 --setup seg_translated_min --model gpt-4

# NB! Manually curate the tables in data/rewritten/asis/std2...segs.tsv: are all the GPT-4 artefacts deleted?
# put clean multiparallel corpora into data/rewritten/curated/std2...segs.tsv
python3 get_multi_parallel_tsv.py --megaouts 4_prompting/output/ --outdir data/rewritten/asis/ --thres_type ratio2.5
python3 get_multi_parallel_tsv.py --megaouts 4_prompting/output/ --outdir data/rewritten/asis/ --thres_type std2



