#!/bin/bash

# using results of feature analysis including best features and their thresholds (=the expected TL norm),
# generate custom made instructions ONCE for each of the segments in data/input/de_most_translated_seg_aligned.tsv.gz for each of 5 modes like this:

python3 prompt/custom_instructions.py --tables prompt/input/new/ --level seg --mode min --vratio 2.0 --approach translated --thresholds analysis/res/ --lang en
# it makes sense to send segments to  API in batches to minimize failures
python3 prompt/chunking.py --tables prompt/ol_prompts/new/ --setup seg_translated_min

