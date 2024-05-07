#!/bin/bash

python3 analysis/feat_analysis.py --sample 0 --level doc
python3 analysis/feat_analysis.py --best_selection --sample contrastive --level seg
python3 analysis/feat_analysis.py --best_selection --sample contrastive --level doc
python3 analysis/feat_colinearity.py

# using results of feature analysis including best features and their thresholds (=the expected TL norm),
# generate custom made instructions ONCE for each of the segments in data/input/de/most_translated_seg_aligned.tsv.gz