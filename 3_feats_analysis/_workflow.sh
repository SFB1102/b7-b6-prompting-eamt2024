#!/bin/bash

python3 3_feats_analysis/univariate_analysis.py --sample contrastive --level seg
python3 3_feats_analysis/univariate_analysis.py  --sample contrastive --level seg --best_selection
python3 3_feats_analysis/univariate_analysis.py --sample contrastive --level doc
python3 3_feats_analysis/feat_colinearity.py

# using results of feature analysis including best features and their thresholds (=the expected TL norm),
# generate custom made instructions ONCE for each of the segments in data/input/de/most_translated_seg_aligned.tsv.gz