#!/bin/bash

python3 1_parse_extract_feats/add_meta_run_stanza.py --tsv_dir data/raw_aligned/ --meta_dir data/raw_aligned/meta/ --outdir data/conllu/
python3 1_parse_extract_feats/tabulate_input.py --indir data/conllu/ --max_docs 1500 --table_unit seg

