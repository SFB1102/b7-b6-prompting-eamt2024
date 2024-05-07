#!/bin/bash
python3 2_classify1/classifier.py --level doc --nbest -1 --nbest_by RFECV

# NB! segment-level classification uses the output of --level doc --nbest -1 --nbest_by RFECV
python3 2_classify1/classifier.py --level seg --nbest -1 --nbest_by RFECV

python3 2_classify1/get_extreme_documents.py --store_item seg