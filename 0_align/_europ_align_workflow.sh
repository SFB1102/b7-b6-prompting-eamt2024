#!/bin/bash
cd 0_align/
### re-produce 2018 Europarl version, where the original speeches are limited to those devlivered by the native speakers of EN, DE and ES ###
# re-extract the 10 July 2018 version of EuroParl-UdS from xml_translationese/ folder
# (up to this step the EP verbatim reports need to be processed using https://github.com/chozelinek/europarl codebase)
python3 extract_interventions_mk.py --lpair deen --src xml_translationese/de/originals_ns/ --tgt xml_translationese/en/translations_from_de_n/ --outdir extracted/
python3 extract_interventions_mk.py --lpair ende --src xml_translationese/en/originals_ns/ --tgt xml_translationese/de/translations_from_en_n/ --outdir extracted/
python3 extract_interventions_mk.py --lpair esen --src xml_translationese/es/originals_ns/ --tgt xml_translationese/en/translations_from_es_n/ --outdir extracted/
python3 extract_interventions_mk.py --lpair enes --src xml_translationese/en/originals_ns/ --tgt xml_translationese/es/translations_from_en_n/ --outdir extracted/

# make sure the txt files have txt extension (required by LFAligner: see /home/maria/tools/aligner/LF_aligner_setup.txt)
# edit the folder appropriately
# sh rename2align.sh

# generate the bash script for LFAligner /home/maria/tools/aligner/deen_europarl-uds.sh for each language pair, e.g.
python3 get_aligner_bash.py --indir extracted/enes/ --lpair enes

# generate domain-specific dicts from parallel termbases from
# https://iate.europa.eu/download-iate
# put the dict into
python3 get_iata_bilingual_dictionaries.py --indir glossaries/iate_dicts/ --lpair ende
python3 get_iata_bilingual_dictionaries.py --indir glossaries/iate_dicts/ --lpair deen
python3 get_iata_bilingual_dictionaries.py --indir glossaries/iate_dicts/ --lpair enes
python3 get_iata_bilingual_dictionaries.py --indir glossaries/iate_dicts/ --lpair esen

# combine rename Europarl-UdS-2018 and IATA parallel term bases for EP
# move the resulting dictionaries (glossaries/2hunalign/) to your LF Aligner: aligner/scripts/hunalign/data/
python3 concat_dic.py --old glossaries/biling_dicts2018/ --new glossaries/iate2hunalign/ --res YOUR_PATH_TO_LFAligner

# install LF Aligner locally (e.g. to /home/maria/tools/aligner/, download from https://sourceforge.net/projects/aligner/)
# put the scripts to the aligner root folder and run:
sh /home/maria/tools/aligner/deen_europarl-uds.sh
sh /home/maria/tools/aligner/ende_europarl-uds.sh
sh /home/maria/tools/aligner/esen_europarl-uds.sh
sh /home/maria/tools/aligner/enes_europarl-uds.sh

# get a wide-tabled versions of the corpus from the temp folders output by LF Aligner
# columns: 'sdoc_id', 'sseg_id', 'sseg', 'tseg', 'hunalign_qua'
# use 0.3 cutoff for es-en and de-en and 0.5 for en-es and en-de
python3 get_raw_aligned.py --indir lf_aligned_xls/esen_align_2023.10.29/ --lpair esen --meta data/raw_aligned/meta/ --docsize 0 --cutoff 0.3
python3 get_raw_aligned.py --indir lf_aligned_xls/enes_align_2023.10.28/ --lpair enes --meta data/raw_aligned/meta/ --docsize 0 --cutoff 0.5
python3 get_raw_aligned.py --indir lf_aligned_xls/deen_align_2023.10.28/ --lpair deen --meta data/raw_aligned/meta/ --docsize 0 --cutoff 0.3
python3 get_raw_aligned.py --indir lf_aligned_xls/ende_align_2023.10.28/ --lpair ende --meta data/raw_aligned/meta/ --docsize 0 --cutoff 0.5