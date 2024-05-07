"""
23 Oct 2023
replicating the last steps in Jose Martinez's pipeline https://github.com/chozelinek/europarl
to get raw doc- and sent-aligned corpora for EN-DE, DE-EN and EN-ES
the output from the previous steps are added in xml_translationese/ folder

DEEN
python3 extract_interventions_mk.py --src data/xml_translationese/de/originals_ns/ --tgt data/xml_translationese/en/translations_from_de_n/ --outdir data/

ENDE
python3 extract_interventions_mk.py --lpair ende --src _data/xml_translationese/en/originals_ns/ --tgt _data/xml_translationese/de/translations_from_en_n/ --outdir _data/
"""

import argparse
import os
from os.path import join
from lxml import etree
import codecs


def create_dic(f):
    ids = {}
    tree = etree.parse(f)
    root = tree.getroot()
    for intervention in root.iter('intervention'):
        id = intervention.get('id')
        sentences = []
        for sent in intervention.iter('s'):
            sentences.append(sent.text.lstrip())
        ids[id] = sentences
    for key, value in sorted(ids.items()):  # can this be done earlier?
        if not value:  # len(value)==0
            del ids[key]

    return ids


def read_files(source_path, target_path, interventions, tlang=None, slang=None):
    trg_filenames = sorted(os.listdir(target_path))

    counter = 0
    for trg_filename in trg_filenames:
        filename = trg_filename.split('.')[0]
        # You need to change the lang based on your src/tgt lang
        trg_file = join(target_path, filename + f'.{tlang.upper()}.xml')
        src_file = join(source_path, filename + f'.{slang.upper()}.xml')
        # print(trg_file, src_file)
        with open(trg_file, 'r') as trg, open(src_file, 'r') as src:
            dict_trg = create_dic(trg)
            dict_src = create_dic(src)
            for key in dict_trg.keys():
                if key in dict_src.keys():
                    # You need to change the lang based on your src/tgt lang
                    with open(interventions + filename + '.' + key + f'.{slang}.txt', 'w') as src_w, \
                            open(interventions + filename + '.' + key + f'.{tlang}.txt', 'w') as trg_w:
                        for sent_src in dict_src[key]:
                            src_w.write(sent_src)  # write value-trg in trg output
                        for sent_trg in dict_trg[key]:
                            trg_w.write(sent_trg)  # write value-src in src output
                else:
                    counter += 1
                    print("Intervention " + key + " in file " + trg_file + " not in target!")
    print(f'Missing targets: {counter}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract interventions from EP-UdS sentence-aligned, translationese-filtered xml')
    parser.add_argument('--src', help='Path to the source language directory', required=True)
    parser.add_argument('--tgt', help='Path to the target language directory', required=True)
    parser.add_argument('--outdir', help='where to write doc pairs with the same id and different lang indices?',
                        required=True)
    parser.add_argument('--lpair', choices=['deen', 'ende', 'esen', 'enes'], help='which lpair')
    args = parser.parse_args()

    lang_outdir = f'{args.outdir}{args.lpair}/'
    os.makedirs(lang_outdir, exist_ok=True)
    read_files(args.src, args.tgt, lang_outdir, slang=args.lpair[:2], tlang=args.lpair[2:])
