"""
29 Feb 2024
produce a table for the actual classifiers input --
built from most_translated segments from 100 most translated documents vs

the segment ids from classify/extremes/de_seg... and en_seg... can be used
they contain ids like: ORG_WR_EN_DE_004910-23 (for non-translated segments) TR_DE_EN_004321-7

USAGE:
python3 get_pars.py --sample contrastive

"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables',
                        help="segs from top 100 documents by SVM probability with 60 feat vals, used in custom_instructions.py",
                        default='prompt/input/')
    # parser.add_argument('--org_table', default='extract/tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--org_table', default='extract/tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--sample', choices=['contrastive', '0'],
                        help='run stat analysis on a sample instead of all',
                        default='contrastive')
    parser.add_argument('--extremes', default='classify/extremes/', help='path to lists of ids, doc and seg level')
    parser.add_argument('--stats', default='stats/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()

    start = time.time()
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')

    os.makedirs(args.stats, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    log_file = f'{args.logs}{formatted_datetime.split("_")[0]}_{sys.argv[0].split("/")[-1].split(".")[0]}.log'

    sys.stdout = Logger(logfile=log_file)

    print(f"\nRun date, UTC: {formatted_datetime}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    res_collector = defaultdict(list)

    # use the original table! the wide prompt/input/ files are NOT filtered in any way the filtering happens at the re-writing stage
    # use raw_tok column for any descriptive stats -- this is what we use for normalisation
    master_table = pd.read_csv(args.org_table, sep='\t', compression='gzip', usecols=['doc_id', 'raw', 'raw_tok', 'lang', 'ttype'])
    master_table = master_table[~master_table['raw_tok'].apply(lambda x: isinstance(x, float))]

    for tlang in ['de', 'en']:
        if tlang == 'de':
            slang = 'en'
        else:
            slang = 'de'

        lang_tab = master_table[master_table['lang'] == tlang]

        print(lang_tab.shape)
        # ids of 100 docs predicted with top and 100 docs predicted with bottom confidence
        selected_docs = [f for f in os.listdir(args.extremes) if f.startswith(f'{tlang}_doc_')][0]
        doc_ids = [itm.strip() for itm in open(args.extremes + selected_docs, 'r').readlines()]

        print(f'Getting parameters for the translated category in {tlang}...')
        tra_lang_tab = lang_tab[lang_tab['ttype'] == 'target']

        if args.sample == 'contrastive':
            # limit data to most confidently predicted docs
            predictable_tra = tra_lang_tab[tra_lang_tab['doc_id'].isin(doc_ids)]
            print(predictable_tra.shape)
            predictable_tra0 = predictable_tra[predictable_tra['raw'].apply(lambda x: len(str(x).split()) > 8)]
            print(tlang, predictable_tra.shape[0] - predictable_tra0.shape[0])

            tgt_docs = len(list(set(predictable_tra0.doc_id.tolist())))
            segs_lst = predictable_tra0.raw_tok.tolist()
            tgt_segs = len(segs_lst)
            tgt_wc = sum([len(i.split()) for i in segs_lst])
        else:  # assume all human targets into this lang

            tgt_docs = len(list(set(tra_lang_tab.doc_id.tolist())))
            segs_lst = tra_lang_tab.raw_tok.tolist()
            tgt_segs = len(segs_lst)
            # print(tgt_segs)
            # segs_lst = [i for i in segs_lst if not isinstance(i, float)]
            # tgt_segs = len(segs_lst)
            # print(tgt_segs)
            tgt_wc = sum([len(i.split()) for i in segs_lst])

        tgt_wc_mean = np.mean([len(i.split()) for i in segs_lst])
        tgt_wc_std = np.std([len(i.split()) for i in segs_lst])

        res_collector['lang'].append(tlang)
        res_collector['type'].append('translated')
        res_collector['docs'].append(tgt_docs)
        res_collector['segs'].append(tgt_segs)
        res_collector['wc'].append(tgt_wc)
        res_collector['seg_len+/-std'].append(f'{tgt_wc_mean:.1f}+/-{tgt_wc_std:.1f}')

        print('Working with the non-translated category... 100 most original category, not parallel with the translations in the other language')

        org_lang_tab = lang_tab[lang_tab['ttype'] == 'source']
        if args.sample == 'contrastive':
            predictable_org = org_lang_tab[org_lang_tab['doc_id'].isin(doc_ids)]
            # print(predictable_org.shape)
            predictable_org0 = predictable_org[predictable_org['raw'].apply(lambda x: len(str(x).split()) > 8)]
            print(tlang, predictable_org.shape[0] - predictable_org0.shape[0])

            src_docs = len(list(set(predictable_org0.doc_id.tolist())))
            src_segs_lst = predictable_org0.raw_tok.tolist()
            src_segs = len(src_segs_lst)
            src_wc = sum([len(i.split()) for i in src_segs_lst])
        else:
            src_docs = len(list(set(org_lang_tab.doc_id.tolist())))
            src_segs_lst = org_lang_tab.raw_tok.tolist()

            src_segs = len(src_segs_lst)
            src_wc = sum([len(i.split()) for i in src_segs_lst])

        src_wc_mean = np.mean([len(i.split()) for i in src_segs_lst])
        src_wc_std = np.std([len(i.split()) for i in src_segs_lst])

        res_collector['lang'].append(tlang)
        res_collector['type'].append('original')
        res_collector['docs'].append(src_docs)
        res_collector['segs'].append(src_segs)
        res_collector['wc'].append(src_wc)
        res_collector['seg_len+/-std'].append(f'{src_wc_mean:.1f}+/-{src_wc_std:.1f}')

    df = pd.DataFrame(res_collector)
    print(df)
    if args.sample == 'contrastive':
        df.to_csv(f'{args.stats}extreme_200docs.stats', sep='\t', index=False)
    else:
        df.to_csv(f'{args.stats}ol_3000docs.stats', sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
