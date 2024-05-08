"""
16 Mar 2024
This script generates a table with 7 aligned versions of the same content, i.e inputs to the rewriting pipeline and all 5 outputs.

It produce a table for each TL with columns:
['seg_id', 'source', 'translation', 'self-guided_min', 'self-guided_detailed', 'feature-based_min', 'feature-based_detailed', 'translated_min']
and raw strings as input for COMET, manual annotation and feature extraction

python3 get_multi_parallel_tsv.py --megaouts 4_prompting/output/ --outdir data/rewritten/asis/ --thres_type ratio2.5

"""

import sys
import os
import argparse
import time
from datetime import datetime
import pandas as pd


def write_this_lang_setup_skip_ids(rewr_df=None, match_to=None, saveas=None):
    count = 0
    with open(saveas, 'w') as outf:
        for index, row in rewr_df.iterrows():
            if row['rewritten'] in match_to:
                count += 1
                outf.write(row['seg_id'] + '\n')
    return count


def reduce_to_extreme_docs(df0=None, tops_by_lang=None):
    two_langs = []
    for my_lang in ['de', 'en']:
        fh = [f for f in os.listdir(tops_by_lang) if f.startswith(f'{my_lang}_doc_')]

        my_extremes_doc_ids = [i.strip() for i in open(f'{tops_by_lang}{fh[0]}', 'r').readlines()]
        print(my_extremes_doc_ids[-5:])
        try:
            smaller_lang = df0[df0['doc_id'].isin(my_extremes_doc_ids)]
        except KeyError:
            smaller_lang = df0[df0['item'].isin(my_extremes_doc_ids)]
        two_langs.append(smaller_lang)
    smaller_df = pd.concat(two_langs, axis=0)

    return smaller_df


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


def to_bitext(my_predictable, master_table=None, lang=None):
    # add sources for targets, if possible
    tgts = my_predictable[my_predictable.ttype == 'target']

    tgt_seg_ids = tgts.seg_id.tolist()
    # replace TR_DE_EN_011846:1 with ORG_WR_DE_EN_011846:1
    if lang == 'en':
        src_seg_ids = [i.replace('TR_DE_EN_', 'ORG_WR_DE_EN_') for i in tgt_seg_ids]
    else:
        src_seg_ids = [i.replace('TR_EN_DE_', 'ORG_WR_EN_DE_') for i in tgt_seg_ids]

    srcs = master_table[master_table['seg_id'].isin(src_seg_ids)]
    print(srcs.shape)

    srcs = srcs.rename(columns={'raw_tok': 'src_tok', 'seg_id': 'src_seg_id', 'raw': 'src_raw'})
    # keep only two columns for src_lang part
    srcs_slice = srcs[['src_raw', 'src_seg_id']]
    assert srcs_slice.shape[0] == tgts.shape[0], 'Huston, we have got problems!'

    # create matching index columns
    srcs_slice.insert(0, 'seg_id', tgt_seg_ids)
    srcs_slice = srcs_slice.set_index('seg_id')

    tgts = tgts.set_index('seg_id')

    out = pd.concat([srcs_slice, tgts], axis=1)

    return out


setup2approach = {
    'seg_self-guided_min': 'self-guided',
    'seg_self-guided_detailed': 'self-guided',
    'seg_feature-based_min': 'feature-based',
    'seg_feature-based_detailed': 'feature-based',
    'seg_translated_min': 'translated'
}
setup2mode = {
    'seg_self-guided_min': 'min',
    'seg_self-guided_detailed': 'detailed',
    'seg_feature-based_min': 'min',
    'seg_feature-based_detailed': 'detailed',
    'seg_translated_min': 'min'
}


def make_dirs(outdir, logsto, sub):
    os.makedirs(f"{outdir}/", exist_ok=True)
    os.makedirs(f'{logsto}/', exist_ok=True)
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{sub}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--megaouts', help='raw outputs of the rewriting pipeline', default='4_prompting/output/')
    parser.add_argument('--level', choices=['seg'], default='seg')
    parser.add_argument('--thres_type', choices=['ratio2.5', 'std2'], default='ratio2.5')
    parser.add_argument('--model', choices=['gpt-4'], default='gpt-4')
    parser.add_argument('--sample', choices=['contrastive'], default='contrastive')
    parser.add_argument('--outdir', default='data/rewritten/asis/')
    parser.add_argument('--logsto', default=f'logs/rewritten/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(args.outdir, args.logsto, sub=args.thres_type)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    for tlang in ['de', 'en']:
        if tlang == 'de':
            slang = 'en'
        else:
            slang = 'de'
        my_lang_texts = []
        # intab is the output from 2_classify1/get_extreme_documents.py
        intab = f'4_prompting/input/{tlang}_most_translated_aligned_seg_ol_tgt_feats.tsv.gz'
        this_lang_wide = pd.read_csv(intab, sep='\t', compression='gzip',
                                     usecols=['src_seg_id', 'src_raw', 'raw'])
        this_lang_wide = this_lang_wide.rename(
            columns={'src_seg_id': 'seg_id', 'src_raw': 'source', 'raw': 'translation'})
        this_lang_wide = this_lang_wide.set_index('seg_id')
        print(this_lang_wide.head())

        my_lang_texts.append(this_lang_wide)

        translations = this_lang_wide.translation.tolist()
        print(len(translations))

        for i in setup2approach:
            this_df = pd.read_csv(f'{args.megaouts}{args.model}/{args.thres_type}/{tlang}_{i}.tsv.gz', sep='\t',
                                  compression='gzip', usecols=['seg_id', 'lang', 'rewritten'])

            col_name = f'{setup2approach[i]}_{setup2mode[i]}'
            this_df = this_df.rename(columns={'rewritten': col_name})

            this_df = this_df.set_index('seg_id')
            this_df = this_df.drop(['lang'], axis=1)
            print(this_df.head())
            print(this_df.shape)

            my_lang_texts.append(this_df)

        print(len(my_lang_texts))

        lang7col = pd.concat(my_lang_texts, axis=1)
        lang7col = lang7col.reset_index()
        lang7col = lang7col.dropna()
        print(lang7col.head())
        print(lang7col.shape)
        print(lang7col.columns.tolist())

        # Check for NaN values in the DataFrame
        has_nans = lang7col.isna().any().any()

        if has_nans:
            print("There are NaN values in the DataFrame.")
        else:
            print("No NaN values found in the DataFrame.")

        # Print the number of NaN values per column
        nan_counts = lang7col.isna().sum()
        print("Number of NaN values per column:")
        print(nan_counts)

        print(lang7col.columns.to_list())

        lang7col.to_csv(f'{args.outdir}{args.thres_type}_{tlang}_7aligned_{lang7col.shape[0]}segs.tsv', sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
