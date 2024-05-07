"""
UPD: 15 Nov 2023
use raw column now available in data/feats_tabled/seg-450-1500.feats.tsv.gz for prompt engineering

29 Sept 2023
use extract/tabled/seg-450-1500.feats.tsv.gz to extract most_original and most_translated documents (classify/extremes)
with highlighted most_translated/most original segments and best_performing features in them

4 Oct: for consistency, it is better to keep most_translated ided at segment level!
returns 100 most_translated docs for DE, 91 docs for EN

python3 2_classify1/get_extreme_documents.py --store_item seg

"""

import numpy as np
import os
import sys
import pandas as pd
import argparse
import time
from datetime import datetime
from collections import Counter
from collections import defaultdict


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


def list_to_newline_sep(lst):
    return '\n'.join(lst)


def to_bitext(my_predictable, lang=None, master_table=None):
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


def write_filtered_extremes(ol_tgt_feats_df=None, store_item=None, lang=None, writeto=None):
    ol_tgt_feats_df = ol_tgt_feats_df.rename(columns={'typical': 'most_translated'})
    if store_item == 'seg':
        write_df = ol_tgt_feats_df
    else:
        # get very lean df grouped by doc_id, containing lists of document segments in raw_tok and seg_nums for most_translated segments
        # restore doc_id and seg_num (which should rather be seg_index)
        ol_tgt_feats_df['doc_id'] = ol_tgt_feats_df['seg_id'].apply(lambda x: x.split('-')[0])
        ol_tgt_feats_df['seg_idx'] = ol_tgt_feats_df['seg_id'].apply(lambda x: x.split('-')[1])
        print(ol_tgt_feats_df.head())
        # get a temp doc-level df with a column containing list of only most_translated segments
        temp_seg_df = ol_tgt_feats_df[ol_tgt_feats_df.most_translated == 'yes']
        temp_seg_df['seg_idx'] = temp_seg_df['seg_idx'].astype(int)

        temp_doc_df = temp_seg_df.groupby('doc_id', as_index=True).aggregate({'src_raw': lambda x: x.tolist(),
                                                                              'raw': lambda x: x.tolist(),
                                                                              'seg_idx': lambda x: (x - 1).tolist(),
                                                                              'most_translated': lambda x: x.tolist(),
                                                                              })
        temp_doc_df = temp_doc_df.drop(['most_translated', 'raw', 'src_raw'], axis=1)
        temp_doc_df = temp_doc_df.rename(columns={'seg_idx': 'most_translated'})

        ol_tgt_feats_df = ol_tgt_feats_df.drop(['most_translated', 'seg_idx'], axis=1)
        main_doc_df = ol_tgt_feats_df.groupby('doc_id', as_index=True).aggregate({'src_raw': lambda x: x.tolist(),
                                                                                  'raw': lambda x: x.tolist()})

        write_df = pd.concat([main_doc_df, temp_doc_df], axis=1)
        write_df = write_df.reset_index()

        # filter out rows where 'most_translated_seg_indices' has NaN
        print(write_df.shape)
        write_df = write_df[~write_df['most_translated'].isna()]
    print(write_df.head())
    print(write_df.shape)
    write_df.to_csv(f'{writeto}{lang}_most_translated_aligned_{store_item}_ol_tgt_feats.tsv.gz', sep='\t',
                    compression='gzip', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', help="", default='data/feats_tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--extremes', default='2_classify1/extremes/', help='path to lists of ids, doc and seg level')
    parser.add_argument('--store_item', choices=['doc', 'seg'], default='seg')
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--res', default='2_classify1/extremes/analyse_this/')
    parser.add_argument('--raw_input', default='3_prompting/input/')
    parser.add_argument('--logs', default='logs/classify1/')

    args = parser.parse_args()

    start = time.time()

    os.makedirs(args.res, exist_ok=True)
    os.makedirs(args.raw_input, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    # os.makedirs(args.pics, exist_ok=True)

    log_file = f'{args.logs}{sys.argv[0].split("/")[-1].split(".")[0]}.log'
    sys.stdout = Logger(logfile=log_file)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    master_table = pd.read_csv(args.table, sep='\t', compression='gzip')
    # print(master_table.head())
    print(master_table.columns.tolist())

    # add seg_id column
    master_table = master_table.astype({'doc_id': 'str', 'seg_num': 'str'})
    master_table['seg_id'] = master_table[["doc_id", "seg_num"]].apply(lambda x: "-".join(x), axis=1)  # this is bad! - not :

    for lang in ['de', 'en']:
        lang_tab = master_table[master_table['lang'] == lang]
        print(lang_tab.shape)
        # ids of 100 docs predicted with top and 100 docs predicted with bottom confidence
        selected_docs = [f for f in os.listdir(args.extremes) if f.startswith(f'{lang}_doc_')][0]
        doc_ids = [itm.strip() for itm in open(args.extremes + selected_docs, 'r').readlines()]
        print(doc_ids[:5])

        #  ids of 200 segs from those docs predicted with top and bottom confidence (prediction errors are filtered out)
        selected_segs = [f for f in os.listdir(args.extremes) if f.startswith(f'{lang}_seg_')][0]
        seg_ids = [itm.strip() for itm in open(args.extremes + selected_segs, 'r').readlines()]
        print(seg_ids[:5])
        # 10 best predictors for doc- and seg- levels
        # Doc-level
        # Best features for DE: ['addit', 'advcl', 'advmod', 'fin', 'mean_sent_wc', 'mhd', 'parataxis', 'pastv', 'poss', 'ttr']
        # Best features for EN: ['addit', 'compound', 'conj', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls', 'ppron']
        #
        # Seg-level
        # Best features for DE: ['addit', 'advmod', 'caus', 'fin', 'nmod', 'nnargs', 'parataxis', 'pastv', 'poss', 'ttr']
        # Best features for EN: ['addit', 'advers', 'advmod', 'compound', 'conj', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls']
        # as of 7 Dec 2023  15 feats
        feat_dict = {'de': {
            'doc': ['addit', 'advcl', 'advmod', 'caus', 'ccomp', 'fin', 'mdd', 'mean_sent_wc', 'mhd', 'nmod',
                    'parataxis', 'pastv', 'poss', 'simple', 'ttr'],
            'seg': ['addit', 'advcl', 'advmod', 'caus', 'fin', 'iobj', 'mdd', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs',
                    'parataxis', 'pastv', 'poss', 'ttr']},

             'en': {'doc': ['addit', 'advcl', 'advers', 'compound', 'conj', 'demdets', 'fin', 'mean_sent_wc',
                            'mhd', 'nmod', 'nnargs', 'numcls', 'obl', 'ppron', 'sconj'],
                    'seg': ['addit', 'advcl', 'advers', 'advmod', 'compound', 'conj', 'mean_sent_wc', 'mhd',
                            'negs', 'nmod', 'nnargs', 'numcls', 'obl', 'pastv', 'ppron']}
                     }
        # if lang == 'de':
        #     docs_predictors = ['addit', 'advcl', 'advmod', 'fin', 'mean_sent_wc', 'mhd', 'parataxis', 'pastv', 'poss', 'ttr']
        #     seg_from_confident_docs_predictors = ['addit', 'advmod', 'caus', 'fin', 'nmod', 'nnargs', 'parataxis', 'pastv', 'poss', 'ttr']
        #     best_indicators = list(set(docs_predictors).union(set(seg_from_confident_docs_predictors)))
        # else:
        #     docs_predictors = ['addit', 'compound', 'conj', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls', 'ppron']
        #     seg_from_confident_docs_predictors = ['addit', 'advers', 'advmod', 'compound', 'conj', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls']
        #     best_indicators = list(set(docs_predictors).union(set(seg_from_confident_docs_predictors)))
        #     # alphabetise
        #     best_indicators = sorted(best_indicators)

        best_indicators = list(set(feat_dict[lang]['doc']).union(set(feat_dict[lang]['seg'])))

        print(lang, len(best_indicators), best_indicators)

        # limit data to most confidently predicted docs
        predictable_docs = lang_tab[lang_tab['doc_id'].isin(doc_ids)]

        # add typical column and add org or tgt (I filtered out prediction errors in classifier.py)
        # predictable_docs['typical'] = predictable_docs['seg_id'].apply(lambda x: 'yes' if x in seg_ids else 'no')
        predictable_docs.insert(0, 'typical', 'no')
        predictable_docs.loc[predictable_docs['seg_id'].isin(seg_ids), 'typical'] = 'yes'

        # drop features outside the union of doc- and seg-level top 10
        meta = ['doc_id', 'seg_id', 'seg_num', 'raw', 'ttype', 'corpus', 'direction', 'lang', 'raw_tok', 'wc_lemma',
                'sents', 'typical']
        keep_them = meta + best_indicators

        # predictable_docs has ol features!!!
        predictable_docs_best = predictable_docs.loc[:, keep_them]

        # print(predictable_docs_best.head())
        # print(predictable_docs_best.shape)
        # # docs 200, tra_segs= ; nontra_segs=
        # print(len(set(predictable_docs_best.doc_id.tolist())))
        # print(list(set(predictable_docs_best.doc_id.tolist()))[:3])
        #
        # nontra_predictable_docs_best = predictable_docs_best[predictable_docs_best.ttype == 'source']
        # print(lang)
        # print(nontra_predictable_docs_best.shape[0])
        # tra_predictable_docs_best = predictable_docs_best[predictable_docs_best.ttype == 'target']
        # print(tra_predictable_docs_best.shape[0])
        # exit()

        # predictable are targets and non-translations in this lang! I need to get sources for the targets
        aligned_very_translated_with_bestfeats = to_bitext(predictable_docs_best, lang=lang, master_table=master_table)
        aligned_very_translated_with_olfeats = to_bitext(predictable_docs, lang=lang, master_table=master_table)

        print(len(set(predictable_docs_best.doc_id.tolist())))
        print(list(set(predictable_docs_best.doc_id.tolist()))[:3])

        print(aligned_very_translated_with_bestfeats.columns.tolist())
        print(aligned_very_translated_with_olfeats.columns.tolist())

        # mark src as typical org if they are in the selection
        if lang == 'de':
            typical_org_segs_as_src = [f for f in os.listdir(args.extremes) if f.startswith('en_seg_')][0]
        else:
            typical_org_segs_as_src = [f for f in os.listdir(args.extremes) if f.startswith('de_seg_')][0]

        slang_specific_org_segs_ids = [itm.strip() for itm in
                                       open(args.extremes + typical_org_segs_as_src, 'r').readlines()]

        aligned_very_translated_with_bestfeats['org_typical'] = aligned_very_translated_with_bestfeats[
            'src_seg_id'].apply(lambda x: 'yes' if x in slang_specific_org_segs_ids else 'no')
        src_yes_count = aligned_very_translated_with_bestfeats['org_typical'].value_counts().get('yes', 0)
        tgt_yes_count = aligned_very_translated_with_bestfeats['typical'].value_counts().get('yes', 0)
        print(f'\n===== {lang.upper()} =====')
        print(
            f'Typical slang segs (> 95% SVM probability) as originals in confidently-predicted docs: {src_yes_count / aligned_very_translated_with_bestfeats.shape[0] * 100:.2f}% ({src_yes_count} of {aligned_very_translated_with_bestfeats.shape[0]})')
        print(
            f'Ratio of machine-recognisable translated segs (> 95% SVM probability) to all segs in confidently predicted translated docs: '
            f'{tgt_yes_count / aligned_very_translated_with_bestfeats.shape[0] * 100:.2f}% ({tgt_yes_count} of {aligned_very_translated_with_bestfeats.shape[0]})')

        # re-order columns
        my_columns = ['src_raw', 'raw', 'typical', 'org_typical'] + best_indicators

        aligned_very_translated_with_bestfeats = aligned_very_translated_with_bestfeats[my_columns]

        # get seg_id as column
        aligned_very_translated_with_bestfeats = aligned_very_translated_with_bestfeats.reset_index()

        aligned_very_translated_sorted = aligned_very_translated_with_bestfeats.sort_values(
            by=['typical', 'org_typical'], ascending=[False, False])
        print(aligned_very_translated_with_bestfeats.columns.tolist())

        aligned_very_translated_sorted.to_csv(f'{args.res}{lang}_aligned_typical-tgt-feats_for_manual_analysis.tsv',
                                              sep='\t', index=False)
        orgs = predictable_docs_best[predictable_docs_best.ttype == 'source']
        orgs_sorted = orgs.sort_values(by='typical', ascending=False)
        orgs_sorted.to_csv(f'{args.res}{lang}_typical-org-feats_for_manual_analysis.tsv', sep='\t', index=False)

        print(f'==== DONE with the dataset for manual analysis ======')
        print('*** Building bitext based on raw text and ol feats to generate custom instructions ***')

        # same for the df with all feats -- I still need to reduce this df to 100 doc pairs!
        aligned_very_translated_with_olfeats['org_typical'] = aligned_very_translated_with_olfeats[
            'src_seg_id'].apply(lambda x: 'yes' if x in slang_specific_org_segs_ids else 'no')

        # write a bitext with ol tgt feats
        write_filtered_extremes(ol_tgt_feats_df=aligned_very_translated_with_olfeats, store_item=args.store_item,
                                lang=lang, writeto=args.raw_input)

        endtime_tot = time.time()
        print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
