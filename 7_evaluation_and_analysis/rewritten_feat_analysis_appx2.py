"""
5 Mar 2024
I need Appendix2 table for EAMT submission that will compare how the feature frequencies were changed in re-writing
using an upward arrow for increased freqs compared to expected and * for no statistical significance

names of the approaches self-guided (modes: min, detailed), feature-based (modes: min, detailed)

on all features (maybe make a column if the feature is among the best) once on HT (in analysis res)

python3 7_evaluation_and_analysis/rewritten_feat_analysis_appx2.py --lose_bypassed --thres_type ratio2.5

"""

import sys
import os
import argparse
import time
from datetime import datetime
from scipy.stats import mannwhitneyu
from scipy.stats.mstats import spearmanr
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from collections import defaultdict

import pandas as pd


def add_weights(weights_dir=None, tlang=None, lang_df=None):
    my_weights = [i for i in os.listdir(weights_dir) if f'{tlang}_svm-weights_seg_' in i]
    new_cols = []
    for i in my_weights:
        lang_svm_df = None
        if i.startswith('allfeats'):
            lang_svm_df = pd.read_csv(f'{args.svm_weights_dir}{i}', sep='\t')
            print(i)

            # Round the values in columns to 3 decimal places
            lang_svm_df = lang_svm_df.drop(['weight'], axis=1)
            lang_svm_df['abs_weight'] = lang_svm_df['abs_weight'].round(3)

            lang_svm_df = lang_svm_df.set_index('feature')
            lang_svm_df = lang_svm_df.rename(columns={'abs_weight': 'all_weights'})

        elif i.startswith('optfeats'):  # for the best performing feature set

            lang_svm_df = pd.read_csv(f'{args.svm_weights_dir}{i}', sep='\t')
            # Round the values in columns to 3 decimal places
            lang_svm_df = lang_svm_df.drop(['weight'], axis=1)
            lang_svm_df['abs_weight'] = lang_svm_df['abs_weight'].round(3)

            lang_svm_df = lang_svm_df.set_index('feature')
            lang_svm_df = lang_svm_df.rename(columns={'abs_weight': 'opt_weights'})

        new_cols.append(lang_svm_df)

    lang_df_updated = pd.concat([lang_df] + new_cols, axis=1)  # adding columns abs_weights and weights
    lang_df_updated = lang_df_updated.reset_index()

    return lang_df_updated


def ht_univariate(feature_lst, df0=None, col_name=None, lang=None):
    res_collector = defaultdict(list)  # key = columns
    lang_df0 = df0[df0.lang == lang]
    for i in feature_lst:
        res_collector['feature'].append(i)
        org_lst = lang_df0.loc[lang_df0.ttype == 'source'][i].tolist()  # source is actually the original here
        tgt_lst = lang_df0.loc[lang_df0.ttype == 'target'][i].tolist()

        org = np.average(org_lst)
        tgt = np.average(tgt_lst)
        res_collector['org_mean'].append(org.round(3))
        res_collector['tgt_mean'].append(tgt.round(3))

        # take a random sample of size 2000 from each category
        # H0 :The distributions of the two groups are equal
        U, p = mannwhitneyu(tgt_lst, org_lst, alternative='two-sided')  # unpaired samples
        if p < 0.05:  # reject H0
            res_collector[f'MannWhit'].append('different')
        else:
            res_collector[f'MannWhit'].append('--')

    res_df = pd.DataFrame(res_collector)
    res_df[col_name] = res_df.apply(lambda row: r'$\uparrow$' if row['org_mean'] < row['tgt_mean'] else r'$\downarrow$',
                                    axis=1)

    # Add an asterisk to col_name if the corresponding value in "MannWhit" is '--'
    res_df.loc[res_df['MannWhit'] == '--', col_name] = res_df.loc[res_df['MannWhit'] == '--', col_name] + '*'
    res_df = res_df.drop(['MannWhit', 'tgt_mean'], axis=1)
    res_df = res_df.rename(columns={'org_mean': 'expected'})

    return res_df


def rewrit_univariate(feature_lst=None, feat_rewr=None, feat_org=None, col_name=None):
    res_collector = defaultdict(list)
    print(col_name)
    print(feat_org.shape)
    print(feat_rewr.shape)

    for i in feature_lst:
        res_collector['feature'].append(i)
        rewr_lst = feat_rewr[i].tolist()
        org_lst = feat_org[i].tolist()

        org = np.average(org_lst)
        rewr = np.average(rewr_lst)
        if len(org_lst) == 0 or len(rewr_lst) == 0:
            print(i)
            print(f'Len org: {len(org_lst)} {org_lst[:5]}')
            print(f'Len rewr: {len(rewr_lst)} {rewr_lst[:5]}')
            exit()

        res_collector['org_mean'].append(org.round(3))

        res_collector['rewr_mean'].append(rewr.round(3))

        U, p = mannwhitneyu(rewr_lst, org_lst, alternative='two-sided')  # unpaired samples
        if p < 0.05:  # reject H0
            res_collector[f'MannWhit'].append('different')
        else:
            res_collector[f'MannWhit'].append('--')

    res_df = pd.DataFrame(res_collector)
    res_df[col_name] = res_df.apply(
        lambda row: r'$\uparrow$' if row['org_mean'] < row['rewr_mean'] else r'$\downarrow$',
        axis=1)

    # Add an asterisk to col_name if the corresponding value in "MannWhit" is '--'
    res_df.loc[res_df['MannWhit'] == '--', col_name] = res_df.loc[res_df['MannWhit'] == '--', col_name] + '*'
    res_df = res_df.drop(['org_mean', 'MannWhit', 'rewr_mean'], axis=1)

    return res_df


def reduce_to_extreme_docs(df0=None, tops_by_lang=None):
    two_langs = []
    for my_lang in ['de', 'en']:
        fh = [f for f in os.listdir(tops_by_lang) if f.startswith(f'{my_lang}_doc_')]
        my_extremes_doc_ids = [i.strip() for i in open(f'{tops_by_lang}{fh[0]}', 'r').readlines()]
        try:
            smaller_lang = df0[df0['doc_id'].isin(my_extremes_doc_ids)]
        except KeyError:
            smaller_lang = df0[df0['item'].isin(my_extremes_doc_ids)]

        smaller_lang = smaller_lang[smaller_lang['raw'].apply(lambda x: len(str(x).split()) > 8)]

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


def my_spearman_rho(trues, predictions):
    return spearmanr(trues, predictions)[0]


def scaled_svm_f1(x, y):
    clf = SVC(kernel='linear', C=1.0, random_state=45, class_weight='balanced', probability=False)
    x = x.reshape(-1, 1)

    sc = StandardScaler()
    x = sc.fit_transform(x)

    scoring = ['f1_macro']
    cv_scores = cross_validate(clf, x, y, cv=5, scoring=scoring, n_jobs=-1)

    scores_f = cv_scores['test_f1_macro']

    return np.average(np.array(scores_f)), np.std(np.array(scores_f))


def logreg_f1(x, y):
    logreg = LogisticRegression()
    x = x.reshape(-1, 1)
    scoring = ['f1_macro']
    cv_scores = cross_validate(logreg, x, y, cv=5, scoring=scoring, n_jobs=-1)

    scores_f = cv_scores['test_f1_macro']

    return np.average(np.array(scores_f)), np.std(np.array(scores_f))


# Function to sample 2500 rows from each group
def sample_group(group, n=2500):
    return group.sample(min(n, len(group)))


def list_to_newline_sep(lst):
    return '\n'.join(lst)


def make_dirs(outdir, logsto, sub):
    os.makedirs(f"{outdir}", exist_ok=True)
    os.makedirs(f'{logsto}', exist_ok=True)
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{sub}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewritten_feats', default='data/rewritten/feats_tabled2/')
    parser.add_argument('--initial_feats', help='feats in HT', default='data/feats_tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--extremes', default='2_classify1/extremes/', help='path to lists of contrastive ids')
    parser.add_argument('--thres_type', choices=['ratio2.5', 'std2'], default='ratio2.5', required=True)
    parser.add_argument('--level', choices=['seg'], default='seg')
    parser.add_argument('--model', choices=['gpt-4'], default='gpt-4')
    parser.add_argument('--sample', choices=['contrastive'], default='contrastive')
    parser.add_argument('--lose_bypassed', action="store_true",
                        help='Do you want to excluse segs that did not make it to re-writing pipeline? classify/bypassed/{lang}_...txt')
    # parser.add_argument('--best_selection', action='store_true',
    #                     help='Do you want to save a smaller df with the best features only?')
    parser.add_argument('--outdir', default='7_evaluation_and_analysis/res/')
    parser.add_argument('--logsto', default=f'logs/rewritten/')
    args = parser.parse_args()

    start = time.time()
    # as of 7 Dec 2023  15 feats
    best_dict = {'de': {
        'doc': ['addit', 'advcl', 'advmod', 'caus', 'ccomp', 'fin', 'mdd', 'mean_sent_wc', 'mhd', 'nmod', 'parataxis',
                'pastv', 'poss', 'simple', 'ttr'],
        'seg': ['addit', 'advcl', 'advmod', 'caus', 'fin', 'iobj', 'mdd', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs',
                'parataxis', 'pastv', 'poss', 'ttr']},
        'en': {'doc': ['addit', 'advcl', 'advers', 'compound', 'conj', 'demdets', 'fin', 'mean_sent_wc', 'mhd',
                       'nmod', 'nnargs', 'numcls', 'obl', 'ppron', 'sconj'],
               'seg': ['addit', 'advcl', 'advers', 'advmod', 'compound', 'conj', 'mean_sent_wc', 'mhd', 'negs',
                       'nmod', 'nnargs', 'numcls', 'obl', 'pastv', 'ppron']}
    }

    setup2approach = {'translated_min': 'translated',
                      'self-guided_min': 'self-guided',
                      'self-guided_detailed': 'self-guided',
                      'feature-based_min': 'feature-based',
                      'feature-based_detailed': 'feature-based'}
    setup2mode = {'translated_min': 'min',
                  'self-guided_min': 'min',
                  'self-guided_detailed': 'detailed',
                  'feature-based_min': 'min',
                  'feature-based_detailed': 'detailed'
                  }

    make_dirs(args.outdir, args.logsto, sub=args.thres_type)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    my_lang_dfs = []
    for tlang in ['de', 'en']:

        feats_europ = pd.read_csv(args.initial_feats, sep='\t')
        meta_all = ['doc_id', 'lang', 'ttype', 'seg_num', 'corpus', 'direction', 'wc_lemma', 'raw_tok', 'raw', 'sents']
        muted = ['relcl', 'case']
        feats_europ = feats_europ.drop(muted, axis=1)
        feats = [i for i in feats_europ.columns.tolist() if i not in meta_all]

        # reduce to extreme docs to have a fair comparison
        # + throw out segs <= 8 words from org and HT categories
        print(feats_europ.shape)
        feats_europ = reduce_to_extreme_docs(df0=feats_europ, tops_by_lang=args.extremes)
        print(feats_europ.shape)

        lang_df = ht_univariate(feats, df0=feats_europ, col_name='HT', lang=tlang)  # expected and HT processed inside
        # Add a new column with 1 if feature is in the 15 most informative translationese indicators
        best_feats = best_dict[tlang]['seg']
        lang_df['best'] = lang_df['feature'].isin(best_feats).astype(int)

        lang_df = lang_df.set_index('feature')

        lang_df_updated = lang_df.sort_values(by=['best', 'feature'], ascending=[False, True])

        this_lang_varieties_dfs = []

        this_lang_org = feats_europ[(feats_europ.lang == tlang) & (feats_europ.ttype == 'source')]  # expected

        for setup in setup2approach:
            if setup.startswith('feature-based_'):
                with_thres = f"{setup}_{args.thres_type}"
            else:
                with_thres = setup
            try:
                df = pd.read_csv(f'{args.rewritten_feats}{args.thres_type}/gpt4_temp0.7_{with_thres}_rewritten_feats.tsv.gz', sep='\t',
                                 compression='gzip')
            except FileNotFoundError:
                continue

            meta = ['doc_id', 'lang', 'ttype', 'seg_num', 'seg_id', 'corpus', 'direction', 'wc_lemma', 'raw_tok',
                    'sents', 'rewritten']
            feats = [i for i in df.columns.tolist() if i not in meta]

            this_df = df[df.lang == tlang]
            this_df = this_df.astype({'seg_id': 'str'})
            this_df['doc_id'] = this_df['seg_id'].apply(lambda x: str(x).split(':')[0])

            if args.lose_bypassed:
                lose_them = [i.strip() for i in
                             open(f'data/rewritten/curated/lose_segids/{args.thres_type}/{tlang}/shorts_only_{setup}.ids',
                                  'r').readlines()]
                this_df = this_df[~this_df['seg_id'].isin(lose_them)]

            # df with one column with arrows named after the approach_mode indexed on feature
            var_df = rewrit_univariate(feature_lst=feats, feat_rewr=this_df,
                                       feat_org=this_lang_org, col_name=f'{setup2approach[setup]}_{setup2mode[setup]}')

            var_df = var_df.set_index('feature')

            this_lang_varieties_dfs.append(var_df)

        this_lang_ol = pd.concat([lang_df_updated] + this_lang_varieties_dfs, axis=1)

        # Create a new DataFrame with {tlang} as values and 'lang' as the index
        new_row = pd.DataFrame({col: [tlang] for col in this_lang_ol.columns}, index=[''])
        # Concatenate the new row DataFrame with the existing DataFrame
        lang_res_df = pd.concat([new_row, this_lang_ol])

        my_lang_dfs.append(lang_res_df)

    tot_res = pd.concat(my_lang_dfs, axis=1)
    tot_res = tot_res.reset_index()
    print(tot_res.head())

    tot_res.to_csv(f'{args.outdir}{args.thres_type}_feature_relevance_appx2.tsv', sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
