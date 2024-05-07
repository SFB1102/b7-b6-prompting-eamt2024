'''
21 Sept 2023

run the classifier first to get SVM weights
The script outputs various statistics on each feature in the entire feature set (or in the best-performing selection if you pass the flag).
It is possible to look at the stats across entire doc- or seg-level datasets, or for contrastive (or random for seg-level only) samples.

compare feats values for sents and docs (inc. significance testing and <>
and formulate prompts

python3 analysis/feat_analysis.py --sample contrastive --level seg
python3 extract/feat_analysis.py --best_selection --sample contrastive --level seg
python3 extract/feat_analysis.py --sample contrastive --level doc
'''

import sys
import os
import argparse
import time
from datetime import datetime
from scipy.stats import mannwhitneyu, shapiro, bartlett, wilcoxon
from scipy.stats.mstats import spearmanr
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def reduce_to_extreme_docs(df0=None, tops_by_lang=None, lang=None):
    # two_langs = []
    # for my_lang in ['de', 'en']:
    fh = [f for f in os.listdir(tops_by_lang) if f.startswith(f'{lang}_doc_')]
    print(fh)
    my_extremes_doc_ids = [i.strip() for i in open(f'{tops_by_lang}{fh[0]}', 'r').readlines()]

    smaller_lang = df0[df0['doc_id'].isin(my_extremes_doc_ids)]
    smaller_lang = smaller_lang[smaller_lang['raw'].apply(lambda x: len(str(x).split()) > 8)]

    #     two_langs.append(smaller_lang)
    # smaller_df = pd.concat(two_langs, axis=0)

    return smaller_lang


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


# consider only the diff within two STD fof the spread in the translated category
def plot_abs_diff_bars(df=None, feats=None, my_lang=None, save_to=None):
    df0 = df.copy()
    df0 = df0.set_index('feature')
    # print(df0.head())
    selected_feat_rows_df = df0.loc[feats]

    # Calculate the difference between observed in translation and expected norm
    selected_feat_rows_df['diff'] = selected_feat_rows_df['tgt_mean'] - selected_feat_rows_df['org_mean']
    # # Calculate the difference between observed in translation and expected norm and compare it to tgt_std
    # diffs = selected_feat_rows_df['tgt_mean'] - selected_feat_rows_df['org_mean']
    # print(diffs)
    # # [x * 2 for x in original_list]
    # for diff, std2 in zip(diffs, [x * 2 for x in selected_feat_rows_df['tgt_std'].tolist()]):
    #     if abs(diff) >= std2:
    #         print(abs(diff), std2)
    #         selected_feat_rows_df['diff'] = selected_feat_rows_df['tgt_mean'] - selected_feat_rows_df['org_mean']
    #     else:
    #         selected_feat_rows_df['diff'] = None

    selected_feat_rows_df = selected_feat_rows_df.dropna(subset=['diff'])
    # print(selected_feat_rows_df)
    # print(selected_feat_rows_df.index)
    #
    # exit()

    # Create a bar plot for the difference
    plt.figure(figsize=(8, 8))
    # Set font properties globally
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'font.weight': 'normal'
    })

    # Width of the bars
    # bar_width = 0.50

    # Generate positions for the bars
    selected_feat_rows_df_sorted = selected_feat_rows_df.sort_values(by='diff')
    # , width=bar_width, palette='Paired'
    ax = sns.barplot(x=selected_feat_rows_df_sorted.index, y='diff', data=selected_feat_rows_df_sorted, color='forestgreen')

    # Rotate x-axis labels
    custom_labels = selected_feat_rows_df_sorted.index.tolist()
    ax.set_xticks(range(len(custom_labels)))
    ax.set_xticklabels(custom_labels, rotation=45)
    # Add vertical grid
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.3, color='gray')
    # Add labels and title
    if my_lang == 'de':
        my_lang = "German"
    else:
        my_lang = 'English'
    plt.xlabel('')
    plt.ylabel('Difference in normalised feature frequencies')
    plt.title(f'{my_lang} best predictors: Deviations from expected TL norm')
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    plt.savefig(save_to)
    plt.show()


def univariate(feature_lst, best=None, lang_df0=None, my_lang=None, picpath=None):
    res_collector = defaultdict(list)  # key = columns

    for i in feature_lst:
        res_collector['feature'].append(i)
        org_lst = lang_df0.loc[lang_df0.ttype == 'source'][i].tolist()  # source is actually the original here
        tgt_lst = lang_df0.loc[lang_df0.ttype == 'target'][i].tolist()

        org = np.average(org_lst)
        tgt = np.average(tgt_lst)
        res_collector['org_mean'].append(org.round(3))
        res_collector['org_std'].append(np.std(org_lst))
        res_collector['tgt_mean'].append(tgt.round(3))
        res_collector['tgt_std'].append(np.std(tgt_lst))

        for name, lst in zip(['org', 'tgt'], [org_lst, tgt_lst]):
            stat1, p1 = shapiro(lst)  # H0: a variable is normally distributed
            if p1 < 0.05:  # reject H0
                res_collector[f'{name}_shapiro'].append('--')
            else:
                res_collector[f'{name}_shapiro'].append('normal')

        stat2, p3 = bartlett(tgt_lst, org_lst)  # H0: variances are equal
        if p3 < 0.05:  # reject H0
            res_collector[f'variances'].append('--')
        else:
            res_collector[f'variances'].append('equal')

        # take a random sample of size 2000 from each category
        # H0 :The distributions of the two groups are equal
        U, p = mannwhitneyu(tgt_lst, org_lst, alternative='two-sided')  # unpaired samples
        if p < 0.05:  # reject H0
            res_collector[f'MannWhit'].append('different')
        else:
            res_collector[f'MannWhit'].append('--')

        f1, std = logreg_f1(lang_df0[i].values, lang_df0['ttype'].values)
        res_collector[f'F1_logreg'].append(f1.round(3))
        res_collector[f'F1_std_cv5'].append(f'+/-{std.round(3)}')

        f1_svm, _ = scaled_svm_f1(lang_df0[i].values, lang_df0['ttype'].values)
        res_collector[f'F1_svm'].append(f1_svm.round(3))

    res_df = pd.DataFrame(res_collector)
    res_df['compare'] = res_df.apply(lambda row: 'org < tgt' if row['org_mean'] < row['tgt_mean'] else '--', axis=1)

    # if 'rewritten' in args.table:
    #     plot_abs_diff_bars(df=res_df, feats=best, ttype='rewritten', my_lang=my_lang, save_to='pics/feats_diffs_contrastive_rewritten.png')
    # else:
    plot_abs_diff_bars(df=res_df, feats=best, my_lang=my_lang, save_to=picpath)

    return res_df


# Function to sample 2500 rows from each group
def sample_group(group, n=2500):
    return group.sample(min(n, len(group)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', default='extract/tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--svm_weights_dir', default='classify/res/weights/')
    parser.add_argument('--level', choices=['seg', 'doc'], default='doc')
    parser.add_argument('--sample', choices=['contrastive', 'random', '0'], help='run stat analysis on a sample instead of all',
                        default='contrastive')
    parser.add_argument('--best_selection', action='store_true',
                        help='Do you want to save a smaller df with the best features only?')
    parser.add_argument('--res', default='analysis/res/')
    parser.add_argument('--pics', default='analysis/pics/')
    parser.add_argument('--logs', default=f'analysis/logs/')
    args = parser.parse_args()

    start = time.time()
    # as of 8 Mar 2024  15 feats: no shorts!
    feat_dict = {'de': {'doc': ['addit', 'advcl', 'advmod', 'caus', 'ccomp', 'fin', 'mdd', 'mean_sent_wc', 'mhd', 'nmod', 'parataxis', 'pastv', 'poss', 'simple', 'ttr'],
                        'seg': ['addit', 'advcl', 'advmod', 'caus', 'fin', 'iobj', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'parataxis', 'pastv', 'poss', 'self', 'ttr']},
                 'en': {'doc': ['addit', 'advcl', 'advers', 'compound', 'conj', 'demdets', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls', 'obl', 'ppron', 'sconj'],
                        'seg': ['addit', 'advcl', 'advers', 'advmod', 'aux:pass', 'compound', 'conj', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls', 'obl', 'pastv']}
                 }

    # # as of 7 Dec 2023  15 feats
    # feat_dict = {'de': {'doc': ['addit', 'advcl', 'advmod', 'caus', 'ccomp', 'fin', 'mdd', 'mean_sent_wc', 'mhd', 'nmod', 'parataxis', 'pastv', 'poss', 'simple', 'ttr'],
    #                     'seg': ['addit', 'advcl', 'advmod', 'caus', 'fin', 'iobj', 'mdd', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'parataxis', 'pastv', 'poss', 'ttr']},
    #              'en': {'doc': ['addit', 'advcl', 'advers', 'compound', 'conj', 'demdets', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls', 'obl', 'ppron', 'sconj'],
    #                     'seg': ['addit', 'advcl', 'advers', 'advmod', 'compound', 'conj', 'mean_sent_wc', 'mhd', 'negs', 'nmod', 'nnargs', 'numcls', 'obl', 'pastv', 'ppron']}
    #              }
    # # as of 30 Nov
    # feat_dict = {'de': {'doc': ['addit', 'advcl', 'advmod', 'fin', 'mean_sent_wc', 'mhd', 'parataxis', 'pastv', 'poss', 'ttr'],
    #                     'seg': ['addit', 'advmod', 'caus', 'fin', 'nmod', 'nnargs', 'parataxis', 'pastv', 'poss', 'ttr']},
    #              'en': {'doc': ['addit', 'compound', 'conj', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls', 'ppron'],
    #                     'seg': ['addit', 'advers', 'advmod', 'compound', 'conj', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs', 'numcls']}
    #              }

    print(f'* DE {args.level} ({len(feat_dict["de"][args.level])})')
    print(f'* EN {args.level} ({len(feat_dict["en"][args.level])})')
    print(f'* shared: {set(feat_dict["de"][args.level]).intersection(set(feat_dict["en"][args.level]))}')

    os.makedirs(args.pics, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    log_file = f'{args.logs}{args.level}_univariate_{args.table.split("/")[-1].rsplit("_", 1)[0]}.log'
    os.makedirs(args.res, exist_ok=True)

    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    df = pd.read_csv(args.table, sep='\t', compression='gzip')

    # # we know from the classifier that there are 96 segments across the entire dataset with NaNs in the mdd column
    # df = df.dropna(subset=['mdd'])
    # print(df.shape)

    meta_all = ['doc_id', 'lang', 'ttype', 'seg_num', 'corpus', 'direction', 'wc_lemma', 'raw_tok', 'raw', 'sents']
    muted = ['relcl', 'case']

    df = df.drop(muted, axis=1)
    print(df.columns.tolist())
    print(df.shape)

    feats = [i for i in df.columns.tolist() if i not in meta_all]

    assert len(feats) == 58, f'Huston! Problems! N feats: {len(feats)}'
    # # how often I have more than 1 sent in a seg?
    # for lang in ['de', 'en']:
    #     this_df = df[(df.lang == lang) & (df.ttype == 'target')]
    #     count_not_1 = len(this_df[this_df['sents'] != 1])
    #     # DE translated: Number of rows where 'sents' is not 1: 2233 (6.2)
    #     # EN translated: Number of rows where 'sents' is not 1: 1110 (2.9)
    #     print(f"{lang.upper()} translated: Number of rows where 'sents' is not 1: {count_not_1} ({round(count_not_1/this_df.shape[0] * 100, 1)})")
    # exit()

    my_lang_dfs = []
    my_best_lang_dfs = []

    for lang in ['de', 'en']:
        this_df = df[df.lang == lang]

        # if 'seg' in args.table:
        if args.level == 'doc':
            # if 'rewritten' in args.table:
            #     print(this_df.head())
            #     this_df = this_df.astype({'seg_id': 'str'})
            #     this_df['doc_id'] = this_df['seg_id'].apply(lambda x: str(x).split('-')[0])
            #     this_df.insert(2, 'ttype', 'rewritten')
            #     meta = ['wc_lemma', 'seg_id']
            # else:
            print(this_df.head())

            meta = ['corpus', 'direction', 'wc_lemma', 'raw', 'raw_tok', 'sents']

            # this_df = this_df.astype({'doc_id': 'str', 'seg_id': 'str'})  # 'seg_num': 'str',
            this_df = this_df.drop(meta, axis=1)
            aggregate_functions = {col: 'mean' for col in feats}
            this_df = this_df.groupby('doc_id', as_index=False).aggregate({'lang': 'first',
                                                                           'ttype': 'first',
                                                                           **aggregate_functions
                                                                           })
            # this_df = this_df.rename(columns={"doc_id": "item"})
        else:  # assume seg
            # if 'rewritten' in args.table:
            #     print(this_df.head())
            #     this_df['doc_id'] = this_df['seg_id'].apply(lambda x: str(x).split('-')[0])
            #     this_df.insert(2, 'ttype', 'rewritten')
            #     # add the second category (orig from another table)
            #     tab_with_org = "data/tabled_feats/seg_500-1500_feats.tsv.gz"
            #     org_df = pd.read_csv(tab_with_org, sep='\t')
            #     org_df = org_df[org_df.ttype == 'source']
            #     print(org_df.columns.tolist())
            #     print(this_df.columns.tolist())
            #     exit()
            # else:
            this_df = this_df.astype({'doc_id': 'str', 'seg_num': 'str'})
            this_df["seg_id"] = this_df[["doc_id", "seg_num"]].apply(lambda x: ":".join(x), axis=1)
            # this_df = this_df.drop(muted, axis=1)
            meta = ["doc_id", 'seg_num', 'corpus', 'direction', 'wc_lemma', 'raw_tok', 'raw', 'sents']

        # else:  # assume we are using a doc_level dataset with a different frequency normalisation approach
        #     # I am not sure I want to use doc-level tables built on other averaging approaches
        #     this_df = this_df

        if args.sample == 'contrastive':
            # if doc-leve, I get ca. 200 rows, if seg-level ca. 8.5K in total in each lang
            sampled_df = reduce_to_extreme_docs(df0=this_df, tops_by_lang='classify/extremes/', lang=lang)
        elif args.sample == 'random':
            if args.level == 'doc':
                # no need to randomly sample the doc-level data: I have only 3000 rows
                sampled_df = this_df
            else:
                # Sample 2500 rows from each category in 'ttype' to avoid UserWarning: p-value may not be accurate for N > 5000
                sampled_df = this_df.groupby('ttype', group_keys=False).apply(sample_group, n=2500)
        else:
            sampled_df = this_df
        print(sampled_df.shape)

        # Discourse does not exist in DE, actiually if I can have assymentric features for two lang, I need to include acl:relcl for EN?
        # just keep zeros for discource in DE
        # columns_with_same_values = sampled_df.columns[sampled_df.apply(lambda x: x.nunique()) == 1]
        # print(columns_with_same_values)
        # # Create a new DataFrame without columns with the same values
        # filtered_df = sampled_df.drop(columns=columns_with_same_values)
        # print(filtered_df.shape)
        # exit()

        best_feats = feat_dict[lang][args.level]
        # print(best_feats)
        figname = f'{args.pics}{lang}_{args.level}_feats_diffs_contrastive_targets.png'

        lang_df = univariate(feature_lst=feats, best=best_feats, lang_df0=sampled_df, my_lang=lang, picpath=figname)
        lang_df = lang_df.set_index('feature')

        if 'relcl' in lang_df.index.tolist():
            print('GOTCHA')
            exit()

        my_weights = [i for i in os.listdir(args.svm_weights_dir) if f'{lang}_svm-weights_{args.level}_' in i]
        new_cols = []
        for i in my_weights:
            lang_svm_df = None
            if i.startswith('allfeats'):
                lang_svm_df = pd.read_csv(f'{args.svm_weights_dir}{i}', sep='\t')
                # print(lang_svm_df.tail())
                # input()

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

            if 'relcl' in lang_svm_df.index.tolist():
                print(i)
                print('GOTCHA')
                exit()
            new_cols.append(lang_svm_df)

        lang_df_updated = pd.concat([lang_df] + new_cols, axis=1)  # adding columns abs_weights and weights
        lang_df_updated = lang_df_updated.reset_index()

        lang_df_updated.insert(0, 'lang', lang)
        lang_df_updated = lang_df_updated.sort_values(by=['lang', 'MannWhit', 'compare', 'all_weights'],
                                                      ascending=[True, False, False, False])

        # print('*********')
        # print(lang_df_updated)
        # print('*********')

        if args.best_selection:
            best_feats = feat_dict[lang][args.level]

            # Select rows where values in feature column are in the list
            lang_df_updated = lang_df_updated[lang_df_updated['feature'].isin(best_feats)]
            lang_df_updated = lang_df_updated.reset_index(drop=True)
            # print('*********')
            # print(lang_df_updated)
            # print('*********')

        my_lang_dfs.append(lang_df_updated)

    for i in my_lang_dfs:
        print('*********')

        print(i)

    result = pd.concat(my_lang_dfs, axis=0)

    if args.best_selection:
        result = result.drop(['org_shapiro', 'tgt_shapiro', 'variances', 'F1_std_cv5'], axis=1)
        result.to_csv(f'{args.res}{args.level}_lean_feat-stats_de{len(feat_dict["de"][args.level])}_en{len(feat_dict["en"][args.level])}_sample_{args.sample}.tsv', sep='\t',
                      index=False)
    else:
        result.to_csv(f'{args.res}{args.level}_feat-stats_all_sample_{args.sample}.tsv', sep='\t', index=False)

    print(result)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
