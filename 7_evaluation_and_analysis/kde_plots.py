"""
14 Dec 2023

python3 7_evaluation_and_analysis/kde_plots.py --n_feats 58 --thres_type ratio2.5 --lose_bypassed
"""

import os
import sys
import pandas as pd
import argparse
import time
from datetime import datetime
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def pca_d1_density(x, y, dim=None, save_name=None, settings=None, sns=None):
    sns.set_style("whitegrid")
    sns.set_context('paper')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))  # 4:3 ratio

    d = {k: [] for k in set(y)}
    for i in range(len(y)):
        if y[i] in list(set(y)):
            d[y[i]].append(x[i])
    lined_items = []
    colored_items = []
    my_len = len(settings["aliases"])
    for idx, i in enumerate(settings['aliases']):
        # line plots
        sns.kdeplot(d[i], color=settings['cols'][idx], linewidth=2, linestyle=settings['lines'][idx], ax=ax)
        # lines and colors legends
        if my_len == 3:
            # colors legend only, I don't need the lines legend
            colclass = mpatches.Patch(color=settings['cols'][idx], label=settings['names'][idx])
            colored_items.append(colclass)
        else:
            if settings['aliases'][idx] == 'pro_tgt' or settings['aliases'][idx] == 'good_tgt':
                colclass = mpatches.Patch(color=settings['cols'][idx], label=settings['names'][idx])
                colored_items.append(colclass)
                ttext = mlines.Line2D([], [], color='black', linestyle=settings['lines'][idx], markersize=10,
                                      label='translation')
                lined_items.append(ttext)
            elif settings['aliases'][idx] == 'stu_tgt' or settings['aliases'][idx] == 'bad_tgt':
                colclass = mpatches.Patch(color=settings['cols'][idx], label=settings['names'][idx])
                colored_items.append(colclass)

            elif settings['aliases'][idx] == 'pro_src' or settings['aliases'][idx] == 'good_src':
                ttext = mlines.Line2D([], [], color='black', linestyle=settings['lines'][idx], markersize=10,
                                      label='source')
                lined_items.append(ttext)
            elif settings['aliases'][idx] == 'stu_src' or settings['aliases'][idx] == 'bad_src':
                continue
            else:  # settings['aliases'][idx] == 'ref_ref'
                colclass = mpatches.Patch(color=settings['cols'][idx], label=settings['names'][idx])
                colored_items.append(colclass)
                ttext = mlines.Line2D([], [], color='black', linestyle=settings['lines'][idx], markersize=10,
                                      label=settings['names'][idx])
                lined_items.append(ttext)

    if not lined_items:
        print(f'Sanity checks:\n\t Colors = 3: {len(colored_items) == 3}\n\t Lines == 3 or None: {type(lined_items)}')
        legend1 = plt.legend(handles=colored_items, fontsize=15, bbox_to_anchor=(1, 1), loc='best')
        plt.gca().add_artist(legend1)
    else:
        print(
            f'Sanity checks:\n\t Colors = 3: {len(colored_items) == 3}\n\t Lines == 3 or None: {len(lined_items) == 3}')
        # Create two legends and add them manually with from matplotlib.legend import Legend
        legend2 = plt.legend(handles=lined_items, fontsize=15, loc=2)  # , bbox_to_anchor=(0.2, 1)
        plt.gca().add_artist(legend2)

        legend1 = plt.legend(handles=colored_items, fontsize=15, loc=1)  # , bbox_to_anchor=(1, 1), loc='best'
        plt.gca().add_artist(legend1)
    ax.set_xlabel(f'PCA D{dim} values', fontsize=15)
    ax.set_ylabel('Density', fontsize=15)

    plt.xlim([-5, 6])

    plt.grid(color='darkgrey', linestyle='--', linewidth=0.5, alpha=0.5)

    # plt.title(f'based on PCA transform of {vect.upper()} representation', ha='center', fontsize=14)

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()

    plt.show()


def preprocess_rewritten_setup(_df0=None, lang=None):
    _df0 = _df0[_df0['lang'] == lang]

    # this is df with features for rewritten segments
    _df0 = _df0.dropna()
    # # Check for NaN values in each column
    nan_counts = _df0.isna().sum()
    nan_counts_filtered = nan_counts[nan_counts != 0]
    total_nan_count = _df0.isna().sum().sum()
    print(f'Counts of NaN in rewritten (empty segs are filtered out at parsing): {_df0["rewritten"].isna().sum()}')
    if total_nan_count:
        # Print the number of NaN values for each column
        print("\nColumns with NaN values:")
        print(nan_counts_filtered)
        print(f'Total NaNs in the df: {total_nan_count}\n')
        exit()

    _df0['doc_id'] = _df0['seg_id'].apply(lambda x: x.split('-')[0])  # add doc_id column
    _df0.insert(2, 'ttype', 'rewritten')  # add ttype column = new label
    _df0 = _df0.drop(['rewritten', 'wc_lemma'], axis=1)  # add

    return _df0


def balance_orig_on_docs(_df0=None, tops_by_lang=None, lang=None):
    _df0_lang = _df0[(_df0['lang'] == lang)]

    fh = [f for f in os.listdir(tops_by_lang) if f.startswith(f'{lang}_doc_200')]
    my_extremes_doc_ids = [i.strip() for i in open(f'{tops_by_lang}{fh[0]}', 'r').readlines()]

    smaller_lang = _df0_lang[_df0_lang['doc_id'].isin(my_extremes_doc_ids)]

    return smaller_lang


def preprocess_originals(_df0=None, lang=None, extremes_dir=None):
    mini_df0 = balance_orig_on_docs(_df0=df0, tops_by_lang=extremes_dir, lang=lang)
    # create unique seg_ids in the initial dataseet:
    mini_df0 = mini_df0.astype({'doc_id': 'str', 'seg_num': 'str'})
    mini_df0["seg_id"] = mini_df0[["doc_id", "seg_num"]].apply(lambda x: "-".join(x), axis=1)
    mini_df0 = mini_df0[mini_df0['raw'].apply(lambda x: len(str(x).split()) > 8)]
    meta = ['seg_num', 'corpus', 'direction', 'wc_lemma', 'raw', 'raw_tok', 'sents']
    originals = mini_df0.drop(meta, axis=1)

    return originals


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


def make_dirs(logsto, picsto, sub):
    os.makedirs(f'{logsto}', exist_ok=True)
    os.makedirs(f'{picsto}', exist_ok=True)
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{sub}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewritten_feats', default='data/rewritten/feats_tabled2/')
    parser.add_argument('--extremes', default='2_classify1/extremes/', help='path to lists of contrastive ids')
    parser.add_argument('--thres_type', choices=['ratio2.5', 'std2'], default='ratio2.5', required=True)
    parser.add_argument('--initial_feats', help='feats in HT', default='data/feats_tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--lose_bypassed', action="store_true", help='Exclude short segs')
    parser.add_argument('--n_feats', type=int, default=58, choices=[15, 58])
    parser.add_argument('--logsto', default='logs/rewritten')
    parser.add_argument('--picsto', default='7_evaluation_and_analysis/pics/')
    args = parser.parse_args()

    start = time.time()

    make_dirs(args.logsto, args.picsto, sub=args.thres_type)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    # as of 7 Dec 2023  15 feats
    feat_dict = {'de': {
        'doc': ['addit', 'advcl', 'advmod', 'caus', 'ccomp', 'fin', 'mdd', 'mean_sent_wc', 'mhd', 'nmod', 'parataxis',
                'pastv', 'poss', 'simple', 'ttr'],
        'seg': ['addit', 'advcl', 'advmod', 'caus', 'fin', 'iobj', 'mdd', 'mean_sent_wc', 'mhd', 'nmod', 'nnargs',
                'parataxis', 'pastv', 'poss', 'ttr']},
        'en': {'doc': ['addit', 'advcl', 'advers', 'compound', 'conj', 'demdets', 'fin', 'mean_sent_wc', 'mhd',
                       'nmod', 'nnargs', 'numcls', 'obl', 'ppron', 'sconj'],
               'seg': ['addit', 'advcl', 'advers', 'advmod', 'compound', 'conj', 'mean_sent_wc', 'mhd', 'negs',
                       'nmod', 'nnargs', 'numcls', 'obl', 'pastv', 'ppron']}
    }

    settings = defaultdict()
    keys = ['aliases', 'names', 'cols', 'lines', 'points', 'edge']
    vals = [['source', 'target', 'rewritten'],
            ['original', 'translated', 'rewritten'],
            ['blue', 'forestgreen', 'red'],
            ['--', 'solid', 'solid'],
            ['.', 'x', 'd'],
            ['darkblue', 'darkgreen', 'darkred']]
    for k, vs in zip(keys, vals):
        settings[k] = vs

    for tlang in ['de', 'en']:
        print(f'{tlang.upper()}')
        df0 = pd.read_csv(args.initial_feats, sep='\t', compression='gzip')
        initial_two = preprocess_originals(_df0=df0, lang=tlang, extremes_dir=args.extremes)

        print(f"Columns in 1.OGR vs HT (NO shorts!!) {initial_two.shape}: {initial_two.columns.tolist()}")

        for setup in ['self-guided_min', 'self-guided_detailed', 'translated_min',
                      'feature-based_min', 'feature-based_detailed']:

            if setup.startswith('feature-based_'):
                with_thres = f"{setup}_{args.thres_type}"
            else:
                with_thres = setup
            try:
                df1 = pd.read_csv(
                    f'{args.rewritten_feats}{args.thres_type}/gpt4_temp0.7_{with_thres}_rewritten_feats.tsv.gz',
                    sep='\t', compression='gzip')
            except FileNotFoundError:
                continue
            rewritten = preprocess_rewritten_setup(_df0=df1, lang=tlang)
            print(f"Columns in 2. classifier data {rewritten.shape}: {rewritten.columns.tolist()}")

            kde_data = pd.concat([initial_two, rewritten], axis=0)
            print(kde_data.shape)

            if args.lose_bypassed:
                lose_them = [i.strip() for i in
                             open(f'data/rewritten/curated/lose_segids/{args.thres_type}/{tlang}/{setup}.ids',
                                  'r').readlines()]
                kde_data = kde_data[~kde_data['seg_id'].isin(lose_them)]
                print(f'*** After losing {len(lose_them)} short: {kde_data.shape}\n')

            print(kde_data.columns.tolist())
            my_dim = 2

            Y = kde_data['ttype'].values
            if args.n_feats > 15:
                print('Using all features')
                meta = ['doc_id', 'ttype', 'lang', 'seg_id', 'relcl', 'case']
                all_feats = [i for i in kde_data.columns.tolist() if i not in meta]
                print(all_feats)
                this_df = kde_data[all_feats]
            else:
                print('Using top features')
                these_feats = feat_dict[tlang]['seg']
                this_df = kde_data[these_feats]
            print(this_df.shape)
            X = this_df.values

            sc = StandardScaler()
            X = sc.fit_transform(X)
            print('\nFeature values are scaled column-wise with StandardScaler()')

            pca = PCA(n_components=my_dim)
            reduced_x = pca.fit_transform(X)
            print(reduced_x.shape)

            reduced_x = reduced_x[:, my_dim - 1]

            # from pca_d1_density
            sns.set_style("whitegrid")
            sns.set_context('paper')
            # Set the default font properties
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.size'] = 17
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))  # 4:3 ratio

            d = {k: [] for k in set(Y)}

            for i in range(len(Y)):
                if Y[i] in list(set(Y)):
                    d[Y[i]].append(reduced_x[i])

            lined_items = []
            colored_items = []
            my_len = len(settings["aliases"])
            for idx, i in enumerate(settings['aliases']):
                # line plots
                sns.kdeplot(d[i], color=settings['cols'][idx], linewidth=2.5, linestyle=settings['lines'][idx], ax=ax)
                colclass = mpatches.Patch(color=settings['cols'][idx], label=settings['names'][idx])
                colored_items.append(colclass)
            print(
                f'Sanity checks:\n\t Colors = 3: {len(colored_items) == 3}\n\t Lines == 3 or None: {type(lined_items)}')
            legend1 = plt.legend(handles=colored_items, fontsize=15, bbox_to_anchor=(1, 1), loc='best')
            plt.gca().add_artist(legend1)

            ax.set_xlabel(f'PCA D{my_dim} values', fontsize=15)
            ax.set_ylabel('Density', fontsize=16)

            plt.grid(color='darkgrey', linestyle='--', linewidth=0.5, alpha=0.5)
            if tlang == 'de':
                plt.title(f'German: {setup.replace("-based", "-guided")}', ha='center', fontsize=17)
            else:
                plt.title(f'English: {setup}', ha='center', fontsize=17)

            plt.xticks(fontsize=15)  # Adjust the font size of x-axis tick labels
            plt.yticks(fontsize=15)

            save_dir = f'{args.picsto}{args.thres_type}/'
            os.makedirs(save_dir, exist_ok=True)

            save_name = f'{save_dir}kdeD{my_dim}_{tlang}_{args.n_feats}feats_{setup}_{args.thres_type}.png'
            plt.savefig(save_name)

            plt.show()
