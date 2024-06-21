"""
UPD 20June 2024
this expects a table with features for the rewritten translations and
the original data/feats_tabled/seg-450-1500.feats.tsv.gz + lists of contrastive non-translated docs from 2_classify1/extremes/

17 Sept 2023
--lose_bypassed is needed to reproduce reported results: It excludes short (<8 tokens) and copied-over segs from subsequent analysis
data/rewritten/curated/no_shorts_and_copies/ folder has filtered aligned outputs by lang, thres_type and mode (20 tsv)

# I have produced full data tables for self-guided and mt modes because of diffs in maual curation and for consistency
python3 6_classify2/classifier2.py --thres_type std2 --level seg --nbest 0 --nbest_by RFECV --verbosity 0 --lose_bypassed
python3 6_classify2/classifier2.py --thres_type ratio2.5 --level seg --nbest 11 --nbest_by RFECV --verbosity 0 --lose_bypassed
"""

import numpy as np
import os
import sys
import pandas as pd
import argparse
import time
import random
from datetime import datetime
from collections import Counter
from collections import defaultdict

from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import shuffle

from sklearn.decomposition import PCA
import matplotlib.lines as mlines

from helpers import crossvalidated_svm, Logger, get_experimental_slice, get_xy_best, plot_weighted_scores


# Function to apply stratified sampling within each group
def stratified_sample(group, cap, seed):
    return group.sample(n=cap, random_state=seed)


def reduce_data_by_lang_and_ttype(_df0=None, cap=None, seed=None):
    # Apply stratified sampling within each group defined by columns
    smaller_df = _df0.groupby(['lang', 'ttype'], group_keys=False).apply(stratified_sample, cap, seed)

    return smaller_df


def reduce_to_extreme_docs(_df0=None, tops_by_lang=None):
    two_langs = []
    for my_lang in ['de', 'en']:
        fh = [f for f in os.listdir(tops_by_lang) if f.startswith(f'{my_lang}_doc_')]
        my_extremes_doc_ids = [i.strip() for i in open(f'{tops_by_lang}{fh[0]}', 'r').readlines()]

        smaller_lang = _df0[_df0['doc_id'].isin(my_extremes_doc_ids)]
        two_langs.append(smaller_lang)
    smaller_df = pd.concat(two_langs, axis=0)

    return smaller_df


def balance_orig_on_docs(_df0=None, how=None, tops_by_lang=None, size=None, lang=None):
    _df0_lang = _df0[(_df0['lang'] == lang) & (_df0['ttype'] == 'source')]
    if how == 'random':
        # get a list of 100 random original filenames, each wih at least 10 segments
        temp_df0_lang = _df0_lang[['doc_id', 'seg_num']]
        temp_df0_lang = temp_df0_lang.groupby('doc_id', as_index=False).aggregate(
            {'seg_num': lambda x: len(x.tolist())})
        temp_df0_lang = temp_df0_lang[temp_df0_lang['seg_num'] > 15]
        origs = list(set(temp_df0_lang['doc_id'].tolist()))
        my100orig = random.sample(origs, size)

        smaller_lang = _df0_lang[_df0_lang['doc_id'].isin(my100orig)]
        print(smaller_lang.shape)
    else:
        fh = [f for f in os.listdir(tops_by_lang) if f.startswith(f'{lang}_doc_')]
        my_extremes_doc_ids = [i.strip() for i in open(f'{tops_by_lang}{fh[0]}', 'r').readlines()]
        # this includes ORG_WR_EN_DE_004910 and TR_DE_EN_004321 !!!
        smaller_lang = _df0_lang[_df0_lang['doc_id'].isin(my_extremes_doc_ids)]

    return smaller_lang


def pca_this_lang(x=None, y=None, fns=None, save_name=None, lang=None, verbose=None):
    # transforming vectors
    print(x.shape)
    pca = PCA(n_components=2)
    x = pca.fit_transform(x)
    print(x.shape)

    settings = defaultdict()
    keys = ['aliases', 'names', 'cols', 'lines', 'points', 'edge']
    vals = [['source', 'rewritten'],
            ['original', 'rewritten'],
            ['blue', 'red'],
            ['solid', 'solid'],
            ['.', 'x'],
            ['darkblue', 'darkred']]
    for k, vs in zip(keys, vals):
        settings[k] = vs

    fig, ax = plt.subplots(figsize=(12, 9))  # 4:3 ratio
    POINTSIZE = 40  # size of point labels
    FONTSIZE = 14
    COLOR = 'black'  # this is the color of the text, not of the datapoint
    XYTEXT = (6, 6)
    colored_items = []
    for idx, cat in enumerate(settings['aliases']):
        for i, _ in enumerate(x):
            if y[i] == cat:
                ax.scatter(*x[i], s=POINTSIZE, marker=settings['points'][idx],
                           # edgecolor=settings['edge'][idx],
                           color=settings['cols'][idx], alpha=0.7)
                if lang == 'en':
                    if ('source' in cat or 'rewritten' in cat) and x[i][1] > 15:
                        # print(fns[i])
                        plt.annotate(fns[i],
                                     fontsize=FONTSIZE,
                                     color=COLOR,  # this is the color of the text, not of the datapoint
                                     xy=(x[i][0], x[i][1]),
                                     xytext=XYTEXT,
                                     # if you don't use this plt throws up a warning that it uses `textcoords` kwarg
                                     textcoords='offset points',
                                     ha='right',
                                     # this counterintuitively places the labels wrt the points (try: left, top)
                                     va='bottom')
                if lang == 'de':
                    if ('source' in cat or 'rewritten' in cat) and x[i][1] > 10:
                        plt.annotate(fns[i],
                                     fontsize=FONTSIZE,
                                     color=COLOR,  # this is the color of the text, not of the datapoint
                                     xy=(x[i][0], x[i][1]),
                                     xytext=XYTEXT,
                                     # if you don't use this plt throws up a warning that it uses `textcoords` kwarg
                                     textcoords='offset points',
                                     ha='right',
                                     # this counterintuitively places the labels wrt the points (try: left, top)
                                     va='bottom')
        colmarker = mlines.Line2D([], [], color=settings['cols'][idx], marker=settings['points'][idx], linestyle='None',
                                  markersize=5, label=settings['names'][idx])
        colored_items.append(colmarker)
    legend1 = plt.legend(handles=colored_items, fontsize=14, loc=1)  # bbox_to_anchor=(1, 1), loc='best'
    plt.gca().add_artist(legend1)

    # hide tick and tick label of the big axis
    plt.grid(color='darkgrey', linestyle='--', linewidth=0.3, alpha=0.5)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    if lang == 'en':
        plt.title('English', fontdict={'fontsize': 14})
    else:
        plt.title('German', fontdict={'fontsize': 14})

    ax.set_xlabel('PCA D1 values', fontsize=14)
    ax.set_ylabel('PCA D2 values', fontsize=14)

    string = f'Variance explained on D1: {pca.explained_variance_ratio_[0]:.2f}, D2: {pca.explained_variance_ratio_[1]:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if lang == 'en':
        plt.text(2, -5, string, fontsize=14, verticalalignment='top', bbox=props)
    else:
        plt.text(2, -6.5, string, fontsize=14, verticalalignment='top', bbox=props)

    if save_name:
        plt.savefig(save_name)
    if verbose > 0:
        plt.show()


def get_preselect_vals(training_set, category='ttype', featnames=None, scaling=None):
    y0 = training_set.loc[:, category].values
    fns = training_set['iid'].tolist()
    # drop remaining meta
    training_set = training_set.drop(['iid', 'ttype', 'lang'], axis=1)
    print(f'Number of input features: {training_set.shape[1]}')

    if scaling:
        print(f'===StandardScaler() ===')
        # centering and scaling for each feature
        # transform your data such that its distribution will have a mean value 0 and standard deviation of 1
        # to meet the assumption that all features are centered around 0 and have variance in the same order
        # each value will have the sample mean subtracted, and then divided by the StD of the whole dataset
        sc = StandardScaler()
        training_set[featnames] = sc.fit_transform(training_set[featnames])

    x0 = training_set[featnames].values
    new_df = training_set[featnames]
    featurelist = featnames

    return x0, y0, featurelist, fns, new_df


def get_singleton_vals(training_set, category='ttype', my_single='mean_sent_wc', scaling=None):
    y0 = training_set.loc[:, category].values
    fns = training_set['iid'].tolist()
    # drop remaining meta
    training_set = training_set.drop(['iid', 'ttype', 'lang'], axis=1)
    print(f'Number of input features: {training_set.shape[1]}')

    if scaling:
        print(f'===StandardScaler() ===')
        # centering and scaling for each feature
        # transform your data such that its distribution will have a mean value 0 and standard deviation of 1
        # to meet the assumption that all features are centered around 0 and have variance in the same order
        # each value will have the sample mean subtracted, and then divided by the StD of the whole dataset
        sc = StandardScaler()
        training_set[my_single] = sc.fit_transform(training_set[my_single].values.reshape(-1, 1))

    x0 = training_set[[my_single]].values
    new_df = training_set[[my_single]]
    featurelist = [my_single]

    return x0, y0, featurelist, fns, new_df


def preprocess_originals(_df0=None, lang=None, level=None, extremes_dir=None):
    if level == 'doc':
        # I need to balance original and rewritten classes!
        mini_df0 = balance_orig_on_docs(_df0=df0, how='random', tops_by_lang=None, size=100, lang=lang)
    else:
        mini_df0 = balance_orig_on_docs(_df0=df0, how='extreme', tops_by_lang=extremes_dir, size=None,
                                        lang=lang)
    # create unique seg_ids in the initial dataset:
    mini_df0 = mini_df0.astype({'doc_id': 'str', 'seg_num': 'str'})
    mini_df0["seg_id"] = mini_df0[["doc_id", "seg_num"]].apply(lambda x: ":".join(x), axis=1)
    # are there segments shorter than 8?
    # exclude short sentences:
    mini_df0 = mini_df0[mini_df0['raw'].apply(lambda x: len(str(x).split()) > 8)]
    meta = ['seg_num', 'corpus', 'direction', 'wc_lemma', 'raw', 'raw_tok', 'sents']

    originals = mini_df0.drop(meta, axis=1)

    return originals


def preprocess_rewritten_setup(_df0=None, lang=None):
    _df0 = _df0[_df0['lang'] == lang]
    print(f"\nColumns in rewritten {_df0.shape}: {_df0.columns.tolist()}")

    # this is df with features for rewritten segments
    _df0 = _df0.dropna()
    # # Check for NaN values in each column
    nan_counts = _df0.isna().sum()
    nan_counts_filtered = nan_counts[nan_counts != 0]
    total_nan_count = _df0.isna().sum().sum()
    print(
        f'Counts of NaN in rewritten (empty segs are filtered out at parsing): {_df0["rewritten"].isna().sum()}')
    if total_nan_count:
        # Print the number of NaN values for each column
        print("\nColumns with NaN values:")
        print(nan_counts_filtered)
        print(f'Total NaNs in the df: {total_nan_count}\n')
        exit()
    # I don't care for doc level anymore, but there is an annoying inconsistency in this - and :
    _df0['doc_id'] = _df0['seg_id'].apply(lambda x: x.split('-')[0])  # add doc_id column

    noise1 = _df0[_df0['rewritten'].str.contains('a revised translation')]
    noise2 = _df0[_df0['rewritten'].str.contains('a revised version of the translation')]

    if not noise1.empty or not noise2.empty:
        num_err = noise1.shape[0] + noise2.shape[0]
    else:
        num_err = None

    _df0.insert(2, 'ttype', 'rewritten')  # add ttype column = new label
    _df0 = _df0.drop(['rewritten', 'wc_lemma'], axis=1)  # add

    return _df0, num_err


def make_dirs(outdir, logsto, picsto, sub):
    os.makedirs(f"{outdir}", exist_ok=True)
    os.makedirs(f'{logsto}', exist_ok=True)
    os.makedirs(f'{picsto}', exist_ok=True)
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{sub}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # unlike classifier.py classifier2.py can reuse features for non-translated class!
    parser.add_argument('--rewritten_feats', help='folder with feature tables', default='data/rewritten/feats_tabled2/')
    parser.add_argument('--thres_type', choices=['ratio2.5', 'std2'], default='ratio2.5', required=True)
    parser.add_argument('--initial_feats', help='feats in HT', default='data/feats_tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--level', choices=['doc', 'seg'], required=True)
    parser.add_argument('--extremes', default='2_classify1/extremes/', help='path to lists of contrastive ids')
    parser.add_argument('--lose_bypassed', action="store_true", help='Exclude short segs')
    parser.add_argument('--nbest', type=int, choices=[1, 11, -1, 15, 0], default=0,
                        help="Features to select: -1 for the optimal number with RFECV; 11 for preselect features")
    parser.add_argument('--nbest_by', default='RFE', choices=['RFE', 'RFECV'],
                        help="ablation-based selection, inc with experimental k")
    parser.add_argument('--cv', type=int, default=10, help="Number of folds")
    parser.add_argument('--scale', type=int, default=1, choices=[1, 0], help="Do you want to use StandardScaler?")
    parser.add_argument('--rand', type=int, default=42)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--outdir', default='6_classify2/res/')
    parser.add_argument('--logsto', default='logs/rewritten/')
    parser.add_argument('--picsto', default='6_classify2/pics/')

    args = parser.parse_args()

    start = time.time()

    make_dirs(args.outdir, args.logsto, args.picsto, sub=args.thres_type)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    # # as of 7 Dec 2023  15 feats
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

    df0 = pd.read_csv(args.initial_feats, sep='\t', compression='gzip')
    print(f"Columns in 1. classifier data {df0.shape}: {df0.columns.tolist()}")
    print(len(set(df0.doc_id.tolist())))
    print(len(set(df0.ttype.tolist())))
    lang_collector = []
    num_feats = None
    for lang in ['de', 'en']:
        # build new originals class: random or most_original_aligned sources?
        nontra = preprocess_originals(_df0=df0, lang=lang, level=args.level, extremes_dir=args.extremes)
        print(f'Non-translated for {lang.upper()} (level={args.level}): {nontra.shape}')

        for setup in ['self-guided_min', 'self-guided_detailed', 'translated_min',
                      'feature-based_min', 'feature-based_detailed']:
            # thres_type does not matter for self-guided modes, they are stored in ratio2.5
            # I have produced full data tables for self-guided and mt modes because of diffs in manual curation and for consistency
            print(setup.upper())
            try:
                my_rewritten = [f for f in os.listdir(f'{args.rewritten_feats}{args.thres_type}/') if f.endswith(f'.tsv.gz') and setup in f][0]
            except IndexError:
                print(setup.upper())
                continue

            df1 = pd.read_csv(f'{args.rewritten_feats}{args.thres_type}/{my_rewritten}', sep='\t', compression='gzip')
            print(df1.head())

            rewritten, noisy = preprocess_rewritten_setup(_df0=df1, lang=lang)
            if noisy:
                print(f'{lang.upper()} {setup} instances: {noisy / rewritten.shape[0] * 100:.1f} ({noisy})')

            # stack them
            task_data = pd.concat([nontra, rewritten], axis=0)

            print(task_data.tail())
            print(task_data.shape)

            more_meta = ['seg_id', 'doc_id', 'lang', 'ttype', 'relcl', 'case']  # added features muted for collinearity
            feats = [i for i in task_data.columns.tolist() if i not in more_meta]

            print(f'Merged data has {len(feats)} features for {task_data.shape[0]} binary-labelled instances')

            if args.level == 'doc':
                task_data = task_data.drop(['seg_id'], axis=1)
                aggregate_functions = {col: 'mean' for col in feats}
                task_data = task_data.groupby('doc_id', as_index=False).aggregate({'lang': 'first',
                                                                                   'ttype': 'first',
                                                                                   **aggregate_functions
                                                                                   })
                task_data = task_data.rename(columns={'doc_id': 'iid'})
            else:  # assume seg: the data is already reduced to 100 nontra and 100 rewritten (for the most_translated selection)
                task_data = task_data.drop(['doc_id'], axis=1)

                if args.lose_bypassed:
                    # lose only shorts, keep copies!
                    lose_them = [i.strip() for i in
                                 open(f'data/rewritten/curated/lose_segids/{args.thres_type}/{lang}/shorts_only_{setup}.ids', 'r').readlines()]
                    # lose_them = [i.strip() for i in
                    #              open(f'data/rewritten/curated/lose_segids/{args.thres_type}/{lang}/{setup}.ids', 'r').readlines()]
                    task_data = task_data[~task_data['seg_id'].isin(lose_them)]

                task_data = task_data.rename(columns={'seg_id': 'iid'})

            # explicitly drop muted features, I am not sure how they creep back in
            try:
                task_data = task_data.drop(['relcl', 'case'], axis=1)
            except KeyError:
                task_data = task_data

            # limit to one language: translations vs non-translation:
            # translations vs non-translation (already limited, but ok
            _X, _Y, exp_df = get_experimental_slice(_df=task_data, _lang=lang, my_feats=feats)

            top_feats = None
            if args.nbest:
                if args.nbest == 1:
                    _X, _Y, top_feats, _fns, exp_df = get_singleton_vals(exp_df, category='ttype',
                                                                         my_single='mean_sent_wc',
                                                                         scaling=args.scale)
                    print('\n********')
                    print(f'Running with one feature only: {top_feats}')
                    print('********\n')
                elif args.nbest == 11:
                    this_level_lang_selection = feat_dict[lang][args.level]
                    _X, _Y, top_feats, _fns, exp_df = get_preselect_vals(exp_df, category='ttype',
                                                                         featnames=this_level_lang_selection,
                                                                         scaling=args.scale)
                    print('\n********')
                    print(f'Running with {len(top_feats)} pre-select features: {top_feats}')
                    assert len(top_feats) == len(this_level_lang_selection), 'Huston, we have got a problem!'
                    print('********\n')
                else:
                    _X, _Y, top_feats, _fns, exp_df = get_xy_best(exp_df, category='ttype', features=args.nbest,
                                                                  select_mode=args.nbest_by,
                                                                  scaling=args.scale, algo='SVM', cv=10)

                    print('\n********')
                    print(f'Best features for {lang.upper()}: {top_feats}')
                    diff = list(set(feats).difference(set(top_feats)))
                    print(f'Discarded ({len(diff)}): {diff}')
                    print('********\n')

            else:
                print('*** Classifying on the full feature set ***')
                _fns = exp_df['iid'].values
                _Y = _Y

                # scaling for no-feature selection scenario
                if args.scale:
                    sc = StandardScaler()
                    _X = sc.fit_transform(_X)
                    print('\nFeature values are scaled column-wise with StandardScaler()')
                else:
                    _X = _X
                    print('\nNo scaling used')

            print('\nData and labels are ready for cross-validation: ')
            print(f'\tMy classes: {set(_Y)}')
            num_feats = _X.shape[1]
            my_classes = Counter(_Y)

            for k, v in my_classes.items():
                print(f'\t\t{k} {v}')
            if len(my_classes) < 2:
                exit()

            print(f'\tLength of labels/instances array: {len(_Y)}')
            print(f'\tFeatures: {num_feats}\n')

            if args.nbest != 1:
                # pca after scaling and feature selection (if any)
                save_pic = f'{args.picsto}pca_rewritten_{setup}_{args.thres_type}_{lang}_{num_feats}feats_{len(_Y)}{args.level}.png'
                pca_this_lang(x=_X, y=_Y, fns=_fns, save_name=save_pic, lang=lang, verbose=args.verbosity)

            # shuffle, i.e randomise the order of labels at splitting time
            splitter = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.rand)

            print(f'\nResults for {args.level}-level: {lang.upper()}\n')
            algos = ['SVM']  # 'DUMMY',
            for my_algo in algos:
                # this is needed to get the relevant subset of feats. None of this meta (except iid for no-selection setup) is in df
                all_meta = ['iid', 'ttype', 'lang', 'seg_num', 'corpus', 'direction', 'wc_lemma', 'raw_tok']
                my_feats = [i for i in exp_df.columns if i not in all_meta]

                accs, precis, recalls, f1, y_pred, classes, probs = crossvalidated_svm(_X, _Y,
                                                                                       algo=my_algo,
                                                                                       splitter=splitter,
                                                                                       class_weight='balanced',
                                                                                       run='default_linear',
                                                                                       rand=args.rand)
                COLOURS = {'source': 'blue', 'rewritten': 'red'}
                os.makedirs(f'{args.outdir}weights/', exist_ok=True)
                if args.nbest != 1:
                    plot_weighted_scores(_X, _Y, feature_names=my_feats, colors=COLOURS,
                                         saveresto=f'{args.outdir}weights/', savepicsto=args.picsto, seed=args.rand,
                                         run=f'{lang}_{args.level}_rewritten_{setup}_{args.thres_type}_svm-weights',
                                         verbose=args.verbosity)

                y_test = _Y

                print(f'{my_algo} on ({args.cv}-fold cv, {num_feats} features):')
                # correctly predicted (errors are ignored)
                accuracy = f'{(sum(accs) / len(accs) * 100):.2f} (+/-{np.std(np.array(accs)) * 100:.2f})'
                # precision: how eager/reluctant is the model to predict this class, even if wrongly?
                precision = f'{(sum(precis) / len(precis) * 100):.2f} (+/-{np.std(np.array(precis)) * 100:.2f})'
                # recall: TP rate (sensitivity): correct positive/all positive: does the model fetch everything?
                recall = f'{(sum(recalls) / len(recalls) * 100):.2f} (+/-{np.std(np.array(recalls)) * 100:.2f})'
                f1_ = f'{(sum(f1) / len(f1) * 100):.2f} (+/-{np.std(np.array(f1)) * 100:.2f})'

                res_dict = {'Lang': lang, 'corpus': 'rewritten',
                            'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1_}
                res_df = pd.DataFrame(res_dict, index=[0])

                if my_algo != 'DUMMY':
                    print(f'\nAccuracies by folds: {accs}')
                    print(f'F1-scores by folds: {f1}')

                    print('\nConfusion matrix:')
                    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                       columns=[f'pred_{classes[0]}', f'pred_{classes[1]}'],
                                       index=[f'{classes[0]}', f'{classes[1]}']))
                res_df.insert(0, 'N_feats', num_feats)
                if len(my_feats) <= 15:
                    res_df.insert(0, 'feats', ' '.join(top_feats))
                else:
                    res_df.insert(0, 'feats', '')

                res_df.insert(0, 'model', my_rewritten.split('_')[0])
                res_df['model'] = res_df['model'].replace({'temp0.7': 'gpt-4'})
                res_df.insert(0, 'setup', setup)
                res_df.insert(0, 'support', len(_Y))
                res_df.insert(0, 'algo', my_algo)
                lang_collector.append(res_df)

                # Create a DataFrame from the predicted probabilities for the optimal feature selection
                if 10 < num_feats < 60:
                    probs_df = pd.DataFrame(data=probs, index=_fns, columns=classes)
                    probs_df = probs_df.reset_index()
                    probs_df = probs_df.rename(columns={'index': 'iid'})
                    os.makedirs(f'{args.outdir}probs/', exist_ok=True)
                    probs_df.to_csv(
                        f'{args.outdir}probs/{lang}_{args.level}_{setup}_{args.thres_type}_{num_feats}feats.tsv',
                        sep='\t',
                        index=False)
    langs_res = pd.concat(lang_collector, axis=0)
    outname = f'{args.outdir}{args.level}_{args.thres_type}_rewritten_results_{num_feats}feats.tsv'
    langs_res.to_csv(outname, sep='\t', index=False)
    print(langs_res)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
