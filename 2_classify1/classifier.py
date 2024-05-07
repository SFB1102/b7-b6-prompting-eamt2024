"""
17 Sept 2023

python3 2_classify1/classifier.py --level doc --nbest -1 --nbest_by RFECV

# NB! segment-level classification uses the output of --level doc --nbest -1 --nbest_by RFECV
python3 2_classify1/classifier.py --level seg --nbest -1 --nbest_by RFECV

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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from helpers import crossvalidated_svm, Logger, get_experimental_slice, get_xy_best, pca_this_lang, \
    plot_weighted_scores


# Function to apply stratified sampling within each group
def stratified_sample(group, cap, seed):
    return group.sample(n=cap, random_state=seed)


def reduce_data_by_lang_and_ttype(df0=None, cap=None, seed=None):
    # Apply stratified sampling within each group defined by columns
    smaller_df = df0.groupby(['lang', 'ttype'], group_keys=False).apply(stratified_sample, cap, seed)

    return smaller_df


def reduce_to_extreme_docs(df0=None, tops_by_lang=None):
    two_langs = []
    for my_lang in ['de', 'en']:
        fh = [f for f in os.listdir(tops_by_lang) if f.startswith(f'{my_lang}_doc_')]
        my_extremes_doc_ids = [i.strip() for i in open(f'{tops_by_lang}{fh[0]}', 'r').readlines()]

        smaller_lang = df0[df0['doc_id'].isin(my_extremes_doc_ids)]
        smaller_lang = smaller_lang[smaller_lang['raw'].apply(lambda x: len(str(x).split()) > 8)]

        two_langs.append(smaller_lang)
    smaller_df = pd.concat(two_langs, axis=0)

    return smaller_df


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
        sc = StandardScaler()
        training_set[my_single] = sc.fit_transform(training_set[my_single].values.reshape(-1, 1))

    x0 = training_set[[my_single]].values
    new_df = training_set[[my_single]]
    featurelist = [my_single]

    return x0, y0, featurelist, fns, new_df


def list_to_newline_sep(lst):
    return '\n'.join(lst)


# type hinting def add_numbers(x: int = None, y: int = None) -> int:
def check_for_nans(df0: pd.DataFrame = None) -> None:
    # Check for NaN values in each column, drop 96 rows with NaNs in mdd (one-word segments)
    nan_counts = df0.isna().sum()
    nan_counts_filtered = nan_counts[nan_counts != 0]
    total_nan_count = df0.isna().sum().sum()
    if total_nan_count:
        print(f'Total NaNs in the df: {total_nan_count}')
        # Print the number of NaN values for each column
        print("Columns with NaN values:")
        print(nan_counts_filtered)

        # Select rows where either Column1 or Column2 has None
        # filtered_rows = df0[df0['mhd'].isna() | df0['mdd'].isna()]
        # print(filtered_rows[['doc_id', 'seg_num', 'mdd', 'raw_tok']])

        df0 = df0.dropna(subset=['raw'])  # I could not sort out NaNs for 96 one-word segments like Warum ? and Nein !
        print(df0.shape)
        print('Your data table contains NaNs. Think what to do with them. Exiting ...')
        exit()
    else:
        print(f'No NaNs detected across all {df0.shape[1]} columns of the dataframe')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', help="fns, labels, text, lemmas, feats and srp vals", default='data/feats_tabled/')
    parser.add_argument('--setups', default=['seg-450-1500'])
    parser.add_argument('--level', choices=['seg', 'doc', 'doc100'], default='doc100')
    parser.add_argument('--nbest', type=int, default=0, help="Features to select. -1 for the optimal number with RFECV")
    parser.add_argument('--nbest_by', default='RFE', choices=['RFE', 'RFECV'],
                        help="ablation-based selection, inc with experimental k")
    parser.add_argument('--cv', type=int, default=10, help="Number of folds")
    parser.add_argument('--scale', type=int, default=1, choices=[1, 0], help="Do you want to use StandardScaler?")
    parser.add_argument('--rand', type=int, default=42)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--res', default='2_classify1/res/')
    parser.add_argument('--extreme_docs', default='2_classify1/extremes/')
    parser.add_argument('--logs', default='2_classify1/logs/')
    parser.add_argument('--pics', default='2_classify1/pics/')
    args = parser.parse_args()

    start = time.time()

    os.makedirs(args.res, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(args.pics, exist_ok=True)
    os.makedirs(args.extreme_docs, exist_ok=True)

    log_file = f'{args.logs}{args.level}_{args.nbest}best_{sys.argv[0].split("/")[-1].split(".")[0]}.log'
    sys.stdout = Logger(logfile=log_file)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    mega_collector = []
    num_res_lines = []
    num_feats = None
    _Y = None
    for my_size in args.setups:
        if args.level == 'seg' and my_size.startswith('doc'):
            continue
        setup_collector = []
        my_setup = [f for f in os.listdir(args.tables) if f.startswith(my_size) and f.endswith('.tsv.gz')]
        if my_setup:
            my_setup = my_setup[0]
            num_res_lines.append(my_setup)

            df = pd.read_csv(args.tables + my_setup, sep='\t', compression='gzip')
            print(df.shape)
            cols = sorted(df.columns.tolist())
            print(cols)

            # lose empty segments and what they are aligned to! the list is produced in tabulate_input.py
            lose_empty = [i.strip() for i in open('data/feats_tabled/empty_segs_europ2018.txt', 'r').readlines()]
            print(lose_empty)
            bitext_lose_empty = []
            for i in lose_empty:
                if i.startswith('ORG_WR_DE_EN'):
                    ii = i.replace('ORG_WR_DE_EN', 'TR_DE_EN')
                elif i.startswith('ORG_WR_EN_DE'):
                    ii = i.replace('ORG_WR_EN_DE', 'TR_EN_DE')
                else:
                    ii = i.replace('TR_DE_EN', 'ORG_WR_DE_EN')
                bitext_lose_empty.append(i)
                bitext_lose_empty.append(ii)
            print(bitext_lose_empty)

            # create unique iids:
            df = df.astype({'doc_id': 'str', 'seg_num': 'str'})
            df["iid"] = df[["doc_id", "seg_num"]].apply(lambda x: "-".join(x), axis=1)

            # drop meta except doc_id, lang, ttype
            meta = ['seg_num', 'corpus', 'direction', 'wc_lemma', 'raw_tok', 'sents']
            df = df.drop(meta, axis=1)

            df = df[~df['iid'].isin(bitext_lose_empty)]
            print(df.shape)
            check_for_nans(df0=df)

            print('\n====\n')

            more_meta = ['iid', 'doc_id', 'lang', 'ttype', 'raw']
            mute_them = ['case', 'relcl']
            df = df.drop(mute_them, axis=1)
            feats = [i for i in df.columns.tolist() if i not in more_meta and i not in mute_them]
            print(feats)
            print(len(feats))

            # the script needs to be run with level='doc' at least once to get classify/extremes/de_doc_200_extreme_items_45feats.txt
            if args.level == 'doc':
                if my_setup.startswith('seg'):
                    df = df.drop(['iid'], axis=1)
                    aggregate_functions = {col: 'mean' for col in feats}
                    df = df.groupby('doc_id', as_index=False).aggregate({'raw': list_to_newline_sep,
                                                                         'lang': 'first',
                                                                         'ttype': 'first',
                                                                         **aggregate_functions
                                                                         })
                    df = df.rename(columns={'doc_id': 'iid'})
                    df = df.drop(['raw'], axis=1)
            else:  # assume seg and reduce data to classify to 10K instances in total
                # seg_cap = 5000  # in each of the category
                # df = reduce_data_by_lang_and_ttype(df0=df, cap=seg_cap, seed=args.rand)
                df = reduce_to_extreme_docs(df0=df, tops_by_lang='classify/extremes/')
                df = df.drop(['doc_id', 'raw'], axis=1)
            print(df.head())

            print(df.shape)  # if args.level=seg expect 20K rows

            lang_collector = defaultdict(list)

            for lang in ['de', 'en']:
                # limit to one language
                _X, _Y, exp_df = get_experimental_slice(_df=df, _lang=lang, my_feats=feats)
                print(f'My X: {_X.shape}')
                print(f'My exp_df: {exp_df.shape}')
                top_feats = None
                if args.nbest:
                    print('Running feature selection ...')
                    _X, _Y, top_feats, _fns, exp_df = get_xy_best(exp_df, category='ttype', features=args.nbest,
                                                                  muted=mute_them,
                                                                  select_mode=args.nbest_by,
                                                                  scaling=args.scale, algo='SVM', cv=10)

                    print('\n********')
                    sorted_best = sorted(top_feats)
                    print(f'Best features for {lang.upper()}: {sorted_best}')
                    diff = list(set(feats).difference(set(top_feats)))
                    print(f'Discarded ({len(diff)}): {diff}')
                    print('********\n')

                else:
                    print('No feature selection. Running on the entire features set ...')
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

                print(f'\nData and labels are ready for cross-validation {args.level}:')
                print(f'\tMy classes: {set(_Y)}')
                num_feats = _X.shape[1]
                my_classes = Counter(_Y)
                for k, v in my_classes.items():
                    print(f'\t\t{k} {v}')
                print(f'\tLength of labels/instances array: {len(_Y)}')
                print(f'\tFeatures: {num_feats}\n')

                if args.nbest != 1:
                    # pca after scaling and feature selection (if any)
                    save_pic = f'{args.pics}pca_{lang}_{num_feats}feats_{len(_Y)}{args.level}.png'
                    pca_this_lang(x=_X, y=_Y, fns=_fns, save_name=save_pic, lang=lang, verbose=args.verbosity)

                # shuffle, i.e randomise the order of labels at splitting time
                splitter = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.rand)

                print(f'\nResults for {args.level}-level: {my_size} - {lang.upper()}\n')
                algos = ['SVM']
                for my_algo in algos:
                    all_meta = ['iid', 'ttype', 'lang', 'seg_num', 'corpus', 'direction', 'wc_lemma', 'raw_tok']
                    my_feat_cols = sorted([i for i in exp_df.columns if i not in all_meta])

                    accs, precis, recalls, f1, y_pred, classes, probs = crossvalidated_svm(_X, _Y,
                                                                                           algo=my_algo,
                                                                                           splitter=splitter,
                                                                                           class_weight='balanced',
                                                                                           run='default_linear',
                                                                                           rand=args.rand)
                    COLOURS = {'source': 'blue', 'target': 'red'}

                    os.makedirs(f'{args.res}weights/', exist_ok=True)

                    plot_weighted_scores(_X, _Y, feature_names=my_feat_cols, colors=COLOURS,
                                         saveresto=f'{args.res}weights/', savepicsto=args.pics, seed=args.rand,
                                         run=f'{lang}_svm-weights_{args.level}_{my_size.split("-", 1)[1]}',
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

                    res_dict = {'Lang': lang, 'Size': my_size,
                                'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1_}
                    res_df = pd.DataFrame(res_dict, index=[0])

                    if my_algo != 'DUMMY':
                        print(f'\nAccuracies by folds: {accs}')
                        print(f'F1-scores by folds: {f1}')

                        print('\nConfusion matrix:')
                        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                           columns=[f'pred_{classes[0]}', f'pred_{classes[1]}'],
                                           index=[f'{classes[0]}', f'{classes[1]}']))
                    # if args.nbest == 11:
                    #     res_df.insert(0, 'N_feats', 11)
                    # else:
                    res_df.insert(0, 'N_feats', num_feats)

                    if len(my_feat_cols) <= 15:
                        print(sorted_best)
                        print(my_feat_cols)
                        res_df.insert(0, 'feats', ' '.join(top_feats))
                    else:
                        res_df.insert(0, 'feats', '')
                    res_df.insert(0, 'support', len(_Y))
                    res_df.insert(0, 'algo', my_algo)
                    setup_collector.append(res_df)

                    # Create a DataFrame from the predicted probabilities for the optimal feature selection
                    if 15 < num_feats < 58:
                        probs_df = pd.DataFrame(data=probs, index=_fns, columns=classes)
                        probs_df = probs_df.reset_index()
                        probs_df = probs_df.rename(columns={'index': 'iid'})
                        os.makedirs(f'{args.res}probs/', exist_ok=True)
                        probs_df.to_csv(f'{args.res}probs/{lang}_{args.level}_samples_probs_{num_feats}feats.tsv',
                                        sep='\t', index=False)

                        if args.level == 'doc':
                            # Sort the DataFrame by column in descending order
                            src_df_sorted_desc = probs_df.sort_values(by='source', ascending=False).head(105)
                        else:
                            # get a longer list of confidently predicted segs: get all segs predicted with 0.99 prob
                            src_df_sorted_desc = probs_df[probs_df['source'] >= 0.95]
                            print(src_df_sorted_desc.shape)

                        # testing for confident classification errors
                        print('===== CONFIDENT ERRORS? tgt as org =====\n')
                        translated_as_original = []
                        for i in src_df_sorted_desc['iid'].tolist():
                            if i.startswith('TR'):
                                translated_as_original.append(i)
                        print(f'Translated items confidently predicted as original: {len(translated_as_original)}')
                        print(translated_as_original)
                        print('==========\n')

                        if args.level == 'doc':
                            tgt_df_sorted_desc = probs_df.sort_values(by='target', ascending=False).head(105)
                        else:
                            tgt_df_sorted_desc = probs_df[probs_df['target'] >= 0.95]
                            print(tgt_df_sorted_desc.shape)

                        # testing for confident classification errors
                        print('===== CONFIDENT ERRORS? org as tgt =====\n')
                        originals_as_translated = []
                        for i in tgt_df_sorted_desc['iid'].tolist():
                            if i.startswith('ORG'):
                                originals_as_translated.append(i)
                        print(f'Original items confidently predicted as translated: {len(originals_as_translated)}')
                        print(originals_as_translated)
                        print('=!=!=!=======\n')

                        # drop prediction errors
                        # print(src_df_sorted_desc.shape)
                        src_df_sorted_desc = src_df_sorted_desc[~src_df_sorted_desc['iid'].isin(translated_as_original)]
                        # print(src_df_sorted_desc.shape)
                        tgt_df_sorted_desc = tgt_df_sorted_desc[
                            ~tgt_df_sorted_desc['iid'].isin(originals_as_translated)]
                        if args.level == 'doc':
                            two_extremes = pd.concat([src_df_sorted_desc[:100], tgt_df_sorted_desc[:100]], axis=0)
                        else:
                            two_extremes = pd.concat([src_df_sorted_desc, tgt_df_sorted_desc], axis=0)
                        fns = two_extremes['iid'].tolist()

                        outpath = f'{args.extreme_docs}{lang}_{args.level}_{len(fns)}_extreme_items_{num_feats}feats.txt'

                        with open(outpath, 'w') as out_fns:
                            for id, i in enumerate(fns):
                                out_fns.write(i + '\n')
        else:
            print(f'You dont have results for {my_size}')
            continue
        setup_res = pd.concat(setup_collector, axis=0)
        mega_collector.append(setup_res)

    all_res = pd.concat(mega_collector, axis=0)

    print(f'\n{args.level.capitalize()}-level results for setups: {set(num_res_lines)}')
    print(all_res)

    outname = f'{args.res}{args.level}_results_{args.nbest}feats.tsv'
    all_res.to_csv(outname, sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
