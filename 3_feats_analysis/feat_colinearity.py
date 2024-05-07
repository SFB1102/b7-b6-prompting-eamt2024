'''
22 Sept 2023

While waiting for re-parsed data
prepare to look at features colinearity

python3 analysis/feat_colinearity.py
'''
import argparse
from datetime import datetime
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', default='extract/tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--level', choices=['seg', 'doc'], default='doc')
    # parser.add_argument('--zoom', nargs='+', default=None, help='pass a list of features to describe univariately')
    # parser.add_argument('--zoom_name', default='best_feats', help='how to name the results file?')
    parser.add_argument('--res', default=f'analysis/res/')
    parser.add_argument('--logs', default=f'analysis/logs/')
    parser.add_argument('--pics', default='analysis/pics/')
    args = parser.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    log_file = f'{args.logs}{args.level}_colinearity_{args.table.split("/")[-1].rsplit("_", 1)[0]}.log'
    os.makedirs(args.res, exist_ok=True)
    os.makedirs(args.pics, exist_ok=True)

    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    df0 = pd.read_csv(args.table, sep='\t', compression='gzip')
    print(df0.columns.tolist())
    print(df0.shape)

    best_dict = {
        "de": ['acl', 'addit', 'advcl', 'advmod', 'case', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'parataxis', 'pastv', 'poss', 'relcl', 'ttr'],
        "en": ['addit', 'advers', 'advmod', 'case', 'compound', 'conj', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'numcls', 'obl']}
    # best_dict = {'de': ['advmod', 'ttr', 'pastv', 'mdd', 'nmod', 'parataxis', 'fin', 'advers', 'mhd', 'poss', 'numcls', 'nnargs', 'addit'],
    #              'en': ['advmod', 'xcomp', 'mean_sent_wc', 'advers', 'case', 'mhd', 'advcl', 'nmod', 'fin', 'numcls', 'obl', 'addit', 'sconj', 'compound']}

    for lang in ['de', 'en']:
        df = df0[df0['lang'] == lang]
        df = df[df['raw'].apply(lambda x: len(str(x).split()) > 8)]
        meta = ['doc_id', 'lang', 'ttype', 'seg_num', 'corpus', 'direction', 'wc_lemma', 'raw', 'raw_tok', 'sents']
        df = df.drop(meta, axis=1)
        print(df.head())

        # interestingly I get a 30 x30 heatmap out of 60 x 60 correlation matrix - wht is that?

        # limiting to the best-performing feature set:
        best_feats = best_dict[lang]
        # Select rows where values in feature column are in the list
        df = df[best_feats]

        # Calculate the correlation matrix
        correlation_matrix = df.corr()
        print(correlation_matrix.shape)

        # Reorder the features based on correlation
        # (optional, you can remove this if you don't want to reorder)
        feature_order = correlation_matrix.mean().sort_values().index
        reordered_corr_matrix = correlation_matrix[feature_order].reindex(feature_order)

        # Create a heatmap
        plt.figure(figsize=(8, 8))
        # Set font properties globally
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 14,
            'font.weight': 'normal'
        })

        ax = sns.heatmap(reordered_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidth=.5)

        # # x-labels to the top axis
        # ax.set(xlabel="", ylabel="")
        # ax.xaxis.tick_top()

        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

        if lang == 'de':
            plt.title('German')
        else:
            plt.title('English')

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        plt.savefig(f'{args.pics}{lang}_{args.level}_colinear_heatmap_{df.shape[1]}feats.png')

        plt.show()
