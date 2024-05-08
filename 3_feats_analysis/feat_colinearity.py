'''
22 Sept 2023

While waiting for re-parsed data
prepare to look at features colinearity

python3 3_feats_analysis/feat_colinearity.py
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


def make_dirs(outdir, logsto):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logsto, exist_ok=True)
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{script_name}.log'
    sys.stdout = Logger(logfile=log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', default='data/feats_tabled/seg-450-1500.feats.tsv.gz')
    parser.add_argument('--level', choices=['seg', 'doc'], default='doc')
    parser.add_argument('--logsto', default=f'logs/initial_analysis/')
    parser.add_argument('--pics', default='3_feats_analysis/pics/')
    args = parser.parse_args()

    make_dirs(args.pics, args.logsto)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    df0 = pd.read_csv(args.table, sep='\t', compression='gzip')
    print(df0.columns.tolist())
    print(df0.shape)

    best_dict = {
        "de": ['acl', 'addit', 'advcl', 'advmod', 'case', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'parataxis', 'pastv', 'poss', 'relcl', 'ttr'],
        "en": ['addit', 'advers', 'advmod', 'case', 'compound', 'conj', 'fin', 'mean_sent_wc', 'mhd', 'nmod', 'numcls', 'obl']}

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

        ax = sns.heatmap(reordered_corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', square=True, linewidth=.5)

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
