"""
27 October 2023

python3 get_iate_bilingual_dictionaries.py --indir glossaries/iate_dicts/ --lpair ende
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

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
    # default='data/iate_dicts/'
    parser.add_argument('--indir', help="a folder with pairs of files *.en.txt and *.de.txt", required=True)
    parser.add_argument('--lpair', required=True, choices=['deen', 'ende', 'esen', 'enes'])
    parser.add_argument('--res', default='glossaries/iate2hunalign/')
    parser.add_argument('--logs', default='logs/bi_dict/')

    args = parser.parse_args()

    start = time.time()
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')

    os.makedirs(args.res, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    log_file = f'{args.logs}{args.lpair}_{formatted_datetime.split("_")[0]}_{sys.argv[0].split("/")[-1].split(".")[0]}.log'

    sys.stdout = Logger(logfile=log_file)

    print(f"\nRun date, UTC: {formatted_datetime}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    slang = args.lpair[:2]
    tlang = args.lpair[2:]
    print(slang, tlang)

    my_dicts = [i for i in os.listdir(args.indir) if i.startswith(f"{slang}") or i.startswith(f"{tlang}")]
    print(my_dicts)

    dfs = []
    langs = defaultdict()
    for i in my_dicts:
        df = pd.read_csv(args.indir + i, sep='|', on_bad_lines='skip')
        df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
        # print(df.head())
        # print(len(df.columns.tolist()))
        df = df[['E_ID', 'T_TERM']]
        # Check for duplicate indices
        duplicate_mask = df['E_ID'].duplicated(keep='first')

        # Use the mask to filter the DataFrame
        dedup_df = df[~duplicate_mask]

        dedup_df = dedup_df.set_index('E_ID')
        lang = i.split('_')[0]

        if lang == tlang:
            dedup_df = dedup_df.rename(columns={'T_TERM': 'tgt_T_TERM'})
            langs['tlang'] = lang
        else:
            dedup_df = dedup_df.rename(columns={'T_TERM': 'src_T_TERM'})
            langs['slang'] = lang
        print(lang)
        print(dedup_df.shape)
        dfs.append(dedup_df)

    my_merged = pd.concat(dfs, axis=1)
    my_merged = my_merged.dropna()
    print(my_merged.head())
    print(my_merged.shape)

    my_merged = my_merged[['tgt_T_TERM', 'src_T_TERM']]

    with open(f'{args.res}{langs["slang"]}-{langs["tlang"]}.dic', 'w') as outdic:
        for tgt, src in zip(my_merged['tgt_T_TERM'], my_merged['src_T_TERM']):
            outdic.write(f'{tgt} @ {src}\n')

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
