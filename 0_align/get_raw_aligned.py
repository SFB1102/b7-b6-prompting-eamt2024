"""
25 Oct 2023
NB! Change in ST naming conventions: ORG_WR_EN_DE_000001 instead of ORG_WR_EN_000001
with the translations naming intact: TR_EN_DE_000001

24 Oct 2023
re-aligned DEEN and ENDE: I don't keep the intermediary output of LFAlighner for considerations of space
Instead, all sent- and doc-aligned items are collected into a single table for each translation direction:
e.g.
_data/raw_aligned/deen_wide2018_cap0_score0.3.tsv.gz

python3 get_raw_aligned.py --indir lf_aligned_xls/deen_align_2023.10.28/ --lpair deen --meta data/raw_aligned/meta/ --docsize 0 --cutoff 0.3

"""

import argparse

import numpy as np
from tqdm import tqdm
import os
import sys
import time

from sklearn.preprocessing import StandardScaler
from datetime import datetime
from collections import defaultdict

import pandas as pd
import pickle


def min_max_normalization(values):
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(x - min_val) / (max_val - min_val) for x in values]

    return normalized_values


def zcoring(values):
    sc = StandardScaler()
    values = values.reshape(-1, 1)
    standardized_scores = sc.fit_transform(values)
    return standardized_scores


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
    # data/helper_files/aligned_de/ --- not an option anymore!
    parser.add_argument('--indir', help="lf_aligned_xls/deen_align_2023.10.24", required=True)
    parser.add_argument('--version', choices=['2018'], help='kept for historical reasons', default=0)
    parser.add_argument('--norming', type=int, default=0)
    parser.add_argument('--cutoff', type=float, required=True)  # use 0.3 for deen, 0.5 for ende
    parser.add_argument('--norm', choices=['zscore', 'minmax'], default='zscore')
    parser.add_argument('--docsize', type=int, required=True)
    parser.add_argument('--mode', choices=['anew', 'legacy'], default='anew',
                        help='anew for deen, legacy implies aligned data (ende), kept for historical reasons')
    # data/meta/europ/  which has files like: de_deen_europ_meta.txt.gz
    parser.add_argument('--meta', help='I need meta to try and get existing ids like OGR_WR_DE_00001', required=True)
    parser.add_argument('--lpair', choices=['deen', 'ende', 'esen', 'enes'], required=True)
    parser.add_argument('--res', default='../data/raw_aligned/')
    parser.add_argument('--logs', default='logs/get_bitext/')

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

    # convert meta to a dict
    # data/europ/de_deen_europ_meta.txt.gz
    slang = args.lpair[:2]
    tlang = args.lpair[-2:]
    print(slang, tlang)
    if 'es' in args.lpair:
        current_id = "000000"
        current_id_number = int(current_id)
        src_keyed_tuples = None
    else:
        os.makedirs('temp/', exist_ok=True)
        dict_name = f'temp/{slang}_{args.lpair}_keyed_tuples.pkl'

        if os.path.exists(dict_name):
            # Load
            src_keyed_tuples = pickle.load(open(dict_name, 'rb'))
            print('The mapping dict is loaded from saved version')
        else:
            # Open the compressed file and read its lines
            src_keyed_tuples = defaultdict()
            my_src_meta_file = f'{args.meta}{slang}_{args.lpair}_europ_meta.txt.gz'
            src_meta_df = pd.read_csv(my_src_meta_file, sep='\t', compression='gzip')
            # print(my_src_meta_file)
            # input()
            # print(src_meta_df.head())
            # tgt_meta_df = pd.read_csv(f'{args.meta}{tlang}_{args.lpair}_europ_meta.txt.gz', sep='\t', compression='gzip')
            src_metas = src_meta_df.meta.tolist()
            for i in src_metas:
                # my_id="ORG_WR_DE_000009"
                did = i.split('my_id="')[1].split('"')[0]
                # text id="3_031"
                interven = i.split('text id="')[1].split('"')[0].strip()

                # date="1999-07-21" day_id="19990721_de"
                day = i.split('day_id="')[1].split(f'_{slang}"')[0]
                # print(did, interven, day)
                src_keyed_tuples[did] = (interven, day)
            # Save
            pickle.dump(src_keyed_tuples, open(dict_name, 'wb'), pickle.HIGHEST_PROTOCOL)
            print(f'The mapping dict is generated anew and saved to {dict_name}')

        my_ids_bases = [i.split('_')[-1].strip() for i in src_keyed_tuples.keys()]
        sorted_my_ids_bases = sorted(my_ids_bases)
        print(sorted_my_ids_bases[:5])
        print(sorted_my_ids_bases[-5:])

        current_id = sorted_my_ids_bases[-1]
        current_id_number = int(sorted_my_ids_bases[-1])

    counter = 0
    errors = 0
    # getting the data tables from re-aligned documents
    entire = []
    # qua_after_filtering_lst = []
    align_qua_score_lst = []
    xls_files = [f for f in os.listdir(args.indir) if f.endswith('.xls')]
    for xls_file in tqdm(xls_files, position=0, leave=True):
        # print(xls_file)
        xls_path = os.path.join(args.indir, xls_file)
        my_date = xls_file.split(f'{slang}-')[0].split('.')[0]
        my_interven = xls_file.split(f'{slang}-')[0].split('.')[1].replace('-', '_')  # 4_087_000, 1_144
        # print(my_date, my_interven)

        # Read the .xls file into a DataFrame
        # http://mokk.bme.hu/resources/hunalign/
        # hunalign_qua=quality metric based on Gale and Church method based on sentence length in characters
        df = pd.read_excel(xls_path, names=['sseg', 'tseg', 'hunalign_qua', 'docpair'])
        df = df.reset_index()
        # makes it a 0-based numbering of segments in a doc in europ!
        # <seg num="1" sent_align="0">
        df = df.rename(columns={'index': 'seg_num'})

        if not df.empty:
            align_qua_score = np.nanmean(df.hunalign_qua.tolist())
            align_qua_score_lst.append(align_qua_score)
            # print(type(align_qua_score))
            # if align_qua_score >= 0.5:
            #     qua_after_filtering_lst.append(align_qua_score)

            if 'es' in args.lpair:
                # generating ORG_WR_ES_EN_
                current_id_number += 1
                # Convert the new ID back to a string with leading zeros
                new_id = str(current_id_number).zfill(len(current_id))
                # print(current_id_number, new_id)
                # dont add _{tlang.upper()} quite yet, see below!
                df.insert(0, 'sdoc_id', f'ORG_WR_{slang.upper()}_{new_id}')
                # print(df.head())
                counter += 1
            else:
                # adding the src_id to rows with aligned segments
                for did, (interven, day) in src_keyed_tuples.items():
                    # print(did, interven, day)
                    if (day == my_date) and (interven.strip() == my_interven.strip()):
                        df.insert(0, 'sdoc_id', did)
                if 'sdoc_id' not in df.columns.tolist():
                    # continue
                    current_id_number += 1
                    # Convert the new ID back to a string with leading zeros
                    new_id = str(current_id_number).zfill(len(current_id))
                    df.insert(0, 'sdoc_id', f'ORG_WR_{slang.upper()}_{new_id}')
                    counter += 1

            # change of original doc-naming conventions:
            df['sdoc_id'] = df['sdoc_id'].apply(lambda x: x.replace(f'ORG_WR_{slang.upper()}_', f'ORG_WR_{slang.upper()}_{tlang.upper()}_'))
            # print(df.sdoc_id.tolist()[:5], df.sdoc_id.tolist()[-5:])

            df = df.astype({'sdoc_id': 'str', 'seg_num': 'str'})
            # try:
            df['sseg_id'] = df[["sdoc_id", "seg_num"]].apply(lambda x: "-".join(x), axis=1)  # use hyphon, not colon
            # except ValueError:

            df = df.drop(["seg_num", 'docpair'], axis=1)
            df = df[["sdoc_id", 'sseg_id', 'sseg', 'tseg', 'hunalign_qua']]

            if args.docsize:
                # filter out document pairs with less than N segment pairs
                if df.shape[0] <= args.docsize:
                    continue
                else:
                    entire.append(df)
            else:
                entire.append(df)
        else:
            errors += 1
            continue
    wide = pd.concat(entire)
    if args.norming:
        # normalise alignment score and skip documents with average alignment < 0.5
        if args.norm == 'minmax':  # this gives me narrow spread of values between 0.0 and 0.27 for deen
            normed = min_max_normalization(wide.hunalign_qua.tolist())
            wide.insert(1, f'normed_score={args.norm}', normed)
            # z-score: A value of 0 will represent the mean,
            # while values greater or less than 0 will indicate the number of standard deviations away from the mean.
            # Mean: 0.00, Max: 5.80, Min: -2.71 for deen I am not sure how to interpret the negative scores
            # working with scaled values do not make much sense
        else:
            normed = zcoring(wide['hunalign_qua'].values)
            wide.insert(1, f'normed_score={args.norm}', normed)
        print(wide.head())
        print(wide.tail())
        print(wide.shape)

        temp_wide = wide.drop(['sseg_id', 'sseg', 'tseg', 'hunalign_qua'], axis=1)
        temp_doc_wide = temp_wide.groupby('sdoc_id', as_index=False).aggregate({f'normed_score={args.norm}': 'mean'})
        print(temp_doc_wide.head())
        # print(temp_doc_wide.shape)
        print(f'{args.lpair}: Parameters of the Min-Max normed scores across {temp_doc_wide.shape[0]} docs:\n'
              f'Mean: {np.mean(temp_doc_wide[f"normed_score={args.norm}"].tolist()):.2f}\n'
              f'Max: {np.max(temp_doc_wide[f"normed_score={args.norm}"].tolist()):.2f}\n'
              f'Min: {np.min(temp_doc_wide[f"normed_score={args.norm}"].tolist()):.2f}')

        temp_doc_wide_filtered = temp_doc_wide[temp_doc_wide[f"normed_score={args.norm}"] >= args.cutoff]
        good_docs = temp_doc_wide_filtered.doc_id.tolist()

        print(f'Number of docs with *normed* alignment score >=0.5: {len(good_docs)}')
        wide_filtered = wide[wide.doc_id.isin(good_docs)]
        print(wide_filtered.head())
        print(wide_filtered.shape)
        exit()
    else:
        temp_wide = wide.drop(['sseg_id', 'sseg', 'tseg'], axis=1)
        temp_doc_wide = temp_wide.groupby('sdoc_id', as_index=False).aggregate({'hunalign_qua': 'mean'})
        print(temp_doc_wide.head())
        # print(temp_doc_wide.shape)
        print(f'{args.lpair}: Parameters of the hunalign scores across {temp_doc_wide.shape[0]} docs:\n'
              f'Mean: {np.mean(temp_doc_wide.hunalign_qua.tolist()):.2f}\n'
              f'Max: {np.max(temp_doc_wide.hunalign_qua.tolist()):.2f}\n'
              f'Min: {np.min(temp_doc_wide.hunalign_qua.tolist()):.2f}')

        print(f'My hunalign score cutoff for {args.lpair} is {args.cutoff}')

        temp_doc_wide_filtered = temp_doc_wide[temp_doc_wide["hunalign_qua"] >= args.cutoff]
        good_docs = temp_doc_wide_filtered.sdoc_id.tolist()

        print(f'Number of docs with alignment score >={args.cutoff}: {len(good_docs)} (out of {temp_doc_wide.shape[0]},'
              f' {len(good_docs)/temp_doc_wide.shape[0]*100:.2f}%)')
        wide_filtered = wide[wide.sdoc_id.isin(good_docs)]
        print(wide_filtered.head())
        print(wide_filtered.shape)

    print(f'Total number of document pairs (>={args.docsize} segment pairs): {len(entire)}')
    # av_align_score = np.nanmean(qua_after_filtering_lst)
    print(f'\nMin: {np.min(align_qua_score_lst):.2f},\nMax: {np.max(align_qua_score_lst):.2f}')

    # Count rows where the 'ss' column contains '--'
    # count = wide['seg_id'].str.contains('--').sum()
    if 'es' in args.lpair:
        print(f'{args.lpair.upper()}: Number of doc pairs (new doc-ids generated): {counter}')
    else:
        print(f'{args.lpair.upper()}: Number of newly-aligned doc pairs NOT matching the preexisting doc_ids (new doc-ids generated): {counter}')
    print(f'More errors of unclear etiology: {errors} (empty files, skipped)')

    wide_raw_to = 'data/wide_raw/europ/'
    os.makedirs(wide_raw_to, exist_ok=True)
    wide_path = f'{wide_raw_to}{args.lpair}_wide{args.version}_cap{args.docsize}_score{args.cutoff}.tsv.gz'
    # add tseg_id column required by the parser
    tseg_ids = [i.replace('ORG_WR_', 'TR_') for i in wide_filtered.sseg_id.tolist()]
    wide_filtered.insert(4, 'tseg_id', tseg_ids)

    # test for empty strings in sseg, tseg and drop the rows if any of them is empty
    print(wide_filtered.shape)
    wide_filtered = wide_filtered.drop(['sseg', 'tseg'], axis=1)
    print(wide_filtered.shape)
    wide_filtered.to_csv(wide_path, sep='\t', index=False, compression='gzip')
    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
