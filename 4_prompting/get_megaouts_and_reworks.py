"""
22 Nov 2023

this script outputs:
(1) mega tables with rewritten containing blanks for the segments that need to be re-worked
    inc. for earlier versions of the re-writing pipeline where the input to rewriting needs to be added to the output.
    Namely, the script tests whether the output tables have 2 or more columns: 21 Nov I changed the output of (lang)_api_prompting.py to include everything
    but I don't want to re-infer with GPT
(2) one input table per run with segments to be reworked e.g. 4_prompting/_reworks/re_temp0.3_de_lazy.tsv

I need a separate script which would take the reworked segments and insert them
instead on blanks into respective tables in 4_prompting/_megaouts/out_temp0.3_de_lazy.tsv

USAGE:
python3 prompt/get_megaouts_and_reworks.py --chunked_output 4_prompting/chunked_output/gpt-4/ --temp 0.3 --setup lazy
"""

import argparse
import ast
import os
import sys
import time
from datetime import datetime

import pandas as pd


def split_tsv_into_chunks(input_df, output_directory=None, chunk_size=None):
    num_rows = len(input_df)
    num_chunks = num_rows // chunk_size  # Calculate the number of full chunks

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_df = input_df.iloc[start_idx:end_idx]

        output_filename = os.path.join(output_directory, f'chunk_{i + 1}.tsv')
        chunk_df.to_csv(output_filename, sep='\t', index=False)

    # Handle the remainder of rows in the last chunk
    if num_rows % chunk_size != 0:
        start_idx = num_chunks * chunk_size
        chunk_df = input_df.iloc[start_idx:]
        output_filename = os.path.join(output_directory, f'chunk_{num_chunks + 1}.tsv')
        chunk_df.to_csv(output_filename, sep='\t', index=False)


def align_with_input(prechunk_dir=None, rewritten_dir=None, expected_n=None, my_lang=None, my_mode=None):
    # the chunks for this lang
    collector = []
    failed_chunks = []
    chunks_with_blanks = 0
    for i in range(expected_n):
        i = i + 1
        try:
            fh = [f for f in os.listdir(rewritten_dir) if f.startswith(f'chunk_{i}.')][0]
            # for f in fh:
            #         df1 = pd.read_csv(rewritten_dir + f, sep='\t')
            df1 = pd.read_csv(rewritten_dir + fh, sep='\t')
            # this is done for counting chunks that have skipped segments in them
            nan_counts = df1['rewritten'].isna().sum()
            if nan_counts > 0:
                chunks_with_blanks += 1
            collector.append(df1)
        except IndexError:
            failed_chunks.append(f'chunk_{i}.tsv')
    # fh = [f for f in os.listdir(rewritten_dir) if f.startswith('new_chunk_')]
    # for f in fh:
    #     df1 = pd.read_csv(rewritten_dir + f, sep='\t')
    #     # this is done for counting chunks that have skipped segments in them
    #     nan_counts = df1['rewritten'].isna().sum()
    #     if nan_counts > 0:
    #         failed_chunks += 1
    #     collector.append(df1)
    add_df = pd.concat(collector, axis=0)

    # testing for the number of columns: 2 or 6?
    if add_df.shape[1] > 2:
        this_lang = add_df
        print(f'---- The input is formatted in a *6-column* format ----')
    else:
        print(f'---- The input is formatted in a *2-column* format ----')
        # the initial file:
        my_file = [f for f in os.listdir(f'{prechunk_dir}') if f.startswith(f'{my_lang}_{my_mode}.tsv')][0]
        my_df = pd.read_csv(f'{prechunk_dir}{my_file}', sep='\t')

        my_df = my_df.set_index('seg_id')
        add_df = add_df.set_index('seg_id')
        try:
            this_lang = pd.concat([my_df, add_df], axis=1)
            print(my_df.shape, add_df.shape, this_lang.shape)
        except pd.errors.InvalidIndexError:
            print(add_df.shape)
            # Merge the duplicates based on 'seg_id': Not sure where these dups come from: multiple chunking-overwriting
            add_df = add_df.groupby('seg_id').aggregate({'rewritten': 'first'})
            print(add_df.shape)
            this_lang = pd.concat([my_df, add_df], axis=1)
            input('Achtung! This should not be happening ...')

    this_lang.insert(1, 'lang', my_lang)

    # get seg_id column back!
    this_lang = this_lang.reset_index()

    # ratio of empty segments in concatenated add_df i.e. Null where the API timed out
    nan_counts = this_lang['rewritten'].isna().sum()
    my_ratio = round(nan_counts / this_lang.shape[0] * 100, 2)

    # Select rows with NaN values in a column
    rework_input = this_lang[this_lang['rewritten'].isna()]

    return this_lang, rework_input, my_ratio, chunks_with_blanks, failed_chunks


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
    parser.add_argument('--all_prompts', help='initial unchunked files', default='4_prompting/ol_prompts/')
    parser.add_argument('--chunked_output', help='chunked output to add to big table',
                        default='4_prompting/chunked_output/')
    parser.add_argument('--model', required=True)  # gpt-3.5-turbo, gpt-4
    parser.add_argument('--temp', choices=['0.7'], default=0.7)  # '0.0', '0.3', '0.5',
    parser.add_argument('--setup', choices=['seg_self-guided_min', 'seg_self-guided_detailed',
                                            'seg_feature-based_min_ratio2.5', 'seg_feature-based_detailed_ratio2.5',
                                            'seg_feature-based_min_std2', 'seg_feature-based_detailed_std2',
                                            'seg_translated_min'],
                        help='for path re-construction', required=True)

    parser.add_argument('--megaouts', default='4_prompting/input/')
    parser.add_argument('--n_segs', help='number of segs per chunk in the re-working loop', default=10)
    parser.add_argument('--reworks', default='4_prompting/_reworks/input/')
    parser.add_argument('--logs', default='logs/collecting_rewritten/')

    args = parser.parse_args()

    start = time.time()
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')

    mega_outdir = f'{args.megaouts}{args.model}/temp{args.temp}/'
    os.makedirs(mega_outdir, exist_ok=True)
    reworks_outdir = f'{args.reworks}{args.model}/temp{args.temp}/'
    os.makedirs(reworks_outdir, exist_ok=True)

    os.makedirs(args.logs, exist_ok=True)

    log_file = f'{args.logs}{formatted_datetime.split("_")[0]}_{args.setup}_{sys.argv[0].split("/")[-1].split(".")[0]}.log'

    sys.stdout = Logger(logfile=log_file)

    print(f"\nRun date, UTC: {formatted_datetime}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    for tlang in ['de', 'en']:  #
        path_to_rewritten = f"{args.chunked_output}{args.model}/temp{args.temp}/{tlang}_{args.setup}/"
        # print(path_to_rewritten)
        # exit()

        mega_pathname = f'{mega_outdir}{tlang}_{args.setup}.tsv'
        if 'reworks' in args.chunked_output:
            input_dir = path_to_rewritten.replace(f'output/', f'input/')
        else:
            input_dir = path_to_rewritten.replace(f'output/{args.model}/temp{args.temp}/', f'input/')
        print(f'Input for {mega_pathname} is from {input_dir}')

        input_n_chunks = len([f for f in os.listdir(input_dir)])
        print(input_n_chunks)
        if 'reworks' in args.chunked_output:
            this_lang_df, this_lang_reworks, skipped_segs, faulty_chunks, send_back_lst = align_with_input(
                prechunk_dir=f"4_prompting/_reworks/input/{args.model}/",
                rewritten_dir=f"{path_to_rewritten}",
                expected_n=input_n_chunks,
                my_lang=tlang,
                my_mode=args.setup)
        else:
            this_lang_df, this_lang_reworks, skipped_segs, faulty_chunks, send_back_lst = align_with_input(
                prechunk_dir=args.all_prompts,
                rewritten_dir=f"{path_to_rewritten}",
                expected_n=input_n_chunks,
                my_lang=tlang,
                my_mode=args.setup)

        print(this_lang_df.head(2))
        print(this_lang_df.tail(2))
        print(this_lang_df.shape)
        print(this_lang_df.columns.tolist())
        # Replace NaN values with an empty string to delete """ if they are inserted
        this_lang_df['rewritten'] = this_lang_df['rewritten'].fillna('')

        this_lang_df['rewritten'] = this_lang_df['rewritten'].apply(lambda x: x.strip('"').strip('```'))

        # If you want to remove leading and trailing whitespaces after the replacements
        this_lang_df['rewritten'] = this_lang_df['rewritten'].str.strip()

        # Replace empty strings with NaN
        this_lang_df['rewritten'] = this_lang_df['rewritten'].replace('', pd.NA)

        # this is not needed for extra_reworks, I will add them to the first reworking collectors  in prompt/_megaouts/input/
        if 'reworks' not in args.chunked_output:
            this_lang_df.to_csv(mega_pathname, sep='\t', index=False)

        # reworks need to be chunked again!
        reworks_outdir_deeper = f'{reworks_outdir}{tlang}_{args.setup}/'
        os.makedirs(reworks_outdir_deeper, exist_ok=True)
        split_tsv_into_chunks(this_lang_reworks, output_directory=reworks_outdir_deeper, chunk_size=args.n_segs)

        print(f'REWORKS: {args.model} API errors in {tlang}_{args.setup}: {this_lang_reworks.shape[0]} ({skipped_segs}% of total segments), '
              f'{faulty_chunks} chunks with blanks, {len(send_back_lst)} skipped entirely')

        # if skipped_segs > 5:
        #     print(f'(0) Leave only these chunks in {input_dir}: {send_back_lst}')
        #     print('(1) Run sh prompt/(extra)_feeding_chunks.sh lazy 0.3 reworks')

        print(f'\n====== New language ======\n')

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
