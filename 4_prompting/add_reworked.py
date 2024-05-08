"""
22 Nov 2023
enriching 4_prompting/output/ with _reworks/output/

* for _extra_rework setup: Don't for get to specify the _extra folder!!!
GPT-4 does not require these extra loop. You can have luckiy runs on GPT3.5-turbo, too
python3 4_prompting/add_reworked.py --temp 0.0 --setup lazy --reworks 4_prompting/_extra_reworks/output/gpt-3.5-turbo/

10-11 Mar 2024
python3 4_prompting/add_reworked.py --setup seg_self-guided_min --model gpt-4
python3 4_prompting/add_reworked.py --setup seg_translated_min --model gpt-4

"""

import argparse
import os
import sys
import time
from datetime import datetime
import shutil
import gzip

import pandas as pd


# this used to be a problem in early re-writing attempts
# TODO this needs to be better, and it still does not capture all patterns.
#  We curated the outputs (ratio2.5 and std2) manually in the end
def filter_llm_artefacts(df_in=None):
    df_in['rewritten'] = df_in['rewritten'].apply(lambda x: x.strip('"'))
    df_in['rewritten'] = df_in['rewritten'].str.replace("Here's a revised version of the translation:\n", '')
    df_in['rewritten'] = df_in['rewritten'].str.replace('"Here is a revised translation: ', '').str.replace('"Here is a revised English translation: ', '').str.replace(
        '"" Revised English translation: ""', '').str.replace('"""', '').str.replace('```', '')
    df_in['rewritten'] = df_in['rewritten'].str.replace('"Here is a revised German translation: ', '').str.replace(
        '"" Revised German translation: ""', '').str.replace('"""', '').str.replace('```', '')
    df_in['rewritten'] = df_in['rewritten'].str.replace('This is an original German text: ', '').str.replace(
        '" Revised German translation: ""', '').str.replace('""', '')
    df_in['rewritten'] = df_in['rewritten'].apply(
        lambda x: x.split("```")[1] if isinstance(x, str) and '```' in x else x)

    return df_in


def update_megatable(this_mega_table_path=None, expected_n=None, rewritten_dir=None):
    my_df = pd.read_csv(this_mega_table_path, sep='\t')
    print(my_df.head())

    # the chunks for this lang
    collector = []
    failed_chunks = []
    chunks_with_blanks = 0
    if expected_n != 0:
        for i in range(expected_n):
            i = i + 1
            try:  # this applies for reworked, I don't have this loop with GPT-4
                fh = [f for f in os.listdir(rewritten_dir) if f.startswith(f'new_chunk_{i}.')][0]
                df1 = pd.read_csv(rewritten_dir + fh, sep='\t')

                # this is done for counting chunks that have skipped segments in them
                nan_counts = df1['rewritten'].isna().sum()
                if nan_counts > 0:
                    chunks_with_blanks += 1
                collector.append(df1)
            except IndexError:
                failed_chunks.append(f'new_chunk_{i}.tsv')

        add_df = pd.concat(collector, axis=0)
        my_df = my_df.set_index('seg_id')
        empty_before = my_df['rewritten'].isna().sum()
        add_df = add_df.set_index('seg_id')
        print(add_df.shape)
        print(my_df.shape)
        add_df = add_df.dropna()
        my_df = my_df.dropna()
        print(add_df.shape)
        print(my_df.shape)

        my_df.update(add_df)

        empty_after = my_df['rewritten'].isna().sum()

        print(
            f'The number of empty segments in the final dataset ({my_df.shape}) changed from {empty_before} to {empty_after}')
        # get seg_id column back!
        my_df = my_df.reset_index()
        print(my_df.head())
        print(my_df.columns.tolist())

        # ratio of empty segments in updated my_df i.e. Null where the API timed out AGAIN
        nan_counts = my_df['rewritten'].isna().sum()
        my_ratio = round(nan_counts / my_df.shape[0] * 100, 2)
        print(my_ratio)

        # Select rows with NaN values in a column
        rework_input = my_df[my_df['rewritten'].isna()]
    else:
        rework_input = pd.DataFrame()
        my_ratio = 0

    return my_df, rework_input, my_ratio, chunks_with_blanks, failed_chunks


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
    parser.add_argument('--megaouts', default='4_prompting/input/')
    parser.add_argument('--reworks', default='4_prompting/_reworks/output/')
    parser.add_argument('--model', required=True, default="gpt-4")  # gpt-3.5-turbo, gpt-4
    parser.add_argument('--temp', choices=['0.7'], default=0.7, help='kept here for path-retrieval consistency')
    parser.add_argument('--setup', choices=['seg_self-guided_min', 'seg_self-guided_detailed',
                                            'seg_feature-based_min_ratio2.5', 'seg_feature-based_detailed_ratio2.5',
                                            'seg_feature-based_min_std2', 'seg_feature-based_detailed_std2',
                                            'seg_translated_min'],
                        help='for path re-construction', required=True)
    parser.add_argument('--res', default='4_prompting/output/')
    parser.add_argument('--logs', default='logs/collecting_rewritten/')

    args = parser.parse_args()

    start = time.time()
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')

    mega_outdir = f'{args.res}{args.model}/'
    os.makedirs(mega_outdir, exist_ok=True)

    os.makedirs(args.logs, exist_ok=True)

    log_file = f'{args.logs}{formatted_datetime.split("_")[0]}_{sys.argv[0].split("/")[-1].split(".")[0]}.log'

    sys.stdout = Logger(logfile=log_file)

    print(f"\nRun date, UTC: {formatted_datetime}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    for tlang in ['de', 'en']:
        path_to_rewritten = f"{args.reworks}{args.model}/temp{args.temp}/{tlang}_{args.setup}/"
        megaouts_infile = f'{args.megaouts}{args.model}/temp{args.temp}/{tlang}_{args.setup}.tsv'
        mega_pathname = f'{mega_outdir}{tlang}_{args.setup}.tsv.gz'
        if 'extra_reworks' in args.reworks:
            input_dir = path_to_rewritten.replace('output/', 'input/')
        else:
            input_dir = path_to_rewritten.replace('output/', 'input/')
        print(f'Input for {megaouts_infile} is originally from {input_dir}, it was reworked to {path_to_rewritten}')
        try:
            input_n_chunks = len([f for f in os.listdir(input_dir)])
            print(tlang, input_n_chunks)
            this_lang_df, this_lang_reworks, skipped_segs, faulty_chunks, send_back_lst = update_megatable(
                this_mega_table_path=megaouts_infile,
                expected_n=input_n_chunks,
                rewritten_dir=path_to_rewritten)

            # filter rewriting artefacts: Revised translation: "" .""" ```
            this_lang_df = filter_llm_artefacts(df_in=this_lang_df)

            print(this_lang_df.head(2))
            print(this_lang_df.tail(2))
            print(this_lang_df.shape)
            print(this_lang_df.columns.tolist())
            # Replace NaN values with an empty string
            this_lang_df['rewritten'] = this_lang_df['rewritten'].fillna('')

            this_lang_df['rewritten'] = this_lang_df['rewritten'].apply(lambda x: x.strip('"').strip('```'))

            # If you want to remove leading and trailing whitespaces after the replacements
            this_lang_df['rewritten'] = this_lang_df['rewritten'].str.strip()

            # Replace empty strings with NaN
            this_lang_df['rewritten'] = this_lang_df['rewritten'].replace('', pd.NA)

            this_lang_df.to_csv(mega_pathname, sep='\t', index=False, compression='gzip')

            print(
                f'REWORKS: API errors in {tlang}_{args.setup}: {this_lang_reworks.shape[0]} ({skipped_segs}% of total segments), '
                f'{faulty_chunks} chunks with blanks, {len(send_back_lst)} skipped entirely')
            print(f'\n====== New language ======\n')
            # if skipped_segs > 5:
            #     print(f'(0) Leave only these chunks in {input_dir}: {send_back_lst}')
            #     print('(1) Run sh prompt/de_feeding_chunks.sh de_lazy 0.3 reworks')
        except FileNotFoundError:
            print(f'There is nothing to rework for {tlang.upper()}')
            # copy over the file for this language from _megaouts/input/ and gzip it
            # Open the source file, read its content, and write it to a gzipped file
            with open(megaouts_infile, 'rb') as f_in, \
                    gzip.open(megaouts_infile.replace('input', 'output').replace('.tsv', '.tsv.gz'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
