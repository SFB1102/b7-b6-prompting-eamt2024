"""
8 Oct 2023

Chunk the input table into files containing max 30 segments to try and fit into the timeout window

python3 prompt/chunking.py --tables prompt/ol_prompts/new/ --setup None

"""

import os
import time
import pandas as pd
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # prompt/ol_prompts/ or prompt/ol_prompts/docs/
    parser.add_argument('--tables', help="tsv with src,tgt,prompts,thresholds", default='prompt/ol_prompts/new/', required=True)
    parser.add_argument('--n_segs', default=15)  # en 74 chunks + 13 segs in chunk 75; de 85 chunks + 6 segs in #86
    parser.add_argument('--setup', help="pass specific setups to chunk, e.g. seg_lazy or 'None' for chunking all files with prompts in one go", required=True)
    parser.add_argument('--res', default='prompt/chunked_input/new/')

    args = parser.parse_args()

    start = time.time()

    os.makedirs(args.res, exist_ok=True)

    fh = [f for f in os.listdir(f'{args.tables}') if f.endswith('.tsv')]
    if args.setup == 'None':
        for my_file in fh:
            my_df = pd.read_csv(f'{args.tables}{my_file}', sep='\t')

            print(my_df.head(3))

            setup = my_file.rsplit('.', 1)[0]

            outto = f'{args.res}/{setup}/'
            os.makedirs(outto, exist_ok=True)

            split_tsv_into_chunks(my_df, output_directory=outto, chunk_size=args.n_segs)
    else:
        my_lang_files = [f for f in fh if args.setup in f]
        print(fh)
        print(args.setup)
        print(my_lang_files)

        assert len(my_lang_files) == 2, 'Huston, we have got problems!'

        for my_lang in my_lang_files:
            my_df = pd.read_csv(f'{args.tables}{my_lang}', sep='\t')

            print(my_df.head(3))

            setup = my_lang.rsplit('.', 1)[0]

            outto = f'{args.res}/{setup}/'
            os.makedirs(outto, exist_ok=True)

            split_tsv_into_chunks(my_df, output_directory=outto, chunk_size=args.n_segs)

