"""
27 Nov 2023
two separate script might be useful for the initial feeding, but for reworks maybe use this copy.
It is called from feeding_chunks.sh which loop langs

20 Nov 2023
I have two identical copies of the *_api_prompting.py script for separate runs with the OpenAI API, one for each lang
pip install openai==0.28

7 October 2023
feeding ready-made prompts, chunk-by-chunk via feeding_chunks.sh
I am not sure how to arrange logging

requires a separate run for each language

This script is supposed to be called from a shell script for each of the 25-segments-long chunks (out of 86 for DE and 75 for EN)

The chunks which timed_out need to be copied over to a timeout/de_lazy/ or timeout/en_lazy/ and run again from bash script
see ids of timed out chunks in logs: prompt/logs/feeding/
"""

import sys
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import argparse
import openai
import os

# Set your OpenAI API key
# openai.api_key = "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"  # Koels
openai.api_key = "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"  # coling15
# openai.api_key = "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"  # Koels
# openai.api_key = "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"  # Koels 10 March


def generate_output(prompt, model_name, temperature=None):
    # print("Python version")
    # print(sys.version)
    # print("Version info.")
    # print(sys.version_info)
    try:
        response = openai.ChatCompletion.create(
            model=model_name,  # Specify your chat model here
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=150,  # Adjust the max tokens as needed
        )

        rewritten_text = response.choices[0].message["content"].strip()
    # for openai==1.3.4; openai=0.28.1 was fine with except openai.error.Timeout as e:
    # except openai.Error as e:
    #     # Handle OpenAIError
    #     print(f"An OpenAIError occurred: {str(e)}")
    #     rewritten_text = None
    except Exception as e:
        # Handle other types of exceptions
        print(f"An unexpected error occurred: {str(e)}")
        rewritten_text = None
    return rewritten_text


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
    parser.add_argument('--chunk_path', help="fed by feeding_chunks.sh", required=True)
    parser.add_argument('--model', required=True)  # gpt-3.5-turbo, gpt-4
    parser.add_argument('--tempr', type=float, required=True)
    parser.add_argument('--lang', required=True)
    # this is changed to add flexibility to the re-use of the pipeline for re-working
    parser.add_argument('--res', required=True)  # default='prompt/chunked_output/new/'
    # parser.add_argument('--logs', default='prompt/logs/')

    args = parser.parse_args()

    start = time.time()

    os.makedirs(args.res, exist_ok=True)
    # os.makedirs(args.logs, exist_ok=True)
    # log_file = f'{args.logs}{args.lang}_{args.chunk_path.split("/")[-2]}_{sys.argv[0].split("/")[-1].split(".")[0]}.log'
    # sys.stdout = Logger(logfile=log_file)
    #
    # print(f"\nRun date, UTC: {datetime.utcnow()}")
    # print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    my_input = pd.read_csv(args.chunk_path, sep='\t')
    # seg_id	src_raw	tgt_raw	prompt	prompt_len	thresholds
    id_text_dict = dict(zip(my_input['seg_id'], zip(my_input['src_raw'],
                                                    my_input['tgt_raw'],
                                                    my_input['prompt'],
                                                    my_input['thresholds'])))

    output_dict = defaultdict(list)
    for seg_id, (src_seg, tgt_seg, my_prompt, thres) in id_text_dict.items():

        if my_prompt == "bypassed re-writing pipeline" or my_prompt == "copy over the translation intact":
            new_seg = tgt_seg
        else:
            new_seg = generate_output(my_prompt, args.model, temperature=args.tempr)

        output_dict['seg_id'].append(seg_id)
        output_dict['src_raw'].append(src_seg)
        output_dict['tgt_raw'].append(tgt_seg)
        output_dict['rewritten'].append(new_seg)
        output_dict['prompt'].append(my_prompt)
        output_dict['thresholds'].append(thres)

    out_df = pd.DataFrame(output_dict)

    bits = args.chunk_path.split("/")
    fn = bits[-1]
    dirname = bits[-2]
    print(f'{dirname}', file=sys.stderr)

    outdir = f'{args.res}{args.model}/temp{args.tempr}/{dirname}/'
    os.makedirs(outdir, exist_ok=True)
    out_df.to_csv(f'{outdir}{fn}', sep='\t', index=False)

    endtime_tot = time.time()
    print(f'{args.lang}, {args.chunk_path.split("/")[-1]} ({out_df.shape[0]}): {((endtime_tot - start) / 60):.2f} min\n')
