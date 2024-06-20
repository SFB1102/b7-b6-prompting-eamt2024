'''
01 October 2023

test translationese hypothesis and produce cross-linguistic examples for translationese indicators

python3 humeval/interactive_analysis.py --temp 0.7 --setup seg_detailed_triple_vratio2.5 --sample_size 25 --model gpt-4 --who heike
detailed German mystery
python3 humeval/interactive_analysis.py --temp 0.7 --setup seg_self-guided_detailed --sample_size 25 --model gpt-4 --who heike
python3 humeval/interactive_analysis.py --temp 0.7 --setup seg_translated_min --sample_size 25 --model gpt-4 --who heike
'''
import math
import re

import numpy as np
import os
import sys
import pandas as pd
import argparse
from ast import literal_eval
import time
import random
from datetime import datetime
from collections import Counter, OrderedDict
from collections import defaultdict
from operator import itemgetter


# Function to apply the regex substitution
def process_cell(cell_value):
    global pattern, match_counter
    match = pattern.search(cell_value)
    if match:
        match_counter += 1
        return pattern.sub('', cell_value)
    else:
        return cell_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # analysis/res2/thres_rethres/gpt-4_temp0.7_seg_detailed_triple_vratio2.5_seg_src_tgt_rewritten_thres_rethres.tsv
    parser.add_argument('--indir', help="for each setup: three versions of segs from top most_translated docs",
                        default='analysis/res2/thres_rethres/new/')  # from parse_tabulate_rewritten
    parser.add_argument('--model', required=True)  # gpt-3.5-turbo, gpt-4
    parser.add_argument('--temp', choices=['0.0', '0.3', '0.5', '0.7'], required=True)
    parser.add_argument('--setup', choices=['seg_feature-based_min', 'seg_feature-based_detailed', 'seg_translated_min',
                                            'seg_self-guided_min', 'seg_self-guided_detailed', "seg_lazy", "seg_expert",
                                            "seg_min_triad_vratio2.5", "seg_detailed_triple_vratio2.5"],
                        help='for path re-construction', required=True)
    # parser.add_argument('--setup', choices=["lazy", "min_srclos_vratio2", "min_tgtlos_vratio2",
    #                                         "min_triad_vratio2"], help='for path re-construction', required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--who', choices=['me', 'heike'], default='heike')
    parser.add_argument('--sample_size', type=int, default=25)
    parser.add_argument('--verbosity', type=int, default=0)
    # parser.add_argument('--pics', default='pics/analyse_this/')
    args = parser.parse_args()

    start = time.time()

    # # read from analysis/res2/thres_rethres/
    # infile = [f for f in os.listdir(args.indir) if f.endswith('.tsv') and f.startswith(f'{args.model}_temp{args.temp}_{args.setup}')][0]
    # file_path = os.path.join(args.indir, infile)
    # print(file_path)
    #
    # df = pd.read_csv(file_path, sep='\t')
    # print(df.head())
    # print(df.columns.tolist())
    #
    # print(" ".join(df.tgt_raw.tolist()).count('“'))
    # print(" ".join(df.rewritten.tolist()).count('"'))
    # print(" ".join(df.src_raw.tolist()).count('“'))
    # exit()

    # each setup is ONE file now, not chunks!
    # indir = f'{args.indir}{args.model}/temp{args.temp}/'
    # fh = [f for f in os.listdir(indir) f.endswith('.tsv.gz')]
    fh = [f for f in os.listdir(args.indir) if
          f.startswith(f'{args.model}_temp{args.temp}_{args.setup}') and f.endswith('.tsv.gz')]
    for tlang in ['de', 'en']:
        # 'prompt/_megaouts/output/'
        # file_name = [f for f in fh if f.startswith(tlang) and args.setup in f][0]
        # file_path = os.path.join(indir, file_name)
        file_name = fh[0]
        file_path = os.path.join(args.indir, file_name)
        print(file_name)
        df = pd.read_csv(file_path, sep='\t', compression='gzip')
        print(df.head())
        print(df.columns.tolist())
        # slant1 = " ".join(df.tgt_raw.tolist()).count('“')
        # straight = " ".join(df.tgt_raw.tolist()).count('"')
        # slant2 = " ".join(df.tgt_raw.tolist()).count('„')
        # print(slant1+straight+slant2)
        # # Replace NaN values with an empty string
        # df['rewritten'] = df['rewritten'].fillna('')
        # print(" ".join(df.rewritten.tolist()).count('"'))
        # print(" ".join(df.src_raw.tolist()).count('“'))
        # exit()

        # pre-processing from parse_tabulate_rewritten.py
        # Define the regex pattern
        # pattern = re.compile(r'\"?(?:Here is a revised)?.*: ')
        # match_counter = 0
        # # Apply the editing operation to cells in Column A and count matches
        # df['rewritten'] = df['rewritten'].apply(lambda x: process_cell(x).strip('"'))

        lang_df = df[df['lang'] == tlang]

        if tlang == 'de':
            print('German')
        else:
            print('English')

        print(f"Total N of segments: {lang_df.shape[0]}")
        # filter out rows which have "copy over" or "bypassed" in prompt column
        filtered_df = lang_df[~lang_df['prompt'].str.contains('copy over the translation intact')]
        # print(filtered_df.shape)
        filtered_df = filtered_df[~filtered_df['prompt'].str.contains('bypassed')]
        print(f"After filtering out short and non-deviating segments: {filtered_df.shape[0]}")

        print(f'Ratio of segments re-written {tlang.upper()}: {filtered_df.shape[0] / lang_df.shape[0] * 100:.2f}%')

        filtered_df['thresholds'] = filtered_df['thresholds'].apply(literal_eval)
        my_thres_lengths = [len(i) for i in filtered_df['thresholds'].tolist()]
        my_fours = [1 if len(i) == np.max(my_thres_lengths) else 0 for i in filtered_df['thresholds'].tolist()]
        my_ones = [1 if len(i) == 1 else 0 for i in filtered_df['thresholds'].tolist()]
        my_ones_which = [i for i in filtered_df['thresholds'].tolist() if len(i) == 1]
        listed_single_feats = [i[0][0] for i in my_ones_which]
        singles_dict = Counter(listed_single_feats)
        sorted_singles = OrderedDict(sorted(singles_dict.items(), key=itemgetter(1), reverse=True))

        print(
            f'Average number of instructions added to the prompt ({tlang.upper()}): {np.mean(my_thres_lengths):.1f}(+/-{np.std(my_thres_lengths):.1f})')
        print(
            f'Min/max number of instructions added to the prompt ({tlang.upper()}): {np.min(my_thres_lengths)},  {np.max(my_thres_lengths)}')
        print(
            f'Prompts with max N of instructions (here: {np.max(my_thres_lengths)}): {sum(my_fours) / len(my_thres_lengths) * 100:.2f}% '
            f'({sum(my_fours)} out of {len(my_thres_lengths)})')
        print(f'Prompts with ONE instruction: {sum(my_ones) / len(my_thres_lengths) * 100:.2f}% '
              f'({sum(my_ones)} out of {len(my_thres_lengths)})')
        print('This is the distribution of the instuctions in the one-instruction prompts:')
        for k, v in sorted_singles.items():
            print(f'\t{k}: {v}')
        print()
        sample_ids = {'de': ['ORG_WR_EN_DE_000794-6', 'ORG_WR_EN_DE_002851-14', 'ORG_WR_EN_DE_001578-11',
                             'ORG_WR_EN_DE_000176-14', 'ORG_WR_EN_DE_000391-7', 'ORG_WR_EN_DE_004808-1',
                             'ORG_WR_EN_DE_004086-5', 'ORG_WR_EN_DE_011577-8', 'ORG_WR_EN_DE_001413-7',
                             'ORG_WR_EN_DE_004844-5', 'ORG_WR_EN_DE_000498-22', 'ORG_WR_EN_DE_001825-12',
                             'ORG_WR_EN_DE_005762-7', 'ORG_WR_EN_DE_000335-4', 'ORG_WR_EN_DE_001881-3',
                             'ORG_WR_EN_DE_010420-12', 'ORG_WR_EN_DE_012929-14', 'ORG_WR_EN_DE_006730-13',
                             'ORG_WR_EN_DE_005025-15', 'ORG_WR_EN_DE_003572-9', 'ORG_WR_EN_DE_005520-7',
                             'ORG_WR_EN_DE_005586-19', 'ORG_WR_EN_DE_005255-27', 'ORG_WR_EN_DE_009661-2',
                             'ORG_WR_EN_DE_008285-2'],
                      "en": ['ORG_WR_DE_EN_015614-18', 'ORG_WR_DE_EN_008444-3', 'ORG_WR_DE_EN_003729-18',
                             'ORG_WR_DE_EN_008791-12', 'ORG_WR_DE_EN_001493-14', 'ORG_WR_DE_EN_003230-9',
                             'ORG_WR_DE_EN_014493-10', 'ORG_WR_DE_EN_000268-22', 'ORG_WR_DE_EN_012032-8',
                             'ORG_WR_DE_EN_004321-21', 'ORG_WR_DE_EN_011195-2', 'ORG_WR_DE_EN_008548-15',
                             'ORG_WR_DE_EN_015614-17', 'ORG_WR_DE_EN_000753-1', 'ORG_WR_DE_EN_013910-17',
                             'ORG_WR_DE_EN_005384-16', 'ORG_WR_DE_EN_014610-16', 'ORG_WR_DE_EN_000268-15',
                             'ORG_WR_DE_EN_014493-7', 'ORG_WR_DE_EN_012937-9', 'ORG_WR_DE_EN_012243-22',
                             'ORG_WR_DE_EN_005289-10', 'ORG_WR_DE_EN_012813-8', 'ORG_WR_DE_EN_001493-13',
                             'ORG_WR_DE_EN_016430-5']}
        sample_lang_df = filtered_df[filtered_df['seg_id'].isin(sample_ids[tlang])]
        # sample_lang_df = filtered_df.sample(n=args.sample_size, random_state=args.seed)
        id_text_dict = dict(zip(sample_lang_df['seg_id'], zip(sample_lang_df['src_raw'], sample_lang_df['tgt_raw'],
                                                              sample_lang_df['prompt'],
                                                              sample_lang_df['thresholds'], sample_lang_df['rewritten'],
                                                              sample_lang_df['rethres'])))

        # generate a artificial text detector:

        # random_sequence = [random.randint(0, 1) for _ in range(25)]
        # print(random_sequence)
        # my_items = []
        # # 0 is tgt, 1 is rewrit
        # for my_idx, (seg_id, (src_seg, tgt_seg, _, _, rewrit, _)) in enumerate(id_text_dict.items()):
        #     my_guess = defaultdict(list)
        #     my_guess['type'].append('source')
        #     my_guess['type'].append('target 1')
        #     my_guess['type'].append('target 2')
        #     my_guess['tuv'].append(src_seg)
        #     my_guess['true_answer'].append('--')
        #
        #     if random_sequence[my_idx] == 0:
        #         my_guess['tuv'].append(tgt_seg)
        #         my_guess['tuv'].append(rewrit)
        #         my_guess['true_answer'].append(0)
        #         my_guess['true_answer'].append(1)
        #     else:
        #         my_guess['tuv'].append(rewrit)
        #         my_guess['tuv'].append(tgt_seg)
        #         my_guess['true_answer'].append(1)
        #         my_guess['true_answer'].append(0)
        #
        #     my_guess['your_guess (0=translated, 1=edited)'].append('--')
        #     my_guess['your_guess (0=translated, 1=edited)'].append('')
        #     my_guess['your_guess (0=translated, 1=edited)'].append('')
        #
        #     this_item = pd.DataFrame(my_guess)
        #     print(this_item)
        #     this_item.insert(4, 'seg_id', seg_id)
        #
        #     my_items.append(this_item)
        #
        # guessing_input = pd.concat(my_items, axis=0)
        # guessing_input.to_csv(f'{tlang}_guessing_game_{len(my_items)}items.tsv', sep='\t', index=False)

        # evaluating accuracy, fluency, is ME simpler than HT? sanity of instructions, does the model conform to instruction, flag extreme effects (good and bad)
        output_dict = defaultdict(list)
        for seg_id, (src_seg, tgt_seg, my_prompt, thres, rewrit, rethres) in id_text_dict.items():
            output_dict['seg_id'].append(seg_id)
            output_dict['instructions'].append(my_prompt.split('target language norm: ')[-1].split('Do not add')[0])
            if args.who == 'heike':
                # (1) are re-writings accurate and grammatical, are they simplified?
                output_dict['accurate? 1(no)-6(yes)'].append(None)
                output_dict['fluent? 1(no)-6(yes)'].append(None)
                output_dict['ME structurally more similar to SRC than HT? (y/n/same)'].append(None)
                if args.setup in ['seg_feature-based_min', 'seg_feature-based_detailed']:
                    # (2) evaluate the instructions and model's compliance with them
                    # are instructions adequate from translationese reduction perspective?
                    output_dict['sanity of instruction (y/n)'].append(None)
                    # does the model's output comply with the instructions?
                    output_dict['compliant? (y/n)'].append(None)

            else:
                print('\n*********\n')
                print(src_seg)
                print()
                print(tgt_seg)
                print()
                print(my_prompt.split('target language norm: ')[-1].split('Do not add')[0])
                print(thres)
                print()
                print(rethres)
                print()
                print(rewrit)

                # (1) are re-writings accurate and grammatical, are they simplified?
                accurate = input('Is the re-writing accurate/faithful to the source? (1-6) ')
                output_dict['accurate? 1(no)-6(yes)'].append(accurate)
                grammar = input('Is the re-writing acceptable in terms of fluency? (1-6) ')
                output_dict['fluent? 1(no)-6(yes)'].append(grammar)
                simpler = input(
                    'Is the re-writing structurally more similar to SRC than human translation? (y/n/same) ')
                output_dict['ME simpler than HT? (y/n)'].append(simpler)

                # (2) evaluate the instructions and model's compliance with them
                sanity = input('Are the instructions adequate from translationese reduction perspective? (y/n) ')
                output_dict['sanity of instruction (y/n)'].append(sanity)
                comply = input('Does the models output comply with the instructions? (y/n) ')
                output_dict['compliant? (y/n)'].append(comply)

        out_df = pd.DataFrame(output_dict)

        out_df = out_df.set_index('seg_id')
        sample_lang_df = sample_lang_df.set_index('seg_id')

        analysed_data = pd.concat([sample_lang_df, out_df], axis=1)
        # print(analysed_data.head())
        analysed_data = analysed_data.drop(['prompt', 'lang', 'index'], axis=1)
        analysed_data = analysed_data.reset_index()

        if not args.setup in ['seg_feature-based_min', 'seg_feature-based_detailed']:
            analysed_data = analysed_data.drop(['thresholds', 'rethres'], axis=1)

        if args.who == 'heike':
            analysed_data.to_csv(f'humeval/{tlang}_4heike_n{args.sample_size}_{args.model}_{args.setup}.tsv', sep='\t',
                                 index=False)
        else:
            analysed_data.to_csv(f'humeval/{tlang}_interactive_n{args.sample_size}_{args.model}_{args.setup}.tsv',
                                 sep='\t', index=False)

        input('Switching to another direction...')
