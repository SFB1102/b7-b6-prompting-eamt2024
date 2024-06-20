"""
UPD 12 Mar
replace the input to parser with the manually curated multiparallel document!

UPD 22 Nov 2023 (08 Oct 2023)
the script expects final mega tables with a complete re-written column extract/tabled/rewritten/

the script outputs tables for manual analysis (with interactive_analysis.py) and feature-tables for classifier2.py
+ general stats for the entire re-written corpus to prompting/chuncked_output/gpt-3.5-turbo/<temp0.*>/<lang>_<setup>/*.new_chunk_1.tsv

* parse segments on the fly, one-by-one, without saving conllu
* calculate all 60 features from the conllu-in-memory and write to a table to be passed to the classifier

# requires a separate run for temperature-setup combination (can be iterated from a shell script)
python3 extract/parse_tabulate_rewritten.py --temp 0.7 --setup lazy
python3 extract/parse_tabulate_rewritten.py --temp 0.7 --setup min_triad_vratio2

# 20 June 2024
python3 5_parse_extract2/multiparallel_parse_tabulate_rewritten.py --setup self-guided_detailed --thres_type ratio2.5

"""

import os
import re
import sys
import time
import ast

import pandas
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import pandas as pd
import stanza
import torch
from stanza.pipeline.core import DownloadMethod
import argparse

from feats import readerdiff, speakdiff, get_tree, content_ttr_and_density, personal_pron, possdet, cconj, sconj
from feats import word_length, nn, modpred, count_dms, polarity, sents_complexity, demdeterm, nouns_to_all_args
from feats import ud_freqs, pasttense, finites, preps, infinitives, nominals, advquantif, relativ, selfsich
from feats import verb_n_obj, main_verb_nsubj_noun, obl_obj, advmod_verb, topical_vorfeld, expected_nachfeld
from feats import mean_sent_length, advmod_no_negation


# def parse_segment(preprocessed_string, my_lang, device=None):  # pass cuda if available
#     # Initialize Stanza pipeline
#     if device == 'cuda':
#         my_parser = stanza.Pipeline(my_lang, processors='tokenize,mwt,pos,lemma,depparse', verbose=False,
#                                     tokenize_pretokenized=False,
#                                     download_method=DownloadMethod.REUSE_RESOURCES,
#                                     use_gpu=True)
#     else:
#         my_parser = stanza.Pipeline(my_lang, processors='tokenize,mwt,pos,lemma,depparse', verbose=False,
#                                     tokenize_pretokenized=False,
#                                     download_method=DownloadMethod.REUSE_RESOURCES,
#                                     use_gpu=False)

def parse_segment(preprocessed_string, my_parser=None):
    parsed_seg = my_parser(preprocessed_string)
    list_of_tokens = []
    # loop thru sents in a segment
    for sent in parsed_seg.sentences:
        word_list = sent.words
        for i, word in enumerate(word_list):
            # Write each token as a tab-separated line to the file
            if not isinstance(word.id, int):  # skip lines with agglutinated items starting with a list id [10, 11]
                continue
            else:
                try:
                    if '_' in word.text:
                        my_word = word.text.replace('_', '-')
                        my_lemma = word.lemma.replace('_', '-')
                        my_upos = word.upos
                    elif word.text == 'NONE':
                        my_word = word.text
                        my_lemma = 'NONE'
                        my_upos = "EMPTY"
                    else:
                        my_word = word.text
                        my_lemma = word.lemma
                        my_upos = word.upos

                    list_of_tokens.append("\t".join([str(word.id),
                                                     my_word,
                                                     f'{my_lemma if my_lemma else word.text}',
                                                     my_upos,
                                                     f'{word.feats if word.feats else "_"}',
                                                     str(word.head),
                                                     f'{word.deprel if word.deprel else "_"}']))
                except TypeError:
                    # this should not happen
                    if word.lemma == 'None':
                        list_of_tokens.append("\t".join([str(word.id), word.text, word.lemma, word.upos,
                                                         f'{word.feats if word.feats else "_"}',
                                                         str(word.head),
                                                         f'{word.deprel if word.deprel else "_"}']) + "\n")
                    else:
                        continue

    return list_of_tokens


def split_list_by_last_index(lst, last_indices):
    sublists = []
    start_index = 0

    for last_index in last_indices:
        sublists.append(lst[start_index:last_index + 1])
        start_index = last_index + 1

    # Add the remaining elements if any
    if start_index < len(lst):
        if lst[start_index:]:
            sublists.append(lst[start_index:])

    return sublists


def get_listed_sents_for_segment(seg_in=None):
    w_ids = [w[0] for w in seg_in]
    starts = w_ids.count(1)
    split_list_ids = [i - 1 for i, w in enumerate(seg_in) if w[0] == 1 and starts > 1]
    if split_list_ids:
        # print(split_list_ids[1:])
        # Split the list based on the last indices
        listed_sents_out = split_list_by_last_index(seg_in, split_list_ids[1:])
    else:
        listed_sents_out = [seg_in]

    return listed_sents_out


# expects a list of trees; no filtering for short sentences
def hits_counter(segment=None, collector=None, my_lang=None, list_link=None):
    mhd, bad_nodes = speakdiff(segment)  # , bad_graphs
    mdd = readerdiff(segment)
    ttr, dens, zlen = content_ttr_and_density(segment)

    wdlen = word_length(segment)
    numcls, simple = sents_complexity(segment)

    # add count-based functions
    ppron = personal_pron(segment, my_lang)
    poss = possdet(segment, my_lang)
    demdets = demdeterm(segment, my_lang)

    cc = cconj(segment, my_lang)
    sc = sconj(segment, my_lang)

    addit = count_dms(segment, my_lang, list_link=f'{list_link}dms/', dm_type='additive')
    advers = count_dms(segment, my_lang, list_link=f'{list_link}dms/', dm_type='adversative')
    caus = count_dms(segment, my_lang, list_link=f'{list_link}dms/', dm_type='causal')
    tempseq = count_dms(segment, my_lang, list_link=f'{list_link}dms/', dm_type='temp_sequen')
    epist = count_dms(segment, my_lang, list_link=f'{list_link}dms/', dm_type='epistemic')
    mpred = modpred(segment, my_lang, list_link=list_link)

    negs = polarity(segment, my_lang)

    nns = nn(segment, my_lang)
    nnargs = nouns_to_all_args(segment)
    pastv = pasttense(segment)
    fin = finites(segment)
    prep = preps(segment, my_lang)
    inf = infinitives(segment, my_lang, list_link=list_link)
    deverb = nominals(segment, my_lang, list_link=list_link)
    advqua = advquantif(segment, my_lang, list_link=list_link)
    # relcl = relativ(segment, my_lang)
    self = selfsich(segment, my_lang)
    verb_obj_n_order = verb_n_obj(segment)
    verb_nsubj_n_order = main_verb_nsubj_noun(segment)  # inversion in main clause
    obl_obj_order = obl_obj(segment)
    advmod_verb_order = advmod_verb(segment)
    vorfeld = topical_vorfeld(segment)
    nachfeld = expected_nachfeld(segment, my_lang)
    sentlength_lst, mean_sent_wc = mean_sent_length(segment)

    seg_wc = sum(sentlength_lst)

    advmod = advmod_no_negation(segment)

    collector['wc_lemma'].append(seg_wc)
    collector['advmod'].append(advmod)

    # German does not have acl:relcl???? 'discourse' has range=0 in DE, but is useful in EN

    # deleted case, relcl
    udrels = ['acl', 'advcl', 'amod', 'appos', 'aux', 'aux:pass', 'conj', 'ccomp',
              'compound', 'cop', 'discourse', 'fixed', 'flat', 'iobj', 'mark', 'nmod', 'nsubj', 'nummod', 'obj',
              'obl', 'parataxis', 'xcomp']
    # counts of each rel in a segment to be averaged to the number of segments
    dep_prob_dict = ud_freqs(segment, relations=udrels)
    for k, val in dep_prob_dict.items():
        collector[k].append(val)

    # these are per-sentence metrics, they need to be averages across segments in a doc
    collector['ttr'].append(ttr)
    collector['dens'].append(dens)
    collector['mdd'].append(mdd)
    collector['mhd'].append(mhd)
    collector['wdlen'].append(wdlen)

    # counts for syntactic phenomena - averaged across sentences in each segment
    # I am not sure I want to normalise them to the wc?
    collector['numcls'].append(numcls)
    collector['simple'].append(simple)
    # collector['relcl'].append(relcl)
    collector['nnargs'].append(nnargs)
    collector['fin'].append(fin)
    collector['mpred'].append(mpred)
    collector['vo_noun'].append(verb_obj_n_order)
    collector['vs_noun'].append(verb_nsubj_n_order)
    collector['obl_obj'].append(obl_obj_order)
    collector['advmod_verb'].append(advmod_verb_order)
    collector['vorfeld'].append(vorfeld)
    collector['nachfeld'].append(nachfeld)
    collector['mean_sent_wc'].append(mean_sent_wc)

    # these are counts of items in the entire segment, regardless the number of sentences in it
    # these raw counts are to be normalised to the segment wc
    collector['ppron'].append(ppron)
    collector['poss'].append(poss)
    collector['demdets'].append(demdets)

    collector['cconj'].append(cc)
    collector['sconj'].append(sc)

    collector['addit'].append(addit)
    collector['advers'].append(advers)
    collector['caus'].append(caus)
    collector['tempseq'].append(tempseq)
    collector['epist'].append(epist)

    collector['negs'].append(negs)

    collector['nn'].append(nns)
    collector['prep'].append(prep)
    collector['inf'].append(inf)
    collector['deverb'].append(deverb)
    collector['advqua'].append(advqua)
    collector['self'].append(self)
    collector['pastv'].append(pastv)

    return collector, bad_nodes, zlen, sentlength_lst


def update_stats(stats=None, my_stats=None):
    for k, v in my_stats.items():
        stats[k].append(v)

    return stats


def read_thres_info(my_thres):
    my_lst = []
    for tup in my_thres:
        my_lst.append(tup[0])
    return my_lst


def get_re_thres_tuples(feats=None, vals_from=None, norm_these=None, norm_by=None):
    tups_listed = []
    for feat_tup in feats:
        try:
            # print(feat_tup)
            # print(feat_tup[0], vals_from[feat_tup[0]][-1])
            feat_val = vals_from[feat_tup[0]][-1]
        except IndexError:
            print(feat_tup[0])

            feat_val = None
            exit()
            # input()
        if feat_tup[0] in norm_these:
            feat_val = round(feat_val / norm_by, 3)
        else:
            feat_val = round(feat_val, 3)
        tups_listed.append((feat_tup[0], feat_tup[1], feat_tup[2], feat_tup[3], feat_val))

    return tups_listed


# def remove_dots(segs=None):
#     _segs = [seg.replace('Euh ', '').replace('Hum ', '') for seg in segs]
#     _segs = [re.sub(r" hm\.", r".", line) for line in _segs]
#     _segs = [re.sub(r" hum\.", r".", line) for line in _segs]
#     _segs = [re.sub(r" euh\.", r".", line) for line in _segs]
#     _segs = [re.sub(r"\. ([a-zäöü])", r" \1", line) for line in _segs]  # this is the only one applicable to europ, right?
#
#     return _segs


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


def estimate_flops(cuda_device):
    props = torch.cuda.get_device_properties(cuda_device)
    cores = props.multi_processor_count * props.multi_processor_count
    # clock_rate = props.clock_rate / 1000  # KHz to MHz
    flops_per_cycle = 2  # Assuming 2 FLOPs per cycle for NVIDIA GPUs
    return cores * flops_per_cycle


def get_device_info():
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("No GPUs available.")
    else:
        print(f"Number of GPUs available: {num_gpus}")
        print("GPU Information:")

        for i in range(int(num_gpus)):
            print(i)
            device_name = torch.cuda.get_device_name(i)
            # gpu_capability = torch.cuda.get_device_capability(device_id)
            # flops = torch.cuda.get_device_properties(device_id).total_flops
            # print(f"GPU {i + 1}: {gpu_name}, Capability: {gpu_capability}, FLOPs: {flops:.2e}")
            flops = estimate_flops(i)
            print(f"Estimated FLOPS on GPU {i} ({device_name}): {flops:.2f} GFLOPS")  # Print estimated FLOPS


# Function to apply the regex substitution
def process_cell(cell_value):
    global pattern, match_counter
    match = pattern.search(cell_value)
    if match:
        match_counter += 1
        print(match)
        print(cell_value)
        out = pattern.sub('', cell_value)
        print(out)

        return pattern.sub('', cell_value)
    else:
        return cell_value


def clean_on_curated(main=None, my_filter=None, my_setup=None):
    replacement_df = my_filter[['seg_id', my_setup]]
    replacement_df = replacement_df.set_index('seg_id', drop=True)

    main = main.set_index('seg_id', drop=True)

    filtered_df = pd.concat([main, replacement_df], axis=1)
    filtered_df = filtered_df.dropna()
    filtered_df = filtered_df.drop(['rewritten'], axis=1)
    filtered_df = filtered_df.rename(columns={my_setup: 'rewritten'})
    filtered_df = filtered_df.reset_index()

    return filtered_df


def cleaner(_df=None, my_setup=None):
    temp_df = _df[_df[my_setup].apply(lambda x: len(x.strip().split('\n')) > 1)]
    if temp_df.empty:
        out_df = _df.copy()
        return out_df
    else:
        print(temp_df.head())
        print(temp_df.shape)
        exit()


def make_dirs(outdir, logsto, statsto, sub):
    os.makedirs(f"{outdir}", exist_ok=True)
    os.makedirs(f'{logsto}', exist_ok=True)
    os.makedirs(f'{statsto}', exist_ok=True)
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{sub}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # seg_id index lang src_raw tgt_raw rewritten prompt  thresholds (e.g. [('iobj', 1.0, 0.135, 'remove'), ('nnargs', 1.0, 0.363, 'remove')])
    parser.add_argument('--megaout', help='this is the GPT outputs for each setup with thresholds',
                        default='4_prompting/output/gpt-4/')  # de_seg_feature-based_detailed_ratio2.5.tsv.gz
    # manually curated multiparallel table
    parser.add_argument('--multiparallel', help='', default='data/rewritten/curated/')  # ratio2.5_de_7aligned_2056segs.tsv
    parser.add_argument('--setup', choices=['translated_min', 'self-guided_min', 'self-guided_detailed',
                                            'feature-based_min', 'feature-based_detailed'],
                        help='for path re-construction', required=True)
    parser.add_argument('--thres_type', choices=['ratio2.5', 'std2'], default='ratio2.5', required=True)
    parser.add_argument('--outdir', default='data/rewritten/feats_tabled2/')
    parser.add_argument('--stats', default='data/stats/rewritten/')
    parser.add_argument('--logsto', default='logs/rewritten/')
    parser.add_argument('--verbosity', type=int, default=0)

    args = parser.parse_args()

    start = time.time()

    make_dirs(args.outdir, args.logsto, args.stats, sub=args.thres_type)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    my_collector = defaultdict(list)
    describe_rewritten = defaultdict(list)
    count0 = 0
    count2 = 0

    # list of count-based features
    hits_feats = ['ppron', 'poss', 'demdets', 'cconj', 'sconj', 'addit', 'advers', 'caus', 'tempseq',
                  'epist', 'negs', 'nn', 'prep', 'inf', 'deverb', 'advqua', 'self']  # 'pastv', 'fin',
    stack_langs = []
    print(device)

    if args.setup.startswith('feature-based_'):
        run_setup = f'{args.setup}_{args.thres_type}'
    else:
        run_setup = args.setup

    for lang in ['de', 'en']:  # target langs
        print(f'*** Running: {lang.upper()}: {run_setup.upper()} ***')
        # we don't need to re-extract features for non-translated class, i.e. for sources!
        if device == 'cpu':
            my_parser = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse', verbose=False,
                                        tokenize_pretokenized=False,
                                        download_method=DownloadMethod.REUSE_RESOURCES,
                                        use_gpu=False, device=device)
        else:
            my_parser = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse', verbose=False,
                                        tokenize_pretokenized=False,
                                        download_method=DownloadMethod.REUSE_RESOURCES,
                                        use_gpu=True, device=device)
        # ratio2.5_de_7aligned_2056segs.tsv
        fh = [f for f in os.listdir(args.multiparallel) if f.startswith(f'{args.thres_type}_{lang}_7aligned_')][0]
        df0 = pd.read_csv(f"{args.multiparallel}/{fh}", sep='\t')
        print(df0.shape)
        print(df0.columns.to_list())

        fh = [f for f in os.listdir(args.megaout) if f.startswith(f'{lang}_seg_{run_setup}.tsv.gz')][0]
        df = pd.read_csv(f"{args.megaout}{fh}", sep='\t', compression='gzip')
        print(f'Raw: {df.shape}')
        filtered = clean_on_curated(main=df, my_filter=df0, my_setup=args.setup)
        print(f'Filtered by IDs after manual curation: {filtered.shape}')

        # make sure thres are interpreted as lists, not as strings
        filtered['thresholds'] = filtered['thresholds'].apply(ast.literal_eval)

        re_thres = []
        segs_count = 0
        sents_count = 0
        wc_count = 0

        lang_df = filtered[filtered.lang == lang]
        # drop nan in 'rewritten' column. I should not have any!
        filtered_rows = lang_df[lang_df['rewritten'].isna()]
        if filtered_rows.shape[0] >= 1:
            print(f'Nan in rewritten (filled with tgt_raw): {filtered_rows.shape[0]}')
            lang_df['rewritten'].fillna(lang_df['tgt_raw'], inplace=True)
        filtered_rows2 = lang_df[lang_df['rewritten'].isna()]
        print(f'Nan in rewritten (should be 0): {filtered_rows2.shape[0]}')
        lang_df['rewritten'] = lang_df['rewritten'].apply(lambda x: x.strip('"'))

        print(lang_df.shape)

        # calculate 60 feats for rewritten, save as a tsv
        id_text_dict = dict(zip(lang_df['seg_id'], zip(lang_df['rewritten'], lang_df['thresholds'])))
        print(len(id_text_dict.items()))

        # I might want to get the actual SVM probs for each segment instead of a 95% cutoff
        for seg_id, (seg, thres) in tqdm(id_text_dict.items(), position=0, leave=True):
            if not isinstance(seg, str):
                print(seg, type(seg))
                print('Nan should not happen anymore!')
                exit()
            else:
                # preprocess document following the pre-parsing procedures in /home/maria/main/proj/b7/add_meta_run_stanza.py
                # this is the only command from remove_dots function that applies to europ
                # seg = re.sub(r"\. ([a-zäöü])", r" \1", seg)
                seg = seg.strip().strip('"')

                # segment can have more than one sentence
                lines = parse_segment(seg, my_parser=my_parser)
                # I need a list of [int(identifier), token, lemma, upos, feats, int(head), rel] for feature extraction
                in_segment = get_tree(lines)
                segment_trees = get_listed_sents_for_segment(seg_in=in_segment)

                # get seg_id and lang
                my_collector['seg_id'].append(seg_id)
                my_collector['lang'].append(lang)
                my_collector['rewritten'].append(seg)
                # get only feature values from annotated segment (here: list of trees), reusing tabulate_input pipeline
                list_dir = '5_parse_extract2/searchlists/'
                my_collector, tot_bad_nodes, zero_len, listed_sent_lengths = hits_counter(segment=segment_trees,
                                                                                          collector=my_collector,
                                                                                          my_lang=lang,
                                                                                          list_link=list_dir)
                segs_count += 1
                sents_count += len(listed_sent_lengths)
                wc_count += sum(listed_sent_lengths)

                count0 += tot_bad_nodes
                count2 += zero_len
                if thres:
                    # if I want to normalise hits to seg_wc for rethres, I need to separately do (hits / seg_wc) here
                    # for the entire table it is done below!
                    if 'mean_sent_length' in [tup[0] for tup in thres]:
                        # print('*** Filtering out inconsistency in feature names again, fixed for the future! ***')
                        # re-writing tuples
                        thres = [(tup[0].replace('mean_sent_length', 'mean_sent_wc'), tup[1], tup[2], tup[3]) for tup in
                                 thres]

                    new_thres = get_re_thres_tuples(feats=thres, vals_from=my_collector,
                                                    norm_these=hits_feats, norm_by=wc_count)
                    my_collector['rethres'].append(new_thres)
                else:
                    my_collector['rethres'].append([])

        describe_rewritten['lang'].append(lang)
        describe_rewritten['segs'].append(segs_count)
        describe_rewritten['sents'].append(sents_count)
        describe_rewritten['wc'].append(wc_count)

        stack_langs.append(df)  # needed for analysis output with both thres and rethres

    print(f'\nNode errors: {count0},\nZero-length sents: {count2}')

    stats_table = pd.DataFrame(describe_rewritten)
    stats_name = f'{args.stats}{args.thres_type}_gpt4_rewritten_stats_temp0.7_{run_setup}.tsv'
    stats_table.to_csv(stats_name, sep='\t', index=False)
    print(f'1. Pars of the rewritten corpus saved to {stats_name}\n')

    tabled_rewritten = pd.DataFrame(my_collector)
    # i normalise to wc inside the fuctions to get result for individual segments before the entire table is available
    tabled_rewritten[hits_feats] = tabled_rewritten[hits_feats].div(tabled_rewritten['wc_lemma'], axis=0)

    analysis_dir = 'data/rewritten/thres_rethres/'
    os.makedirs(analysis_dir, exist_ok=True)
    mega_data_table_analysis = f'{analysis_dir}{args.thres_type}_gpt4_temp0.7_{run_setup}_seg_src_tgt_rewritten_thres_rethres.tsv.gz'

    # adding a rethres column to the main manual analysis table using seg_id as index
    my_seg_indexed_rethres_tuples = tabled_rewritten[['seg_id', 'rethres']]
    my_seg_indexed_rethres_tuples = my_seg_indexed_rethres_tuples.set_index('seg_id')

    two_langs_df = pd.concat(stack_langs, axis=0)
    two_langs_df = two_langs_df.set_index('seg_id')
    added_rethres_res_df = pd.concat([two_langs_df, my_seg_indexed_rethres_tuples], axis=1)
    added_rethres_res_df = added_rethres_res_df.reset_index()

    added_rethres_res_df.to_csv(mega_data_table_analysis, sep='\t', index=False)
    print(added_rethres_res_df.head())
    print(f'2. A table (de and en) is saved to {analysis_dir}{mega_data_table_analysis}.')
    print(f"Columns: {added_rethres_res_df.columns.tolist()}")

    # FEATURE table per se: for consistency to avoid potential classifier confusion
    tabled_rewritten = tabled_rewritten.drop(['rethres'], axis=1)
    print(tabled_rewritten.head())
    print(tabled_rewritten.shape)

    # add a subdir for each thres_type
    my_thres_sub = f'{args.outdir}{args.thres_type}/'
    os.makedirs(my_thres_sub, exist_ok=True)
    rewritten_feats = f'{my_thres_sub}gpt4_temp0.7_{run_setup}_rewritten_feats.tsv.gz'
    tabled_rewritten.to_csv(rewritten_feats, sep='\t', compression='gzip', index=False)
    print(f'3. Tabulated features for re-written segs (classifier input) is saved to {rewritten_feats}')
    print(tabled_rewritten.columns.tolist())

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
