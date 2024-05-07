"""
10 Nov 2023
Balance and filter the corpus and tabulate feature vals (and metadata) for various doc-size cut-offs and corpus sizes

-- filter docs using doc_size_in_tokens threshold and balance across translation directions by taking random max_docs
-- produce descriptive stats of the filtered corpus
-- extract numerical data (feature values) from the resulting subsets

python3 1_parse_extract_feats/tabulate_input.py --indir data/conllu/used/ --max_docs 1500 --table_unit seg

"""

import os
import sys
import argparse
import gzip
from collections import defaultdict
import time
import random
from datetime import datetime
import re
import pandas as pd
import numpy as np
from feats import readerdiff, speakdiff, get_tree, content_ttr_and_density, personal_pron, possdet, cconj, sconj
from feats import word_length, nn, modpred, count_dms, polarity, sents_complexity, demdeterm, nouns_to_all_args
from feats import ud_freqs, pasttense, finites, preps, infinitives, nominals, advquantif, relativ, selfsich, advmod_no_negation
from feats import verb_n_obj, main_verb_nsubj_noun, obl_obj, advmod_verb, topical_vorfeld, expected_nachfeld, mean_sent_length


def get_doc_id(chunk=None):
    doc_id = None
    lines0 = chunk.split('\n')
    for line0 in lines0:
        if line0.startswith('<text'):
            bits = line0.split()
            for bit in bits:
                if bit.startswith('my_id="'):
                    doc_id = bit.split('"')[1]
    return doc_id


def lose_shorts(df=None, by=None, threshold=None):
    df_filtered = df[~(df[by] < threshold)]

    good_long = df_filtered['doc_id'].tolist()

    return good_long


def get_percentile_outliers(df=None, by=None, threshold=None, sided=None):
    names = df['doc_id'].values
    vals = df[by].values

    if sided == 'two':
        # from both sides of the continuum
        diff0 = threshold / 2.0
        min_val, max_val = np.percentile(vals, [diff0, 100 - diff0])  # calculate upper and lower boundary
        print(min_val, max_val)
    elif sided == 'lose_short':
        # only very long texts
        min_val, max_val = np.percentile(vals, [threshold, 100])  # calculate bottom percentile
        print(min_val, max_val)
    elif sided == 'lose_long':
        # only very long texts
        min_val, max_val = np.percentile(vals, [0, 100 - threshold])  # calculate upper and lower boundary
        print(min_val, max_val)
    else:
        min_val = None
        max_val = None
        print('ERROR')
        exit()

    res = (vals < min_val) | (vals > max_val)  # bools

    lose_outliers_names = []
    lose_outliers_vals = []
    for k in range(len(names)):
        if res[k]:
            lose_outliers_names.append(names[k])
            # lose_outliers_vals.append(mean_sent_lens[k])
            lose_outliers_vals.append(vals[k])
    print(f'These docs are {100 - threshold}% outliers by {by} ({sided}):')

    for idx, (n, v) in enumerate(zip(lose_outliers_names, lose_outliers_vals)):
        if idx < 10:
            print(n, v)

    return lose_outliers_names


def get_by_doc_pars(my_corp_string=None):
    sizes_by_doc = defaultdict(list)  # keys for columns: doc_id, segs, tokens
    data = my_corp_string.replace('</text>', '</text></theend>')
    chunks = data.strip().split('</theend>')[:-1]  # skip the empty chunk after the last bit
    for this_doc_chunk in chunks:
        doc_id = get_doc_id(chunk=this_doc_chunk)

        sizes_by_doc['doc_id'].append(doc_id)

        this_doc_chunk = this_doc_chunk.replace('</seg>', '</seg></segend>')
        shards = this_doc_chunk.strip().split('</segend>')[:-1]  # skip the last chunk containing only </text>
        sizes_by_doc['segs'].append(len(shards))
        doc_toks = 0
        for shard in shards:
            toks = shard.strip().split('\n')[:-1]  # skip the last chunk containing only <\seg>
            doc_toks += len(toks)
        sizes_by_doc['tokens'].append(doc_toks)

    sizes_df = pd.DataFrame.from_dict(sizes_by_doc)

    return sizes_df


def produce_balanced_ids_list(my_dir=None, size_metric=None, thres=None, seed=None, max_docs=None, my_path=None):
    my_files = [f for f in os.listdir(my_dir) if f.endswith('.gz')]
    for f in my_files:
        filepath = f'{my_dir}{f}'
        inhalt = gzip.open(filepath, 'rt').read()

        by_doc_df = get_by_doc_pars(my_corp_string=inhalt)
        # my_outliers = get_percentile_outliers(df=by_doc_df, by='tokens', threshold=args.thres, sided='lose_short')
        # print(len(my_outliers))
        # print(list(my_outliers)[:10])
        long_enough = lose_shorts(df=by_doc_df, by=size_metric, threshold=thres)
        print(f'{f} docs: {len(long_enough)}')
        my_keepthem_filter[f] = long_enough

    # from each list, take 5000 random docs
    # making sure that we sample aligned documents to increase domain comparability of translation directions
    # this results in some documents being shorter than the established doc length threshold
    balanced_src_fns = []
    random.seed(seed)
    for k, v in my_keepthem_filter.items():
        if k.startswith('ORG'):
            random_src_fns = random.sample(v, max_docs)
            balanced_src_fns.extend(random_src_fns)
        else:
            continue
    balanced_tgt_fns = [i.replace('ORG_WR_DE_EN', 'TR_DE_EN').replace('ORG_WR_EN_DE', 'TR_EN_DE') for i in
                        balanced_src_fns]
    balanced_fns = balanced_src_fns + balanced_tgt_fns

    with open(my_path, 'w') as fns_out:
        for fn in balanced_fns:
            fns_out.write(fn + '\n')
    print(f"\nDocs in doc-length controlled, balanced corpus: {len(balanced_fns)}")

    return balanced_fns


def update_stats(stats=None, my_stats=None):
    for k, v in my_stats.items():
        stats[k].append(v)

    return stats


def get_metadata(fn=None):
    if "ORG" in fn:
        my_type = 'source'
        if 'EN_DE' in fn:
            my_lang = 'en'
            my_langpair = 'EN-DE'
        else:
            my_lang = 'de'
            my_langpair = 'DE-EN'
    else:
        my_type = 'target'
        if 'DE_EN' in fn:
            my_lang = 'en'
            my_langpair = "DE-EN"
        else:
            my_lang = 'de'
            my_langpair = "EN-DE"

    return my_langpair, my_type, my_lang


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


def seg_num_info(seg=None):
    # <seg num="19" sent_align="19">
    num = None
    s_lines = seg.strip().split('\n')
    for idx, sline in enumerate(s_lines):
        if sline.startswith('<seg'):
            num = sline.split('"')[1]

    return num


def raw_seg_from_tag(seg=None):
    # <seg num="19" sent_align="19" seg_in="ksdjfsjdg">
    raw = None
    s_lines = seg.strip().split('\n')
    for idx, sline in enumerate(s_lines):
        if sline.startswith('<seg'):
            raw = sline.split('"')[5]
            # print(raw)
            # exit()

    return raw


def split_list_by_last_index(lst, last_indices):
    sublists = []
    start_index = 0

    for last_index in last_indices:
        sublists.append(lst[start_index:last_index + 1])
        start_index = last_index + 1

    # Add the remaining elements if any
    if start_index < len(lst):
        sublists.append(lst[start_index:])

    return sublists


def get_listed_sents_for_segment(seg_in=None):
    w_ids = [w[0] for w in seg_in]
    starts = w_ids.count(1)
    split_list_ids = [i-1 for i, w in enumerate(seg_in) if w[0] == 1 and starts > 1]
    if split_list_ids:
        # print(split_list_ids[1:])
        # Split the list based on the last indices
        listed_sents_out = split_list_by_last_index(seg_in, split_list_ids[1:])
    else:
        listed_sents_out = [seg_in]

    return listed_sents_out


# no filtering for short sentences
def hits_counter(seg=None, collector=None, lang=None, list_link=None):
    TAG_RE = re.compile(r'<[^>]+>')
    string = TAG_RE.sub('', seg)
    string = string.replace("\n\n", "\n")
    lines = string.strip().split('\n')
    in_segment = get_tree(lines)
    segment = get_listed_sents_for_segment(seg_in=in_segment)

    mhd, bad_nodes = speakdiff(segment)  # , bad_graphs
    # mhd2, pics = speakdiff_visuals(segment)
    # for pic in pics:
    #     ig.plot(pic)
    #     ig.plot(pic, target='myfile.pdf')
    mdd = readerdiff(segment)
    ttr, dens, zlen = content_ttr_and_density(segment)

    wdlen = word_length(segment)
    numcls, simple = sents_complexity(segment)

    # add count-based functions
    ppron = personal_pron(segment, lang)
    poss = possdet(segment, lang)
    demdets = demdeterm(segment, lang)

    cc = cconj(segment, lang)
    sc = sconj(segment, lang)

    addit = count_dms(segment, lang, list_link=f'{list_link}dms/', dm_type='additive')
    advers = count_dms(segment, lang, list_link=f'{list_link}dms/', dm_type='adversative')
    caus = count_dms(segment, lang, list_link=f'{list_link}dms/', dm_type='causal')
    tempseq = count_dms(segment, lang, list_link=f'{list_link}dms/', dm_type='temp_sequen')
    epist = count_dms(segment, lang, list_link=f'{list_link}dms/', dm_type='epistemic')
    mpred = modpred(segment, lang, list_link=list_link)

    negs = polarity(segment, lang)

    nns = nn(segment, lang)
    nnargs = nouns_to_all_args(segment)
    pastv = pasttense(segment)
    fin = finites(segment)
    prep = preps(segment, lang)
    inf = infinitives(segment, lang, list_link=list_link)
    deverb = nominals(segment, lang, list_link=list_link)
    advqua = advquantif(segment, lang, list_link=list_link)
    relcl = relativ(segment, lang)
    self = selfsich(segment, lang)
    verb_obj_n_order = verb_n_obj(segment)
    verb_nsubj_n_order = main_verb_nsubj_noun(segment)  # inversion in main clause
    obl_obj_order = obl_obj(segment)
    advmod_verb_order = advmod_verb(segment)
    vorfeld = topical_vorfeld(segment)
    nachfeld = expected_nachfeld(segment, lang)
    sentlength_lst, mean_sent_wc = mean_sent_length(segment)
    advmod = advmod_no_negation(segment)

    # German does not have acl:relcl???? 'discourse' has range=0 in DE
    udrels = ['acl', 'advcl', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'conj', 'ccomp',
              'compound', 'cop', 'discourse', 'fixed', 'flat', 'iobj', 'mark', 'nmod', 'nsubj', 'nummod', 'obj',
              'obl', 'parataxis', 'xcomp']
    # counts of each rel in a segment to be averaged to the number of segments
    dep_prob_dict = ud_freqs(segment, relations=udrels)
    for k, val in dep_prob_dict.items():
        collector[k].append(val)

    collector['advmod'].append(advmod)

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
    collector['relcl'].append(relcl)
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

    # collect raw texts and meta
    seg_raw = []
    seg_lemmas = []
    seg_postags = []
    s_lines = seg.strip().split('\n')
    for idx, sline in enumerate(s_lines):
        if sline.startswith('<'):
            continue
        else:
            if len(sline) > 0:
                # get the true hyphon back in compounds, esp in translated DE
                if "@" in sline.strip():
                    sline = sline.replace('@', '-')

                lemma = sline.strip().split('\t')[2]

                tok = sline.split('\t')[1]
                if "_" in lemma or '|' in lemma:
                    lemma = tok

                seg_raw.append(tok.strip())
                seg_lemmas.append(lemma.strip())

                seg_postag = sline.strip().split('\t')[3]
                seg_postags.append(seg_postag)

    assert len(seg_raw) == len(seg_lemmas), 'Num tokens != num lemmas'

    seg_raw_toks = " ".join(seg_raw)
    collector['raw_tok'].append(seg_raw_toks)
    collector['wc_lemma'].append(len(seg_lemmas))
    collector['sents'].append(len(segment))

    return collector, bad_nodes, zlen, sentlength_lst


# this function outputs TWO dfs: features and descriptive stats for the corpus
def tabulate_all(my_dir=None, in_filter=None, temp_write_to=None, list_dir=None):

    stats_collector = defaultdict(list)

    my_collector = defaultdict(list)
    count0 = 0
    count2 = 0
    subcorp_files = [f for f in os.listdir(my_dir) if f.endswith('.gz')]
    empty_segs = []
    for f in subcorp_files:
        doc_num = 0
        all_segs = 0
        sents_per_doc_lst = []
        toks_per_docs_lst = []

        filepath = f'{my_dir}{f}'
        langpair, ttype, lang = get_metadata(fn=f"{filepath.split('/')[-1]}")
        inhalt = gzip.open(filepath, 'rt').read()
        inhalt = inhalt.replace('</text>', '</text></theend>')

        docs = inhalt.strip().split('</theend>')[:-1]  # skip the empty chunk after the last bit
        for doc in docs:
            sents_in_this_doc_lst = []
            doc_id = get_doc_id(chunk=doc)
            if doc_id not in in_filter:  # skip unlisted ids
                continue
            doc_num += 1
            # process segment
            segs = doc.split('</seg>')[:-1]
            for seg in segs:
                # annotate each seg pair with doc-level parameters
                my_collector['doc_id'].append(doc_id)

                seg_num = seg_num_info(seg=seg)
                my_collector['seg_num'].append(seg_num)  # SL and TL segs are aligned, hence same number

                seg_raw = raw_seg_from_tag(seg=seg)
                my_collector['raw'].append(seg_raw)
                # keep a list, save it to a file for filtering out respective seg PAIRS
                if not seg_raw:
                    empty_segs.append(f'{doc_id}-{seg_num}')

                my_collector['ttype'].append(ttype)
                my_collector['corpus'].append('europ2018')
                my_collector['direction'].append(langpair)
                my_collector['lang'].append(lang)

                my_collector, tot_bad_nodes, zlen, listed_sentlengths = hits_counter(seg=seg, collector=my_collector,
                                                                                     lang=lang, list_link=list_dir)

                count0 += tot_bad_nodes
                count2 += zlen

                sents_in_this_doc_lst.extend(listed_sentlengths)

            all_segs += len(segs)
            toks_per_docs_lst.append(sum(sents_in_this_doc_lst))
            sents_per_doc_lst.append(len(sents_in_this_doc_lst))
        mydict = update_stats(stats=stats_collector, my_stats={'corpus': 'europ2018',  # last folder
                                                               'direction': langpair,
                                                               'lang': lang,
                                                               'ttype': ttype,
                                                               'docs': doc_num,
                                                               'segs': all_segs,
                                                               'sents': sum(sents_per_doc_lst),
                                                               'mean_sents/doc': f'{sum(sents_per_doc_lst) / len(sents_per_doc_lst):.2f} (+/-{np.std(sents_per_doc_lst):.2f})',
                                                               'min_sents/doc': np.min(sents_per_doc_lst),
                                                               'max_sents/doc': np.max(sents_per_doc_lst),
                                                               'tokens': sum(toks_per_docs_lst),
                                                               'mean_toks/doc': f'{sum(toks_per_docs_lst) / len(toks_per_docs_lst):.2f} (+/-{np.std(toks_per_docs_lst):.2f})',
                                                               'min_toks/doc': np.min(toks_per_docs_lst),
                                                               'max_toks/doc': np.max(toks_per_docs_lst)
                                                               })

    print(f'\nNode errors: {count0},\nZero-length sents: {count2}')
    # print(f'Two-word segments like Warum ? Nein ! Jawhol ! Wieso ? Nice ? : {two_items_all}')
    df = pd.DataFrame(my_collector)

    with open(f'{temp_write_to}empty_segs_europ2018.txt', 'w') as outf:
        for i in empty_segs:
            outf.write(i.strip() + '\n')

    return df, pd.DataFrame(stats_collector)


def list_to_newline_sep(lst):
    return '\n'.join(lst)


def make_dirs(outdir, out_stats, logsto):
    os.makedirs(out_stats, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logsto, exist_ok=True)
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{script_name}.log'
    sys.stdout = Logger(logfile=log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='produce a lpair balanced corpus')
    parser.add_argument("--indir", help="data/conllu/, output of add_meta_run_stanza.py", required=True)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--metric", default='tokens')
    parser.add_argument("--thres", type=int, default=450, help='Min doc size in tokens, lose shorter documents')
    parser.add_argument("--max_docs", type=int, default=1500, help='Size of the random sample')
    parser.add_argument("--table_unit", choices=['doc', 'seg'], help="are instances documents or segments?",
                        default='seg')
    parser.add_argument("--outdir", help="write a new xml with conllu inside", default='data/feats_tabled/')
    parser.add_argument("--out_stats", help="where to put raw_stats by lang", default='data/stats/feats/')
    parser.add_argument('--logsto', default=f'logs/feats/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(args.outdir, args.out_stats, args.logsto)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    my_keepthem_filter = defaultdict(list)

    balanced_path = f'{args.outdir}{args.metric}-{args.thres}-{args.max_docs}.fns'

    if os.path.exists(balanced_path):
        balanced_ids = [i.strip() for i in open(balanced_path, 'r').readlines()]
        print(f'Bilingual fns list is loaded from {balanced_path}')
    else:
        # for each sub, get a list of fns that are long enough
        # translations in each monolingual set are translations of the originals in the other monolingual set
        balanced_ids = produce_balanced_ids_list(my_dir=args.indir, size_metric=args.metric, thres=args.thres,
                                                 seed=args.seed,
                                                 max_docs=args.max_docs, my_path=balanced_path)
        print('Bilingual fns list is generated anew and written to a file')

    # collecting stats and features in one go
    outstats_fn = f'{args.out_stats}filtered_{args.table_unit}-{args.thres}-{args.max_docs}.pars.tsv'
    table_name = f'{args.outdir}{args.table_unit}-{args.thres}-{args.max_docs}.feats.tsv.gz'
    print('Filtering. Calculating conllu features, tabulating and describing filtered data ...')
    feats_df, stats_df = tabulate_all(my_dir=args.indir, in_filter=balanced_ids,
                                      temp_write_to=args.outdir,
                                      list_dir=f'{sys.argv[0].split("/")[-2]}/searchlists/')  # keep the lists next to script
    print(feats_df.head())
    print(feats_df.columns.tolist())
    print(len(feats_df.columns.tolist()))

    print(f'\nCorpus parameters (after filtering):')
    print(stats_df)
    stats_df.to_csv(outstats_fn, sep='\t', index=False)
    print(f'Filtered corpus description is written to {outstats_fn}')

    featd_df = feats_df.drop(['seg_num'], axis=1)
    meta = ['doc_id', 'ttype', 'corpus', 'direction', 'lang']  # excludes raw_tok, wc_lemma, sents
    hits_feats = ['ppron', 'poss', 'demdets', 'cconj', 'sconj', 'addit', 'advers', 'caus', 'tempseq',
                  'epist', 'negs', 'nn', 'prep', 'inf', 'deverb', 'advqua', 'self']  #
    metrics_feats = ['ttr', 'dens', 'mdd', 'mhd', 'wdlen', 'nnargs', 'numcls', 'simple', 'relcl', 'mpred',
                     'vo_order', 'vs_order', 'obl_obj', 'advmod_verb', 'vorfeld', 'nachfeld', 'mean_sent_wc',
                     'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'pastv', 'fin',
                     'conj', 'ccomp', 'compound', 'cop', 'discourse', 'fixed', 'flat', 'iobj', 'mark', 'nmod', 'nsubj',
                     'nummod', 'obj', 'obl', 'parataxis', 'xcomp']
    if args.table_unit == 'seg':
        # Normalise the values in hits columns by segment wc
        feats_df[hits_feats] = feats_df[hits_feats].div(feats_df['wc_lemma'], axis=0)

    else:  # assume doc:
        # experiments have shown that aggregation method had no influence on the classification outcomes
        # Aggregate document hits and normalise by the document wc_lemma (which excludes skipped < 3-token sents)
        # Average metrics across doc segments
        mean_functions = {col: 'mean' for col in metrics_feats}
        sum_functions = {col: 'sum' for col in hits_feats}
        first_functions = {col: 'first' for col in meta}
        feats_df = feats_df.groupby('doc_id', as_index=False).aggregate({'raw_tok': list_to_newline_sep,
                                                                         'wc_lemma': 'sum',
                                                                         'sents': 'sum',
                                                                         **mean_functions,
                                                                         **sum_functions,
                                                                         **first_functions
                                                                         })
        # Divide the selected columns by the value in the divisor column
        feats_df[hits_feats] = feats_df[hits_feats].div(feats_df['wc_lemma'], axis=0)

    print(feats_df.head())
    print(feats_df.shape)
    feats_df.to_csv(table_name, sep='\t', compression='gzip', index=False)
    print(f'Tabulated data ({args.table_unit}) is written to {table_name}')

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
