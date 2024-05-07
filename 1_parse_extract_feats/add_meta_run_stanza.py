"""
09 November 2023
parse with STANZA
- wrap sents into <s num="n"> tags, where n is the ordinal number of segments in each text

Two input tables deen_*.tsv.gz, ende_*.tsv.gz + 4 tables with meta ex. deen_epic_meta.tsv.gz, ende_europ_meta.tsv.gz
result in four files, original and translated documents for each language:
ORG_WR_DE_EN.conllu.xml.gz, TR_DE_EN.conllu.xml.gz, etc

Note that it is unlikely that it will be possible to read a large file in this pseudo-xml format with an xml-parser because of the line-breaks

USAGE:
python3 1_parse_extract_feats/add_meta_run_stanza.py --tsv_dir data/raw_aligned/ --meta_dir data/raw_aligned/meta/ --outdir data/conllu/
"""

import os
import sys
import stanza
from stanza.pipeline.core import DownloadMethod
import argparse
import gzip
import time
import torch
import numpy as np
from datetime import datetime
import pandas as pd
from collections import defaultdict


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


def define_pipelines(my_langs=None, my_device=None):
    sl = my_langs[:2]
    tl = my_langs[-2:]

    if my_device == 'cuda':
        sl_stanza_pipe = stanza.Pipeline(sl, processors='tokenize,mwt,pos,lemma,depparse', verbose=False,
                                         tokenize_pretokenized=False,
                                         download_method=DownloadMethod.REUSE_RESOURCES,
                                         use_gpu=True, device=my_device)
        tl_stanza_pipe = stanza.Pipeline(tl, processors='tokenize,mwt,pos,lemma,depparse', verbose=False,
                                         tokenize_pretokenized=False,
                                         download_method=DownloadMethod.REUSE_RESOURCES,
                                         use_gpu=True, device=my_device)
    else:
        sl_stanza_pipe = stanza.Pipeline(sl, processors='tokenize,mwt,pos,lemma,depparse', verbose=False,
                                         tokenize_pretokenized=False,
                                         download_method=DownloadMethod.REUSE_RESOURCES,
                                         use_gpu=False, device=my_device)
        tl_stanza_pipe = stanza.Pipeline(tl, processors='tokenize,mwt,pos,lemma,depparse', verbose=False,
                                         tokenize_pretokenized=False,
                                         download_method=DownloadMethod.REUSE_RESOURCES,
                                         use_gpu=False, device=my_device)

    return sl_stanza_pipe, tl_stanza_pipe


def file_naming_routine(outdir=None, slang=None, tlang=None):
    src_out = f'{outdir}ORG_WR_{slang.upper()}_{tlang.upper()}.conllu.xml.gz'
    tgt_out = f'{outdir}TR_{slang.upper()}_{tlang.upper()}.conllu.xml.gz'

    return src_out, tgt_out


def write_vrt_seg(sents=None, my_outf=None):
    # print('no parallelism in N sents in sseg and tseg! processing them separately')
    wc = 0
    # loop thru sents in a segment
    for sent in sents:
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

                    my_outf.write("\t".join([str(word.id),
                                             my_word,
                                             f'{my_lemma if my_lemma else word.text}',
                                             my_upos,
                                             f'{word.feats if word.feats else "_"}',
                                             str(word.head),
                                             f'{word.deprel if word.deprel else "_"}']) + "\n")

                    wc += 1

                except TypeError:
                    # this should not happen
                    if word.lemma == 'None':
                        my_outf.write("\t".join([str(word.id), word.text, word.lemma, word.upos,
                                                 f'{word.feats if word.feats else "_"}',
                                                 str(word.head),
                                                 f'{word.deprel if word.deprel else "_"}']) + "\n")
                    else:
                        continue
    return my_outf, wc


def add_fake_meta(smeta=None, tmeta=None, missing_fns=None, slang=None, tlang=None):
    for i in missing_fns:
        if slang == 'de':
            national = 'Germany'
        else:
            national = ''
        new_srow = {
            'text_id': i,
            'meta': f'<text id="" birth_date="" birth_place="" is_mep="True" m_state="{slang}" mode="WR_ORG" n_party="" name="" nationality="{national}" p_group="" speaker_id="" role="" my_id="{i}" subcorpus="ORG_{slang.upper()}_{tlang.upper()}" date="" day_id="" lang="{slang}" place="Brüssel" edition="">',
        }
        tr_i = i.replace('ORG_WR_', 'TR_')
        new_trow = {
            'text_id': tr_i,
            'meta': f'<text id="" birth_date="" birth_place="" is_mep="True" m_state="{slang}" mode="TR" n_party="" name="" nationality="{national}" p_group="" speaker_id="" role="" my_id="{tr_i}" subcorpus="TR_{slang.upper()}_{tlang.upper()}" date="" day_id="" lang="{tlang}" place="Brüssel" edition="">',
        }
        # Using loc[] function
        smeta.loc[len(slang_meta)] = new_srow
        tmeta.loc[len(tlang_meta)] = new_trow

    return smeta, tmeta


def etwas_neues(smeta=None, current_fns=None, lpair=None):
    # compare fns from newly-aligned data and existing metadata from previous vcorpus version
    text_ids = set(smeta['text_id'].tolist())

    mismatching_fns = current_fns.difference(text_ids)

    if mismatching_fns:
        print(f'Document pairs missing from metadata file ({lpair}): {len(mismatching_fns)} e.g. {list(mismatching_fns)[:2]}')
    else:
        print(f'No differences between the current and existing ST filenames detected. Filename format: {list(text_ids)[:2]}')

    return list(mismatching_fns)


def make_dirs(conllu, out_stats, logsto):
    os.makedirs(out_stats, exist_ok=True)
    os.makedirs(conllu, exist_ok=True)
    os.makedirs(logsto, exist_ok=True)
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logsto}{script_name}.log'
    sys.stdout = Logger(logfile=log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get a conllu with <text id= > and <seg num= >')
    parser.add_argument("--tsv_dir", help="a folder with bitext seg-aligned tables and their ids", required=True)
    parser.add_argument("--meta_dir", help="a folder with 4 text-id meta tables Europarl", required=True)
    parser.add_argument("--outdir", help="write a new xml with conllu inside", default='data/conllu/')
    parser.add_argument("--out_stats", help="where to put raw_stats by lang", default='data/stats/stanza/')
    parser.add_argument('--logsto', default=f'logs/stanza/')

    args = parser.parse_args()
    start = time.time()
    make_dirs(args.outdir, args.out_stats, args.logsto)

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]}, {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # select one tsv and corresponding meta in two files
    my_dict = defaultdict(list)
    doc_segs_dict = defaultdict(list)
    for langpair in ['deen', 'ende']:
        start = time.time()
        print(f'\nLanguage pair: {langpair}')
        fh_meta = [f for f in os.listdir(args.meta_dir) if f.endswith('.gz') and langpair in f]  # two files
        src_lang = None
        tgt_lang = None
        slang_meta = None
        tlang_meta = None
        for file in fh_meta:
            lang = file.split('_')[0]
            if file.startswith(lang) and file.split('_')[1].startswith(lang):
                slang_meta = pd.read_csv(args.meta_dir + file, sep='\t', compression='gzip')
                print(f'\tSL meta from {file}: {slang_meta.shape}')
                print(slang_meta.head())
                src_lang = lang
            else:
                tlang_meta = pd.read_csv(args.meta_dir + file, sep='\t', compression='gzip')
                print(f'\tTL meta from {file}: {tlang_meta.shape}')
                tgt_lang = lang
        print(f'My SL and TL: {src_lang, tgt_lang}')

        # create pipelines for SL and TL
        sl_stanza, tl_stanza = define_pipelines(my_langs=langpair, my_device='cpu')  # device

        bitext_file = [f for f in os.listdir(args.tsv_dir) if f.endswith('.gz') and langpair in f][0]
        df_texts = pd.read_csv(args.tsv_dir + bitext_file, sep='\t', compression='gzip')
        print(df_texts.columns.tolist())
        print(f'\nWide table with raw text is loaded: {df_texts.shape}\n')

        s_outf, t_outf = file_naming_routine(outdir=args.outdir, slang=src_lang, tlang=tgt_lang)
        # the number of files in raw and annotated DE-EN corpus is different due to lack of meta on the target side
        # for 'ORG_SP_DE_000011' 'ORG_SP_DE_000118'
        not_found = []

        with gzip.open(s_outf, "wt") as src_out, gzip.open(t_outf, "wt") as tgt_out:
            # create nested lists for texts:
            fnames = set([i.strip() for i in df_texts.sdoc_id.tolist()])  # ORG_WR_EN_DE_000001
            my_dict['langpair'].append(langpair)
            my_dict['docs'].append(len(fnames))

            ol_swc = []
            ol_twc = []

            # edit text_id column to reflect the change in the ORG_WR naming conventions introduced to accommodate EN-to-ES lpair
            # this is kept for historical reasons
            if langpair == 'deen':
                slang_meta['text_id'] = slang_meta['text_id'].apply(lambda x: x.replace('DE_', 'DE_EN_'))
                # fix the actual meta! my_id="ORG_WR_DE_011065"
                slang_meta['meta'] = slang_meta['meta'].apply(lambda x: x.replace('my_id="ORG_WR_DE_', 'my_id="ORG_WR_DE_EN_'))
                missing_them = etwas_neues(smeta=slang_meta, current_fns=fnames, lpair=langpair)
                slang_meta, tlang_meta = add_fake_meta(smeta=slang_meta, tmeta=tlang_meta, missing_fns=missing_them, slang='de', tlang='en')
            else:
                slang_meta['text_id'] = slang_meta['text_id'].apply(lambda x: x.replace('EN_', 'EN_DE_'))
                slang_meta['meta'] = slang_meta['meta'].apply(lambda x: x.replace('my_id="ORG_WR_EN_', 'my_id="ORG_WR_EN_DE_'))
                missing_them = etwas_neues(smeta=slang_meta, current_fns=fnames, lpair=langpair)
                slang_meta, tlang_meta = add_fake_meta(smeta=slang_meta, tmeta=tlang_meta, missing_fns=missing_them,
                                                       slang='en', tlang='de')
            # just checking again
            _ = etwas_neues(smeta=slang_meta, current_fns=fnames, lpair=langpair)

            # iterate doc pairs
            corpus_total_segs = 0
            for idx, st_id in enumerate(fnames):
                doc_segs_dict['doc_id'].append(st_id)
                try:
                    st_tag = slang_meta[slang_meta['text_id'].str.contains(st_id)]['meta'].item()
                    mini_df = df_texts[df_texts.sdoc_id.str.contains(st_id)]  # leading zeros guard against overmatching
                    # print(st_id)

                    # I still get float object (probably NaN) as seg, drop them:
                    mini_df = mini_df.dropna(subset=['sseg', 'tseg'])
                    ssegs = [seg.strip() for seg in mini_df['sseg'].tolist()]  # .split() is evil! STANZA treats sent in seg as ONE sentence with dots inside!
                    align_sseg_ids = [num.strip().split('-')[-1] for num in mini_df['sseg_id'].tolist()]
                    tt_tag = tlang_meta[tlang_meta['text_id'] ==
                                        st_id.replace('ORG_WR_', 'TR_')]['meta'].item()
                    tsegs = [seg.strip() for seg in mini_df['tseg'].tolist()]  # .split()
                    align_tseg_ids = [num.strip().split('-')[-1] for num in mini_df['tseg_id'].tolist()]

                except ValueError:
                    print(f'Doc NOT in meta! {st_id}. Skipping ...')
                    not_found.append(st_id)
                    continue
                    # exit()

                src_out.write(st_tag + '\n')
                tgt_out.write(tt_tag + '\n')
                doc_swc = []
                doc_twc = []

                new_counter = 1
                tot_segs = 0

                # iterate segs in this doc pair
                for (sseg, tseg, sid, tid) in zip(ssegs, tsegs, align_sseg_ids, align_tseg_ids):
                    tot_segs += 1

                    # 06 November 2023: I am using raw input!
                    parsed_sseg = sl_stanza(sseg)
                    parsed_tseg = tl_stanza(tseg)
                    # otherwise for europ sid/tid are 0-based indices not to be trusted
                    src_out.write(f'<seg num="{new_counter}" sent_align="{new_counter}" seg_in="{sseg}">\n')
                    tgt_out.write(f'<seg num="{new_counter}" sent_align="{new_counter}" seg_in="{tseg}">\n')

                    new_counter += 1

                    # average wc per sent is counted for actual sents, NOT seg!
                    src_out, swc = write_vrt_seg(sents=parsed_sseg.sentences, my_outf=src_out)
                    tgt_out, twc = write_vrt_seg(sents=parsed_tseg.sentences, my_outf=tgt_out)
                    doc_swc.append(swc)
                    doc_twc.append(twc)

                    src_out.write("</seg>\n")
                    tgt_out.write("</seg>\n")

                if idx % 100 == 0 and idx > 0:
                    print(f'{idx / len(fnames) * 100:.2f} % of {langpair.upper()} processed')

                ol_swc.append(doc_swc)
                ol_twc.append(doc_twc)

                src_out.write(f'</text>\n')  # the closing </text> tag
                tgt_out.write(f'</text>\n')

                doc_segs_dict['segs'].append(tot_segs)
                corpus_total_segs += tot_segs

            my_dict['segs'].append(corpus_total_segs)

            my_dict['sl_toks'].append(np.sum([i for sub in ol_swc for i in sub]))
            my_dict['tl_toks'].append(np.sum([i for sub in ol_twc for i in sub]))
            my_dict['segs/doc'].append(
                f'{np.mean([len(sub) for sub in ol_swc]):.2f} +/-{np.std([len(sub) for sub in ol_swc]):.2f}')
            my_dict['toks/sseg'].append(
                f'{np.mean([i for sub in ol_swc for i in sub]):.2f} +/-{np.std([i for sub in ol_swc for i in sub]):.2f}')

        end = time.time()
        print(f'\nParsing {langpair} took {int(end - start) / 60:.2f} mins (with processors pre-loaded)\n')

    df = pd.DataFrame.from_dict(my_dict)
    print(df)
    df.to_csv(f'{args.out_stats}parsed_aligned_stats.tsv', sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
