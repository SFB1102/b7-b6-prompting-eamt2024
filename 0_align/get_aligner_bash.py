"""
23 Oct 2023

from a deen folder with 20936 doc pairs like 19990720.2-021.de--19990720.2-021.en
obtained by running
python3 0_align/extract_interventions.py xml_translationese/de/originals_ns/ xml_translationese/en/translations_from_de_n/ extract/deen/

produce a sh script deen_europarl-uds.sh
containing for each doc-pair lines like (ADJUST the paths!)
perl ./scripts/LF_aligner_3.11_with_modules.pl --filetype="t" --infiles="ende/19991117.3-223.en.txt","ende/19991117.3-223.de.txt" --languages="en","de" --segment="y" --review="xn" --tmx="y" --outfile="ende/out_deen.tmx"

This script is run from your local installation of LF Aligner (download from https://sourceforge.net/projects/aligner/)

python3 get_aligner_bash.py --indir extracted/enes/ --lpair enes
"""
import argparse
import os
import sys
import time
from datetime import datetime


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
    parser.add_argument('--indir', help="a folder with pairs of files *.en.txt and *.de.txt", required=True)
    parser.add_argument('--lpair', required=True, choices=['deen', 'ende', 'esen', 'enes'])
    parser.add_argument('--res', default='/home/maria/tools/aligner/')
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

    slang = args.indir.rsplit('/', 2)[1][:2]
    tlang = args.indir.rsplit('/', 2)[1][2:]
    print(slang, tlang)

    fh = [f for f in os.listdir(args.indir)]
    # I need files ending in .txt
    my_bases = set([f[:-7] for f in fh])
    print(len(my_bases))

    outf = f'{args.res}{args.lpair}_europarl-uds.sh'
    with open(outf, 'w') as outfile:
        outfile.write('#!/bin/bash\ncd "`dirname "$0"`"' + '\n')
        for i in my_bases:
            template = f'perl ./scripts/LF_aligner_3.11_with_modules.pl --filetype="t" --infiles="/home/maria/main/proj/b7/{args.indir}{i}.{slang}.txt","/home/maria/main/proj/b7/{args.indir}{i}.{tlang}.txt" --languages="{slang}","{tlang}" --segment="auto" --review="xn" --tmx="n" --outfile="/home/maria/main/out_{args.lpair}.tmx"'
            outfile.write(template + '\n')
