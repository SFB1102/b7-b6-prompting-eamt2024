"""
25 Oct 2023
--combine rename Europarl-UdS-2018 and IATA parallel term bases for EP
--reverse en-es (einmal)
--save to /home/maria/tools/aligner/scripts/hunalign/data/

NB! give your path to LF Aligner folder to --res
python3 concat_dic.py --old glossaries/biling_dicts2018/ --new glossaries/iate2hunalign/ --res aligner/scripts/hunalign/data/

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
    # data/2018/biling_dicts2018/renamed/
    parser.add_argument('--old', help="dir with renamed karakanta's dics", required=True)
    # data/iate_dicts/2hunalign/
    parser.add_argument('--new', help="dir with iata bi-ling term glossaries from EP", required=True)
    # parser.add_argument('--lpair', required=True, choices=['deen', 'ende', 'esen', 'enes'])
    parser.add_argument('--res', default='aligner/scripts/hunalign/data/', required=True,
                        help='move these dictionaries to your LF Aligner: aligner/scripts/hunalign/data/')
    parser.add_argument('--logs', default='logs/bi_dict/')

    args = parser.parse_args()

    start = time.time()
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')

    os.makedirs(args.res, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    log_file = f'{args.logs}{formatted_datetime}_{sys.argv[0].split("/")[-1].split(".")[0]}.log'

    sys.stdout = Logger(logfile=log_file)

    print(f"\nRun date, UTC: {formatted_datetime}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")
    esen_generic = []
    old = None
    for lpair in ['deen', 'ende', 'esen', 'enes']:
        print(lpair)
        try:
            old = [i for i in os.listdir(args.old) if f"{lpair[:2]}-{lpair[2:]}" in i][0]
        except IndexError:
            print(f"{lpair[:2]}-{lpair[2:]}")  # es-en
            # continue
            if lpair == 'esen':
                # revers the generic dict
                opposite = open(args.old + 'en-es.dic', 'r').readlines()
                reversed_name = 'es-en.dic'
                if os.path.exists(reversed_name):
                    esen_generic = open(args.old + reversed_name, 'r').read()
                else:
                    with open(args.old + reversed_name, 'w') as out_esen:
                        for line in opposite:
                            bits = line.split('@')
                            out_esen.write(f'{bits[1].strip()} @ {bits[0].strip()}\n')
                            esen_generic.append(f'{bits[1].strip()} @ {bits[0].strip()}')
            else:
                print('This should not happen!')
                continue

        new = [i for i in os.listdir(args.new) if f"{lpair[:2]}-{lpair[2:]}" in i][0]
        with open(args.res + f"{lpair[:2]}-{lpair[2:]}.dic", 'w') as outf:
            add_terms = open(args.new + new, 'r').read()
            if lpair != 'esen':
                generic = open(args.old + old, 'r').read()
            else:
                generic = "\n".join(esen_generic)
            for i in [generic, add_terms]:
                outf.write(i)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
