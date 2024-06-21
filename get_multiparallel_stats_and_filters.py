"""
UPD 21 June 2024
13 Mar
output counts of <8words and bypassed segments:

count <8 words in sseg + copies/bypassed segs -- in feature-based modes they are the segments bypassed by the instruction selector
in the self-guided modes and in the translation they are cases where the model decided not to improve the existing translation

python3 get_multiparallel_stats_and_filters.py
"""
import os
from collections import defaultdict

import pandas as pd

dfs = []
for thres_type in ['ratio2.5', 'std2']:
    print(f'Thres: {thres_type}')
    for tlang in ['de', "en"]:
        print(f'Lang: {tlang}')
        collector = defaultdict(list)
        multitab_filepath = [f for f in os.listdir('data/rewritten/curated/') if f'{thres_type}_{tlang}_7aligned' in f][0]
        multitab = pd.read_csv(f'data/rewritten/curated/{multitab_filepath}', sep='\t')

        for setup in ['translated_min', 'self-guided_min', 'self-guided_detailed',
                      'feature-based_min', 'feature-based_detailed']:
            long_ht = multitab[multitab['translation'].apply(lambda x: len(str(x).split()) > 8)]
            short_ht = multitab[multitab['translation'].apply(lambda x: len(str(x).split()) <= 8)]

            lose_short_ids = short_ht['seg_id'].tolist()
            shorts = multitab.shape[0] - long_ht.shape[0]
            assert shorts == len(lose_short_ids), 'Huston!'

            versions = long_ht[long_ht['translation'] != long_ht[setup]]
            versions = versions[['seg_id', 'source', 'translation', setup]]  # no copies between translation and mode
            # print(versions.head())
            # print(versions.shape)

            copies_df = long_ht[long_ht['translation'] == long_ht[setup]]
            lose_copies_ids = copies_df['seg_id'].tolist()

            copies = long_ht.shape[0] - versions.shape[0]
            assert copies == len(lose_copies_ids), 'Huston! Huston!'

            collector[setup].append(f'{(copies)/long_ht.shape[0] *100:.2f}')

            # lose_short_ids.extend(lose_copies_ids)  # we want to exclude only shorts, not copies to maintain comparability with HT

            save_filtered_mode_tables_to = f'data/rewritten/curated/no_shorts_and_copies/{thres_type}/{tlang}/'
            os.makedirs(save_filtered_mode_tables_to, exist_ok=True)
            versions.to_csv(f'{save_filtered_mode_tables_to}{setup}_aligned.tsv', sep='\t')

            save_unaffected_segids_to = f'data/rewritten/curated/lose_segids/{thres_type}/{tlang}/'
            os.makedirs(save_unaffected_segids_to, exist_ok=True)
            with open(f'{save_unaffected_segids_to}shorts_only_{setup}.ids', 'w') as outf:
                for i in lose_short_ids:
                    outf.write(i + '\n')

        lang_df = pd.DataFrame(collector)
        lang_df.insert(0, 'lang', tlang)
        lang_df.insert(0, 'thres_type', thres_type)

        dfs.append(lang_df)

res = pd.concat(dfs, axis=0)
print(res)

res.to_csv(f'data/stats/ratios_of_copied_over_items_by_mode_and_thres.tsv', sep='\t', index=False)



