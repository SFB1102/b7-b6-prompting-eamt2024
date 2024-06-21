"""
UPD 20 June 2024
9 Mar
get a table with differences between the baseline and the results on rewritten

python3 7_evaluation_and_analysis/format_diff_res.py
"""
from collections import defaultdict
import pandas as pd


def juggle_tables(base=None, test=None):
    minus_this = float(base['F1'].values[0].split()[0])
    print(minus_this)
    self_guided_collector = defaultdict(list)
    features_collector = defaultdict(list)
    retranslated_collector = defaultdict(list)
    ht_collector = defaultdict(list)
    
    ht_collector['--'].append(minus_this)
    
    # explicit control for the order
    for my_setup in ['translated_min', 'self-guided_min', 'self-guided_detailed',
                     'feature-based_min', 'feature-based_detailed']:

        setup_df = test[test['setup'] == my_setup]
        print(setup_df)
        this = float(setup_df['F1'].values[0].split()[0])
        print(my_setup, this)
        
        res = round((this - minus_this), 2)
        if my_setup in ['self-guided_min', 'self-guided_detailed']:
            if 'self-guided_min' in my_setup:  # 'lazy'
                self_guided_collector['Min'].append(res)
            else:
                self_guided_collector['Detail'].append(res)
        elif my_setup == 'translated_min':
            retranslated_collector['--'].append(res)
        else:
            if 'min' in my_setup:  #
                features_collector['Min'].append(res)
            else:
                features_collector['Detail'].append(res)
    
    ht_df = pd.DataFrame(ht_collector)
    # Create a new DataFrame with the header as the first row
    ht_header_as_row = pd.DataFrame([ht_df.columns], columns=ht_df.columns)
    # Concatenate the new DataFrame with the original DataFrame
    ht_df0 = pd.concat([ht_header_as_row, ht_df], ignore_index=True)
    # Create a new DataFrame with {tlang} as values and 'lang' as the index
    ht_new_row = pd.DataFrame({col: ['HT'] for col in ht_df0.columns}, index=[0])
    # Concatenate the new row DataFrame with the existing DataFrame
    ht = pd.concat([ht_new_row, ht_df0])
    ht = ht.reset_index(drop=True)

    mt_df = pd.DataFrame(retranslated_collector)
    # Create a new DataFrame with the header as the first row
    mt_header_as_row = pd.DataFrame([mt_df.columns], columns=mt_df.columns)
    # Concatenate the new DataFrame with the original DataFrame
    mt_df0 = pd.concat([mt_header_as_row, mt_df], ignore_index=True)
    # Create a new DataFrame with {tlang} as values and 'lang' as the index
    mt_new_row = pd.DataFrame({col: ['MT'] for col in mt_df0.columns}, index=[0])
    # Concatenate the new row DataFrame with the existing DataFrame
    mt = pd.concat([mt_new_row, mt_df0])
    mt = mt.reset_index(drop=True)

    self_df = pd.DataFrame(self_guided_collector)
    # Create a new DataFrame with the header as the first row
    self_header_as_row = pd.DataFrame([self_df.columns], columns=self_df.columns)

    # Concatenate the new DataFrame with the original DataFrame
    self_df0 = pd.concat([self_header_as_row, self_df], ignore_index=True)
    # Create a new DataFrame with {tlang} as values and 'lang' as the index
    self_new_row = pd.DataFrame({col: ['Self-guided'] for col in self_df0.columns}, index=[0])
    # Concatenate the new row DataFrame with the existing DataFrame
    first = pd.concat([self_new_row, self_df0])
    first = first.reset_index(drop=True)

    feats_df = pd.DataFrame(features_collector)
    # Create a new DataFrame with the header as the first row
    df_header_as_row = pd.DataFrame([feats_df.columns], columns=feats_df.columns)
    # Concatenate the new DataFrame with the original DataFrame
    feats_df0 = pd.concat([df_header_as_row, feats_df], ignore_index=True)

    # Create a new DataFrame with {tlang} as values and 'lang' as the index
    feats_new_row = pd.DataFrame({col: ['Feature-based'] for col in feats_df0.columns}, index=[0])
    # Concatenate the new row DataFrame with the existing DataFrame
    second = pd.concat([feats_new_row, feats_df0])
    second = second.reset_index(drop=True)

    both = pd.concat([ht, mt, first, second], axis=1)
    header = both.head(2)
    # Delete the first two rows
    my_vals = both.iloc[2:]

    return my_vals, header


# self-guided modes should be the same for the two thresholds but the multiparallel tables were curated separately
# which led to jitters in results (inc number of segments). I trust the std2 results for self-guided modes more.
for thres_type in ['ratio2.5', 'std2']:
    header = None
    final_tab_data = []
    for l in ['de', 'en']:
        for feats in [15, 58]:
            if feats == 15:
                baseline = pd.read_csv('2_classify1/res/seg_results_15feats.tsv', sep='\t')
                tests = pd.read_csv(f'6_classify2/res/seg_{thres_type}_rewritten_results_15feats.tsv', sep='\t')
            else:
                baseline = pd.read_csv('2_classify1/res/seg_results_0feats.tsv', sep='\t')
                tests = pd.read_csv(f'6_classify2/res/seg_{thres_type}_rewritten_results_58feats.tsv', sep='\t')
            # print(baseline)
            # print(tests)
            baseline = baseline[(baseline['N_feats'] == feats) & (baseline['Lang'] == l)]
            tests = tests[(tests['model'] == 'gpt4') & (tests['N_feats'] == feats) & (tests['Lang'] == l)]
            # print(tests)
            vals, header_rows = juggle_tables(base=baseline, test=tests)

            vals.insert(0, 'feats', feats)
            vals.insert(0, 'lang', l)
            vals.insert(0, 'model', 'gpt4')

            print(vals)

            vals.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8]

            header_rows.insert(0, 'feats', feats)
            header_rows.insert(0, 'lang', l)
            header_rows.insert(0, 'model', 'gpt4')
            header_rows.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8]

            header = header_rows
            final_tab_data.append(vals)
    # stack the rows
    my_res = pd.concat([header] + final_tab_data, axis=0)
    # Replace values in the first two rows of the first three columns with None
    my_res.iloc[:2, :3] = ''
    print(my_res)

    my_res.to_csv(f'{thres_type}_diffs_table.tsv', sep='\t', index=False)
