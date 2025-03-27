import os

import numpy as np
import pandas as pd

# Read data
conferences = ['ACL', 'EMNLP', 'NAACL', 'EACL']
positions = ['main', 'findings']

num_papers = 0
num_pages = 0

for conference in conferences:
    for position in positions:
        print(f"Start for {conference}-{position}")
        
        data_path = f'./data/{conference}/{position}'

        data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pkl')]
        print("Num files: ", len(data_files))

        data = pd.concat([pd.read_pickle(f) for f in data_files], ignore_index=True)

        data['position'] = position

        # Drop row where references if empty or NULL
        idx_drop = data['text'].index[data['references'].isnull()]
        data.drop(idx_drop, inplace=True)
        idx_drop = data['text'].index[data['references'] == '']
        data.drop(idx_drop, inplace=True)

        # Drop all rows where titel contains 'Proceedings of the [0-9]*(?:st|nd|rd|th) Annual Meeting of the Association for Computational Linguistics'
        idx_drop = data['titel'].index[data['titel'].str.contains(r'Proceedings of the [0-9]*(?:st|nd|rd|th) Annual Meeting of the Association for Computational Linguistics')]
        data.drop(idx_drop, inplace=True)
        # Drop all rows where titel contains 'Findings of the Association for Computational Linguistic'
        idx_drop = data['titel'].index[data['titel'].str.contains(r'Findings of the Association for Computational Linguistic')]
        data.drop(idx_drop, inplace=True)

        # Save 
        data.to_pickle(f'data/{conference}_{position}.pkl')
        print(f"Finished for {conference}-{position}")

        num_papers += data.shape[0]
        num_pages += data['pages'].sum()

# Statistics
print("Num Papers: ", num_papers)
print("Num Pages: ", num_pages)

print("Finished..")