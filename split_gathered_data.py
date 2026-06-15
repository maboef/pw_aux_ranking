import numpy as np
import pandas as pd
from src.split import dissimilaritydrive_global_balanced_cluster_split

kinase_data_random = pd.read_csv('../kinase_benchmarking/data/datasets/luukkonen_split/PCM_luukkonen_split_censored_extended.csv.gz')


targets = kinase_data_random['uniprot'].unique()
kinase_data_random = kinase_data_random.pivot(values='pchembl_value_Mean', index='SMILES', columns='uniprot')
kinase_data_random_subset = kinase_data_random.reset_index()
split = dissimilaritydrive_global_balanced_cluster_split(kinase_data_random_subset, targets, [80, 10, 10])
split.to_csv('kinase_data_cluster_split.csv', index=False)
