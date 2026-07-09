import pandas as pd
import numpy as np
from pathlib import Path

from src.split import dissimilaritydrive_global_balanced_cluster_split, random_global_equilibrated_random_split

def keep_valid_spread(val, max_spread):
    '''
    apply function for removing rows of pchembl_value with higher spread than max_spread
    '''
    if not isinstance(val, str) or ';' not in val:
        return True
    numbers = [float(x.strip()) for x in val.split(';') if x.strip()]
    return (max(numbers) - min(numbers)) <= max_spread

def filter_targets(data, min_datapoints=200):
    '''
    Filter out targets that have less than 200 regression high quality datapoints
    '''
    hq_data = data[data['Quality'] == 'High']
    targets = hq_data['target_id'].value_counts()[hq_data['target_id'].value_counts() > min_datapoints]
    data = data[data['target_id'].isin(targets.index)]
    print(f'{len(targets.index)} targets with more than 200 datapoints')
    return data, targets.index


if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent
    data_path = (current_dir / 'data/raw')

    orthosteric_kinase_data = pd.read_csv(data_path / '04_orthosteric_kinase_data_corr_rel.csv.gz') # NOTE switch to .csv.gz

    # Pre filtering
    regression_data = orthosteric_kinase_data[orthosteric_kinase_data['pchembl_value_Mean'].notna()]
    binary_data =  orthosteric_kinase_data[orthosteric_kinase_data['pchembl_value_Mean'].isna()]
    regression_data = regression_data[regression_data['type_other'] != '1']
    regression_data = regression_data[regression_data['pchembl_value'].apply(keep_valid_spread, max_spread=1)]
    regression_data, targets = filter_targets(regression_data)
    binary_data = binary_data[binary_data['target_id'].isin(targets)]
    binary_data.to_csv(data_path / 'binary_kinase_data.csv', index=False)
    
    # pivotting and splitting on compound SMILES
    regression_data = regression_data.reset_index()
    
    regression_data_pivot = regression_data.pivot(values='pchembl_value_Mean', index='SMILES', columns='target_id')
    regression_data_pivot = regression_data_pivot.reset_index()
    random_split = random_global_equilibrated_random_split(regression_data_pivot, targets, seed=24)
    random_split.to_csv(data_path / 'pivotted_kinase_data_random_split.csv', index=False)
    cluster_split = dissimilaritydrive_global_balanced_cluster_split(regression_data_pivot, targets, [80, 10, 10])
    cluster_split.to_csv(data_path / 'pivotted_kinase_data_cluster_split.csv', index=False)

    # inclusion of rank_split column
    regression_data['rank_split'] = np.nan
    regression_data[['Activity_ID', 'Quality', 'source', 'SMILES', 'target_id', 'accession', 'fixed_relation', 'pchembl_value_Mean']]
    cluster_base = pd.merge(regression_data, cluster_split[['SMILES', 'Subset']], on='SMILES')
    random_base = pd.merge(regression_data, random_split[['SMILES', 'Subset']], on='SMILES')
    cluster_base.to_csv(current_dir / 'data/datasets/cluster_split_base_set.csv', index=False)
    random_base.to_csv(current_dir / 'data/datasets/random_split_base_set.csv', index=False)

