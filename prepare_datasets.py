import pandas as pd
import numpy as np

from rdkit import DataStructs
from rdkit.Chem import Descriptors, AllChem
from papyrus_structure_pipeline import standardizer as Papyrus_standardizer

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standardize_compounds(smiles):
    try:
        mol = Papyrus_standardizer.Chem.MolFromSmiles(smiles)
        mol = Papyrus_standardizer.standardize(mol)
        Papyrus_standardizer.Chem.RemoveStereochemistry(mol)
        InChIKey = Papyrus_standardizer.AllChem.MolToInchiKey(mol)
        InChI, InChI_AuxInfo = Papyrus_standardizer.AllChem.MolToInchiAndAuxInfo(mol)
        smiles_ = Papyrus_standardizer.Chem.MolToSmiles(mol)
        return smiles_, InChIKey
    except:
        print(f'error standardizing smiles {smiles}')
        smiles_ = smiles
        InChIKey = np.nan
        InChI = np.nan
        return smiles_, InChIKey

def standardize_saifudeen(saifudeen_data):
    saifudeen_data['source'] = 'Saifudeen2026'
    saifudeen_data['accession'] = saifudeen_data['Entry'] 
    saifudeen_data['target_id'] = saifudeen_data['accession'] + '_' + saifudeen_data['mut']
    saifudeen_data['Activity_ID'] = saifudeen_data['InChiKey'].str[:-13] + '_on_' + saifudeen_data['target_id']
    saifudeen_data['InChIKey'] = saifudeen_data['InChiKey'] 
    saifudeen_data['Year'] = 2026
    saifudeen_data[['relation', 'fixed_relation']] = '=', '='
    saifudeen_data[['type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other']] = 0, 0, 0, 0, 1
    saifudeen_data['Subset'] = 'train'
    saifudeen_data['pchembl_value_Mean'] = np.nan
    bins = [-1, 30, 70, 100]
    labels = [0, 1, 2]
    pct_inh_saifudeen_data['rank_split'] = pd.cut(pct_inh_saifudeen_data['percent_inhibition'], bins=bins, labels=labels)
    return saifudeen_data

def standardize_base(base_data)
    bins = [0, 5.5, 7.5, 15]
    labels = [0, 1, 2]
    base_data['rank_split'] = pd.cut(base_data['pchembl_value_Mean'], bins=bins, labels=labels)
    condition = (base_data['fixed_relation'].str.contains('<', na=False)) & (cluster_base['pchembl_value_Mean'] > 5.5)
    base_data.loc[condition, 'rank_split'] = np.nan
    base_data['percent_inhibition'] = np.nan
    base_data['mut'] = 'WT'
    return base_data

def merge_sets(base_data, saifudeen_data):
    merge_key = 'Activity_ID'
    base_data_indexed = base_data.set_index(merge_key)
    saifudeen_data_indexed = saifudeen_data.set_index(merge_key)
    merged_sets = base_data_indexed.combine_first(saifudeen_data_indexed)
    merged_sets = merged_sets.reset_index(drop=True)
    merged_sets = merged_sets[merged_sets['mut'] == 'WT']
    return merged_sets
    


if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent
    data_path = (current_dir / 'data/raw')

    random_split = pd.read_csv(data_path / 'pivotted_kinase_data_random_split.csv')
    cluster_split = pd.read_csv(data_path / 'pivotted_kinase_data_cluster_split.csv')
    regression_data = pd.read_csv(data_path / 'regression_kinase_data.csv.gz')
    
    regression_data['rank_split'] = np.nan
    regression_data[['Activity_ID', 'Quality', 'source', 'SMILES', 'target_id', 'accession', 'fixed_relation', 'pchembl_value_Mean']]
    necessary_columns = ['SMILES', 'Activity_ID', 'InChIKey', 'Subset', 'accession', 'mut', 'pchembl_value_Mean', 'Quality', 'rank_split', 'percent_inhibition', 'relation', 'fixed_relation']
    cluster_base = cluster_base[necessary_columns]
    random_base = random_base[necessary_columns]
    cluster_base = pd.merge(regression_data, cluster_split[['SMILES', 'Subset']], on='SMILES')
    random_base = pd.merge(regression_data, random_split[['SMILES', 'Subset']], on='SMILES')
    cluster_base.to_csv(current_dir / 'data/datasets/filtered_cluster_split_base_set.csv', index=False)
    random_base.to_csv(current_dir / 'data/datasets/filtered_random_split_base_set.csv', index=False)
    
    # inclusion of rank_split column
    cluster_rank = standardize_base(cluster_base)
    random_rank = standardize_base(random_base)
    cluster_rank.to_csv('data/datasets/cluster_split_base_rank_set.csv', index=False)
    random_rank.to_csv('data/datasets/random_split_base_rank_set.csv', index=False)

    # gather saifudeen percent inhibition data and merge with base data
    pct_inh_saifudeen_data = pd.read_csv('data/datasets/saifudeen_percent_inhibition_data.csv')
    compounds = pd.DataFrame({'SMILES':pct_inh_saifudeen_data['SMILES'].unique(), 'drug_name':pct_inh_saifudeen_data['drug_name'].unique()})
    smiles, inchikeys = standardize_compounds(compounds.SMILES)
    compounds['SMILES'] = smiles

    pct_inh_saifudeen_data = pct_inh_saifudeen_data.drop('SMILES', axis=1)
    pct_inh_saifudeen_data = pct_inh_saifudeen_data.merge(compounds, on='drug_name')
    pct_inh_saifudeen_data = standardize_saifudeen(pct_inh_saifudeen_data)
    pct_inh_saifudeen_data = pct_inh_saifudeen_data[necessary_columns]

    cluster_merged = merge_sets(cluster_rank, pct_inh_saifudeen_data)
    random_merged = merge_sets(random_rank, pct_inh_saifudeen_data)

    cluster_merged['rank_pchembl_value_Mean'] = cluster_merged['pchembl_value_Mean'] 
    cluster_merged['rank_percent_inhibition'] = cluster_merged['percent_inhibition'] 
    
    random_merged['rank_pchembl_value_Mean'] = random_merged['pchembl_value_Mean'] 
    random_merged['rank_percent_inhibition'] = random_merged['percent_inhibition'] 
    
    cluster_merged.to_csv('data/datasets/cluster_split_ext_rank_cont_set.csv', index=False)
    random_merged.to_csv('data/datasets/random_split_ext_rank_cont_set.csv', index=False)







