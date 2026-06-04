import os
import tqdm
import numpy as np
import pandas as pd

import requests

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator


def mkdirs(path : str):
    """Create a directory if it does not exist."""

    if not os.path.exists(path) : os.makedirs(path)

def compute_fps(data):
    """Compute Morgan Fingerprints from SMILES."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fps = pd.DataFrame(np.array([mfpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in tqdm.tqdm(data.SMILES, desc='Computing Morgan Fingerprints from SMILES')]), index=data.index)

    # fps = pd.DataFrame(np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=2048) for smiles in tqdm.tqdm(data.SMILES, desc='Computing Morgan Fingerprints from SMILES')]), index=data.index)

    return fps

def add_protein_descriptors_and_pad(data, prot_descriptor_path):
    prot_descriptor = pd.read_pickle(prot_descriptor_path)
    prot_descriptor['protein_descriptor'] = prot_descriptor['protein_descriptor'].apply(list)
    # print(prot_descriptor['protein_descriptor'].apply(type))
    data = pd.merge(left=data, right=prot_descriptor, left_on='accession', right_on='target_id', how='left')
    missing = data[data['protein_descriptor'].isna()]
    print(f'{len(missing)} datapoints without protein descriptor')
    print(f"proteins without descriptor added: {(missing['accession'].unique())}")
    data = data[~data['protein_descriptor'].isna()].reset_index()
    return data


'''
def add_protein_descriptors_and_pad(data, prot_descriptor_path):
    prot_descriptor = pd.read_pickle(prot_descriptor_path)
    prot_descriptor = prot_descriptor.drop(['UniProtID', 'Status', 'Organism',
                                            'Classification', 'Status',
                                            'Length', 'Sequence', 'TID',
                                            'uniprot', 'mut', 'accession',
                                            'atp_binding_sites', 'min_site', 'max_site',
                                            'numbered', 'embedding', 'window',
                                            'Sequence_window'], axis=1, errors='ignore')
    columns = prot_descriptor.columns
    prot_descriptor = prot_descriptor.rename(columns = {f"{columns[0]}": f"{columns[0]}", f"{columns[1]}": "protein_descriptor"}) 
    prot_descriptor['protein_descriptor'] = prot_descriptor['protein_descriptor'].apply(list)
    # print(prot_descriptor['protein_descriptor'].apply(type))
    data = pd.merge(left=data, right=prot_descriptor, left_on='accession', right_on='target_id', how='left')
    missing = data[data['protein_descriptor'].isna()]
    print(f'{len(missing)} datapoints without protein descriptor')
    print(f"proteins without descriptor added: {(missing['accession'].unique())}")
    data = data[~data['protein_descriptor'].isna()].reset_index()
    max_len = data['protein_descriptor'].apply(len).max()
    # data['protein_descriptor'] = data['protein_descriptor'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data['protein_descriptor'] = data['protein_descriptor'].apply(lambda l: l + [0] * (max_len - len(l)))
    return data
'''

def get_atp_binding_residues(accession):
    print(accession)
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    response = requests.get(url)

    if not response.ok:
        return {"accession": accession, "error": f"Failed with {response.status_code}"}

    data = response.json()
    binding_sites = []

    for feature in data.get("features", []):
        if feature.get("ligand", {}).get("name") == "ATP":
            print(f"ATP binding site found for {accession}")
            binding_sites.append(feature["location"]["start"]["value"])
            binding_sites.append(feature["location"]["end"]["value"])
        elif feature.get("ligand", {}).get("name") == "ADP":
            print(f"ADP binding site found for {accession}")
            binding_sites.append(feature["location"]["start"]["value"])
            binding_sites.append(feature["location"]["end"]["value"])
        elif feature.get("ligand", {}).get("name") == "GTP":
            print(f"GTP binding site found for {accession}")
            binding_sites.append(feature["location"]["start"]["value"])
            binding_sites.append(feature["location"]["end"]["value"])
        elif feature.get("description", {}) == "PI3K/PI4K catalytic": # specifically made for P78527
            print(f"PI3K/PI4K catalytic domain found for {accession}")
            binding_sites.append(feature["location"]["start"]["value"])
            binding_sites.append(feature["location"]["end"]["value"])
        # else:
        #     print(f"No relevant site found for {accession}")
    return {"accession": accession, "atp_binding_sites": binding_sites}