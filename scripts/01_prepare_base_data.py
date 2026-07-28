import numpy as np
import pandas as pd

import re
import json
import requests

from pathlib import Path
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import List

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem

from papyrus_scripts import PapyrusDataset, keep_protein_class, keep_quality, consume_chunks, keep_source

from chembl_webresource_client.new_client import new_client


def retrieve_kinase_data_from_Papyrus(version : str = '05.7', plusplus : bool = False, min_quality : str = 'low', sources: str = 'any'):

    """
    Gather kinase data from Papyrus v05.7

    Parameters
    ----------
    source_path : str
        Path to Papyrus data

    Returns
    -------
    all_kinase_data : pd.DataFrame
        Filtered kinase data
    """

    print('Retrieve kinase data from Papyrus...')
    data = (PapyrusDataset(version=version, plusplus=plusplus)
            .keep_protein_class(classes=[{'l3': 'Protein Kinase'}])
            .keep_quality(min_quality=min_quality)
            .keep_activity_type(activity_types=['any'])
            .keep_source(source=sources))
    all_kinase_data = data.consume_chunks(progress=True)
    print('Number of kinase targets: {}'.format(all_kinase_data.target_id.nunique()))
    print('Number of activity points before filtering: {}'.format(all_kinase_data.shape[0]))
    return all_kinase_data

# %%
def filter_data(data: pd.DataFrame):
    data['SMILES'] = (
        data['SMILES']
        .apply(Chem.MolFromSmiles)
        .apply(Chem.MolToSmiles)
    )

    # Remove compounds with MW > 1000 Da
    data = data[
        data.SMILES.apply(Chem.MolFromSmiles).apply(Descriptors.MolWt) < 1000
    ].reset_index(drop=True)

    print(f'Number of activity points after removing molecules with MW > 1000: {data.shape[0]}')
    return data


def abstract_parser(document_ids : List, keyword_list : List):

    """ 
    Parse abstracts from PubMed, PubChem and Crossref
    for binding type keywords

    Parameters:
        document_ids (list of str) : list of document IDs
        keyword_list (list of str) : list of keywords to search for in abstracts

    Returns:
        selected_abstracts (list of str) : list of document IDs (PMID) with abstracts containing keywords
    """

    # Get, parse and annotate abstracts
    selected_abstracts = []
    for doc_id in tqdm(document_ids, desc='Parsing abstracts'):
        try :
            if 'PMID' in doc_id:
                pmid = doc_id[5:]
                handle = efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
                abstract = handle.read().lower()
            elif 'PubChemAID' in doc_id:
                aid = doc_id[11:]
                handle = efetch(db='pcassay', id=aid, retmode='text', rettype='abstract')
                abstract = handle.read().lower()
            elif 'DOI' in doc_id:
                try:
                    url = f'https://api.crossref.org/works/{doc_id[4:]}'
                    r = requests.get(url)
                    crossref = r.json()
                except json.decoder.JSONDecodeError:
                    continue
                abstract = crossref['message']['abstract'].lower()
            else:
                continue
        except:
            print("Couldn't parse the abstract from {}".format(doc_id))
            continue

        for keyword in keyword_list:
            if keyword in abstract:
                selected_abstracts.append(doc_id)
                break

    return selected_abstracts

def chembl_description_parser(assay_ids: List, keyword_list: List):

    """ Parse chembl assay descriptions for binding type keywords

    Parameters:
        assay_ids (list of str) : list of assay IDs
        keyword_list (list of str) : list of keywords to search for in assay descriptions

    Returns:
        selected_assays (list of str) : list of assay IDs with descriptions containing keywords
    """

    assay = new_client.assay
    descriptions = assay.filter(assay_id__in=assay_ids).only(['description'])
    selected_assays = []
    for assay_id, description in tqdm(zip(assay_ids, descriptions), total=len(assay_ids), desc='Parsing assay descriptions'):
        description = description['description']
        if description is not None:
            description = description.lower()
            for keyword in keyword_list:
                if keyword in description:
                    selected_assays.append(assay_id)
                    break

    return selected_assays

def patent_parser(pantents : List, keywords : List):

    selected_abstracts = []
    for patent in tqdm(pantents, desc='Parsing patents'):
        url = f'https://patents.google.com/patent/{patent[7:]}/en'
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
        meta = soup.find_all("meta")

        for m in meta:
            abstract = 'test'
            if m.get("name") == "description":
                abstract = m.get("content").lower()
                # remove newlines and replace all blancs of any lenght with a single space
                abstract = re.sub(r' +', ' ', abstract.replace('\n', ' '))

        for keyword in keywords:
            if keyword in abstract:
                selected_abstracts.append(patent)
                break

    return selected_abstracts


def filter_allosteric_compounds(data):

    """
    Filter out allosteric compounds in 3 steps:

    1. Filter out compounds with a binding type keyword in the chembl assays description
    2. Filter out compounds with a binding type keyword in the abstracts (from PubMed or PubChem)
    3. Filter out compounds with a binding type keyword in the patents
    4. Filter out compounds with maximum Tanimoto similarity > 0.9 to compounds assigned to be allosteric in step 1,2 or 3

    Parameters
    ----------
    data : pd.DataFrame
        Papyrus data

    Returns
    -------
    data : pd.DataFrame
        Filtered Papyrus data
    """

    allosteric_keywords = [
        'activators',
        'allosteric',
        'allosterism',
        'allostery',
        'alosteric',
        'alostery',
        'alosterism',
        'indirect activation',
        'indirectly activate'
        'indirectly inhibit',
        'indirectly modulate',
        'negative modulator',
        'negative modulators',
        'nnrti', #non-nucleoside reverse transcriptase inhibitor
        'non-competitive',
        'non-nucleoside reverse transcriptase inhibitor',
        'non-substrate',
        'noncompetitive',
        'nonsubstrate',
        'positive modulator',
        'positive modulators',
        'receptor modulator',
        'regulatory site',
        'secondary binding site',
        'secondary pocket',
        'un-competitive',
        'uncompetitive',
        'pif', #PIF-binding pocket
        'myristoyl', #myristoyl pocket
        'pseudo-kinase',
        'pseudokinase',


    ]

    data_allo = data.copy()

    # 1. Filter out compounds with a binding type keyword in the chembl assays description    
    # Get all chembl assay IDs
    chembl_assay_ids = []
    for aids in data.AID.unique():
        for aid in aids.split(';'):
            if 'CHEMBL' in aid:
                chembl_assay_ids.append(aid)
    chembl_assay_ids = list(set(chembl_assay_ids))

    # Parse chembl assay descriptions
    selected_assays = chembl_description_parser(chembl_assay_ids, allosteric_keywords)

    # Drop allosteric compounds
    allosteric_smiles = []
    for smiles, aids in zip(data.SMILES, data.AID):
        for aid in aids.split(';'):
            if aid in selected_assays:
                allosteric_smiles.append(smiles)
                break
    data = data[~data.SMILES.isin(allosteric_smiles)].reset_index(drop=True)
    print(f'Number of activity points after removing allosteric compounds (ChEMBL assay descriptions): {data.shape[0]}')

    # 2. Filter out compounds with a binding type keyword in the abstracts (from PubMed or PubChem)
    # Get all PubChemAIDs, PMIDs and DOIs from document IDs
    parsable_docs = []
    for doc_ids in data.all_doc_ids.unique():
        for doc_id in doc_ids.split(';'):
            if 'PMID' in doc_id or 'PubChemAID' in doc_id or 'DOI' in doc_id:
                parsable_docs.append(doc_id)
    parsable_docs = list(set(parsable_docs))

    # Parse abstracts
    selected_doc_ids = abstract_parser(parsable_docs, allosteric_keywords)

    # Drop allosteric compounds
    for smiles, doc_ids in zip(data.SMILES, data.all_doc_ids):
        for doc_id in doc_ids.split(';'):
            if doc_id in selected_doc_ids:
                allosteric_smiles.append(smiles)
                break
    data = data[~data.SMILES.isin(allosteric_smiles)].reset_index(drop=True)
    print(f'Number of activity points after removing allosteric compounds (abstracts): {data.shape[0]}')

    # 3. Filter out compounds with a binding type keyword in the patents
    # Get all patent IDs
    patent_ids = []
    for doc_ids in data.all_doc_ids.unique():
        for doc_id in doc_ids.split(';'):
            if 'PATENT' in doc_id:
                patent_ids.append(doc_id)
    patent_ids = list(set(patent_ids))

    # Parse patents
    selected_patents = patent_parser(patent_ids, allosteric_keywords)

    # Drop allosteric compounds
    for smiles, doc_ids in zip(data.SMILES, data.all_doc_ids):
        for doc_id in doc_ids.split(';'):
            if doc_id in selected_patents:
                allosteric_smiles.append(smiles)
                break
    data = data[~data.SMILES.isin(allosteric_smiles)].reset_index(drop=True)
    print(f'Number of activity points after removing allosteric compounds (patents): {data.shape[0]}')


    # 4. Drop compounds similar to allosteric compounds
    if len(allosteric_smiles) == 0:
        return data

    allosteric_smiles = list(set(allosteric_smiles))
    other_smiles = data['SMILES'].unique()
    allosteric_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=2048) for smiles in allosteric_smiles]
    other_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=2048) for smiles in other_smiles]

    for smiles, fp in zip(other_smiles, other_fps):
        max_tanimoto = np.max(DataStructs.BulkTanimotoSimilarity(fp, allosteric_fps))
        if max_tanimoto > 0.8:
            data = data[data.SMILES != smiles]
            allosteric_smiles.append(smiles)

    data = data.reset_index(drop=True)
    print(f'Number of activity points after removing compounds similar to allosteric compounds: {data.shape[0]}')

    data_allo = data_allo[data_allo.SMILES.isin(allosteric_smiles)].reset_index(drop=True)

    return data, data_allo


def correct_relation(data): #, sources, pchem_vals):
    print(f"initiation relation counts: {data['relation'].value_counts()}")
    data['fixed_relation'] = data['relation']
    # is_selected = data['source'].isin(sources) & data['pchembl_value_Mean'].isin(pchem_vals)
    # data.loc[is_selected, 'fixed_relation'] = '<'
    mapping = {
        '<;<=': '<',
        '<=;<': '<',
        '<=':   '<',
        '>;>=': '>',
        '>=;>': '>',
        '>=':   '>'
    }
    data['fixed_relation'] = data['fixed_relation'].replace(mapping).fillna('=')
    print(f"fixed relation counts: {data['fixed_relation'].value_counts()}")
    return data


if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent.parent
    data_path = (current_dir / 'data/raw')
    Path(data_path).mkdir(parents=True, exist_ok=True)

    complete_papyrus_kinase_data = retrieve_kinase_data_from_Papyrus(version='05.7', plusplus=False, min_quality='low')
    # complete_papyrus_kinase_data.to_csv((data_path / '01_raw_kinase_data.csv.gz'), index=False)
    complete_papyrus_kinase_data = complete_papyrus_kinase_data.drop(['Classification', 'pchembl_value_MAD', 'pchembl_value_Median', 'pchembl_value_N', 'pchembl_value_SEM', 'pchembl_value_StdDev', 'InChI_AuxInfo', 'InChI'], axis=1, errors='raise')
    # complete_papyrus_kinase_data.to_csv((data_path / '01_raw_kinase_data.csv.gz'), index=False)
    size_filtered_kinase_data = filter_data(complete_papyrus_kinase_data)
    # size_filtered_kinase_data.to_csv((data_path / '02_size_kinase_data.csv.gz'), index=False)
    orthosteric_kinase_data, allosteric_kinase_data = filter_allosteric_compounds(size_filtered_kinase_data)
    # orthosteric_kinase_data.to_csv((data_path / '03_orthosteric_kinase_data.csv.gz'), index=False)
    # allosteric_kinase_data.to_csv((data_path / '03_allosteric_kinase_data.csv.gz'), index=False)
    # orthosteric_kinase_data = pd.read_csv(data_path / '03_orthosteric_kinase_data.csv.gz')
    # allosteric_kinase_data = pd.read_csv(data_path / '03_allosteric_kinase_data.csv.gz')

    allosteric_kinase_data_fixed = correct_relation(allosteric_kinase_data)
    orthosteric_kinase_data_fixed = correct_relation(orthosteric_kinase_data)
    
    orthosteric_kinase_data.to_csv((data_path / 'orthosteric_kinase_data_corr_rel.csv.gz'), index=False)
    allosteric_kinase_data.to_csv((data_path / 'allosteric_kinase_data_corr_rel.csv.gz'), index=False)


