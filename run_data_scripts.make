
data/raw/orthosteric_kinase_data_corr_rel.csv.gz:
	python scripts/01_prepare_base_data.py

data/raw/pivotted_kinase_data_cluster_split.csv:
    python scripts/02_prepare_splits.py

data/protein_descriptors/kinase_domain_sequences.fasta:
    python scripts/03_prepare_protein_descriptors.py

data/datasets/saifudeen_percent_inhibition_data.csv:
    python scripts/04_prepare_saifudeen_data.py

data/datasets/cluster_split_ext_rank_cont_set.csv:
    python scripts/05_prepare_datasets.py