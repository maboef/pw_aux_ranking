.PHONY: all

all: data/saifudeen_2026_raw/raw_percent_inhibition_full_set.csv data/raw/orthosteric_kinase_data_corr_rel.csv.gz data/raw/pivotted_kinase_data_cluster_split.csv protein_descriptors/kinase_domain_sequences.fasta data/datasets/saifudeen_percent_inhibition_data.csv data/datasets/cluster_split_ext_rank_cont_set.csv

data/saifudeen_2026_raw/raw_percent_inhibition_full_set.csv:
	python scripts/extract_xlsx.py

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