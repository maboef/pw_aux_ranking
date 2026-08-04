# Pairwise Auxiliary Ranking Task for Bioactivity Modelling

![Status](https://img.shields.io/badge/status-WIP-yellow)

> 🚧 **Work in progress.** This repo is under active development — code works but data gathering is being automated. 

## About

Repo for the work on Pairwise Auxiliary Ranking Loss for Bioactivity modelling - [PREPRINT](https://doi.org/10.26434/chemrxiv.15006577/v1)


.xlsx file under data/raw_saifudeen_2026 is from the SI [Saifudeen_2026](https://doi.org/10.1038/s41587-026-03090-8), Supplementary Tables 1-13. This falls under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, http://creativecommons.org/licenses/by-nc-nd/4.0/.

## Environment

Needs Python version >=3.11 - then install using the requirements.txt file.


## Generating of datasets

```
clone the repo

cd pw_aux_ranking

$>make -f run_data_scripts.make
```
