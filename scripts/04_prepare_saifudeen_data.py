import pandas as pd
import numpy as np
from pathlib import Path
import scipy.optimize as opt

def prep_percent_inhibition(pct_inhibition_data):
    pct_inhibition_melt = pct_inhibition_data.melt(id_vars=['Compound'], var_name='gene_name', value_name='percent_res_activity_1_uM')
    pct_inhibition_melt['percent_inhibition_1uM'] = 100 - pct_inhibition_melt['percent_res_activity_1_uM']
    pct_inhibition_melt['Name'] = pct_inhibition_melt['Compound']
    pct_inhibition_melt['Dose'] = '1uM'
    pct_inhibition_melt['percent_inhibition'] = pct_inhibition_melt['percent_inhibition_1uM']
    pct_inhibition_melt['experiment'] = 'pct_inhibition'
    
    pct_inhibition_melt['mut'] = np.where(
        pct_inhibition_melt['gene_name'].str.contains(r'\(.*\)', na=False),
        pct_inhibition_melt['gene_name'].str.extract(r'\((.*?)\)', expand=False),
        'WT'
    )
    pct_inhibition_melt['gene_name'] = pct_inhibition_melt['gene_name'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
    return pct_inhibition_melt

def prep_dose_curve(dose_curve_data):
    dose_curve_melt = dose_curve_data.melt(id_vars=['Compound', 'Name', 'Dose'], var_name='gene_name', value_name='percent_inhibition')
    dose_curve_melt['relative_activity'] = 100 - dose_curve_melt['percent_inhibition']
    dose_curve_melt['experiment'] = 'dose_curve'
    dose_curve_melt['Concentration (uM)'] = dose_curve_melt['Dose'].str.replace(r'uM', '', regex=True).str.strip()
    dose_curve_melt['Concentration (uM)'] = pd.to_numeric(dose_curve_melt['Concentration (uM)'])
    return dose_curve_melt


def ll2(x, hill, ec50):
    # Add a tiny epsilon to x to prevent log(0) errors
    x = np.where(x <= 0, 1e-9, x) 
    return (100 / (1 + np.exp(hill * (np.log(x) - np.log(ec50)))))


def pDose(x):
    '''This is just a helper function, to compute easily log transformed concentrations used in drug discovery'''
    return(-np.log10(1e-6*x))
    

def fit_curves(dose_curve_data):
    output = []
    for inhibitor in dose_curve_data['Name'].unique():
        input_table_1 = dose_curve_data[dose_curve_data['Name'] == inhibitor]
        #Define some text for plot    
        labeltxt = "Fit of {compound} \npEC50: {pic50:.2f} ± {sd:.2f} \nHill-slope: {hill:.1f}"
        ylabel = "Relative LFQ intensity"
        xlabel = f"log['{inhibitor}'] (M)"
        
        #Set upper and lower bounds for min and max response
        
        # maxconc = input_table_1[input_table_1.nlargest(1, 'Concentration (uM)')]
        minhill = 0.
        maxhill = 5.
        minec50 = -np.inf
        maxec50 = np.inf
        #find min dose of inhibitor
        # print(f'maxec50: {maxconc}')
        minconc = input_table_1.nsmallest(1, 'Concentration (uM)')
        
        protein_list = list(input_table_1['Entry'].unique())
        
        for protein in protein_list:
            fitCoefs = ['','']
            sds = ['','']
            sem = ''
            input_table_1_prot = input_table_1[input_table_1['Entry'] == protein]
            
            print('\n---')
            print(f'fitting {inhibitor} on {protein}')
    
            current_concentrations = input_table_1_prot['Concentration (uM)']
            guess_ec50 = np.median(current_concentrations)
            guess_hill = 1.0
            
            # 2. Smarter Bounds
            # EC50 shouldn't really be smaller than your lowest dose/100 
            # or larger than your highest dose*10
            min_c = current_concentrations[current_concentrations > 0].min()
            max_c = current_concentrations.max()
            
            try:
                # Use the 'lm' method or ensure bounds are realistic
                fitCoefs, covMatrix = opt.curve_fit(
                    ll2, 
                    input_table_1_prot['Concentration (uM)'], 
                    input_table_1_prot['relative_activity'], 
                    p0=[guess_hill, guess_ec50], 
                    bounds=([0.1, min_c / 100], [10.0, max_c * 10]), # Constrain to realistic chemistry
                    maxfev=2000
                )
                # Calculate Standard Errors from the covariance matrix
                perr = np.sqrt(np.diag(covMatrix))
                sds = perr 
            except Exception as e:
                print(f'Fit failed for {protein}: {e}')
                fitCoefs = [np.nan, np.nan]
            
            print(f"maxconc: {input_table_1['Concentration (uM)'].max()} uM")
            if fitCoefs[1] > input_table_1['Concentration (uM)'].max():
                print(f'calculated EC50 larger than max conc - not fitted - pEC50: ({pDose(fitCoefs[1])})')
                output.append([inhibitor, protein, None, None])
            else:
                print(f'pEC50 {inhibitor} - {protein} fitted at ({pDose(fitCoefs[1])})')
                output.append([inhibitor, protein, str(fitCoefs[1]),str(fitCoefs[0]), pDose(fitCoefs[1])])
            print('---\n')
            
    output_table_1 = pd.DataFrame(output)
    output_table_1 = output_table_1.rename(columns={0: 'Inhibitor', 1: 'Uniprot ID', 2: 'EC50 (uM)', 3: 'Hillslope', 4:'pEC50'})
    return output_table_1



if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent.parent
    
    pct_inhibition_data = pd.read_csv(current_dir / 'data/saifudeen_2026_raw/raw_percent_inhibition_full_set.csv')
    dose_curve_data = pd.read_csv(current_dir / 'data/saifudeen_2026_raw/raw_dose_curves_full_set.csv')
    compound_mapping = pd.read_csv(current_dir / 'data/saifudeen_2026_raw/name_SMILES_compound_mapping.csv')
    target_mapping = pd.read_csv(current_dir / 'data/saifudeen_2026_raw/kinase_uniprot_target_mapping.csv')
    
    pct_inhibition_melt = prep_percent_inhibition(pct_inhibition_data)
    pct_inhibition_melt = pd.merge(pct_inhibition_melt, target_mapping[['Entry', 'Gene names (primary)']], left_on='gene_name', right_on='Gene names (primary)')
    pct_inhibition_melt = pd.merge(pct_inhibition_melt, compound_mapping[['InChiKey', 'drug_code', 'drug_name', 'SMILES']], left_on='Name', right_on='drug_name')
    pct_inhibition_melt['activity_id'] = pct_inhibition_melt['drug_name'] + '_on_' + pct_inhibition_melt['Entry']
    pct_inhibition_melt.to_csv(current_dir / 'data/datasets/saifudeen_percent_inhibition_data.csv', index=False)

    dose_curve_melt = prep_dose_curve(dose_curve_data)
    dose_curve_melt = pd.merge(dose_curve_melt, target_mapping[['Entry', 'Gene names (primary)']], left_on='gene_name', right_on='Gene names (primary)')
    dose_curve_melt = pd.merge(dose_curve_melt, compound_mapping[['InChiKey', 'drug_code', 'drug_name', 'SMILES']], left_on='Name', right_on='drug_name')
    dose_curve_melt['activity_id'] = dose_curve_melt['drug_name'] + '_on_' + dose_curve_melt['Entry']
    ec50s = fit_curves(dose_curve_melt)
    ec50s['activity_id'] = ec50s['Inhibitor'] + '_on_' + ec50s['Uniprot ID']
    dose_curve_melt = dose_curve_melt[['Name', 'Entry', 'gene_name', 'activity_id', 'InChiKey', 'drug_code', 'drug_name', 'SMILES']].drop_duplicates()
    dose_curve_melt = pd.merge(dose_curve_melt, ec50s[['EC50 (uM)', 'Hillslope', 'pEC50', 'activity_id']], on='activity_id')
    dose_curve_melt.to_csv(current_dir / 'data/datasets/saifudeen_pec50_data.csv', index=False)
    
    

