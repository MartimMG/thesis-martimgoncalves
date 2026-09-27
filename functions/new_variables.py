# her functions

import numpy as np
import pandas as pd

def cT(dataset):
    
    #Definição do mapping
    mapping = {
    'cTX': ['X'],
    'cTIs': ['Is', 0],
    'cT1': [1, '1a', '1b', '1c'],
    'cT2': [2],
    'cT3': [3],
    'cT4': [4, '4a', '4b', '4c', '4d']
    }
    
    #Function to apply the mapping
    
    # Define a function to apply the mapping
    def map_to_group(value):
        for group, values in mapping.items():
            if value in values:
                return group
        return None
    
    dataset['Grouped_T'] = dataset['T by imaging or clinical examination'].apply(map_to_group)

    dataset['cT'] = dataset['Grouped_T']

    #to remove the unknow value to then replace them
    dataset['cT'] = dataset['cT'].replace('cTX', np.nan)

def clinical_staging(dataset):
    dataset['Cancer stage'] = np.nan
    
    for i in range(len(dataset)):
        
        #Stage 3
        if (dataset.loc[i, 'N (Regional nodes affected)'] in [2, 3]) or \
           (dataset.loc[i, 'cT'] == 'cT4') or \
           (dataset.loc[i, 'cT'] == 'cT3' and dataset.loc[i, 'N (Regional nodes affected)'] == 1):
            dataset.at[i, 'Cancer stage'] = 3
            
        #Stage 1
        if dataset.loc[i, 'cT'] == 'cT1' and dataset.loc[i, 'N (Regional nodes affected)'] == 0:
            dataset.at[i, 'Cancer stage'] = 1
            
        #Stage 2
        if pd.isna(dataset.loc[i, 'Cancer stage']) and pd.notna(dataset.loc[i, 'cT']) and pd.notna(dataset.loc[i, 'N (Regional nodes affected)']) and dataset.loc[i, 'M (Distant metastases)'] != 1:
            dataset.at[i, 'Cancer stage'] = 2
            
       #Stage 0
        if (dataset.loc[i, 'cT'] == 'cTIs' and dataset.loc[i, 'N (Regional nodes affected)'] == 0 and dataset.loc[i, 'M (Distant metastases)'] != 1) or \
            (dataset.loc[i, 'cT'] == 'cTIs' and pd.isna(dataset.loc[i, 'N (Regional nodes affected)']) and dataset.loc[i, 'M (Distant metastases)'] != 1) or \
            (dataset.loc[i, 'cT'] == 'cT0' and dataset.loc[i, 'N (Regional nodes affected)'] == '0' and dataset.loc[i, 'M (Distant metastases)'] != 0):
            dataset.at[i, 'Cancer stage'] = 0
        #Stage 4
        if dataset.loc[i, 'M (Distant metastases)'] == 1:
            dataset.at[i, 'Cancer stage'] = 4    


def pT(dataset):
    
    #Definição do mapping
    mapping = {
    'pTIs': ['IS', 0],
    'pT1': [1, '1A', '1B', '1C', '1MIC'],
    'pT2': [2],
    'pT3': [3],
    'pT4': [4, '4A', '4B', '4C', '4D']
    }
    
    #Function to apply the mapping
    
    # Define a function to apply the mapping
    def map_to_group(value):
        for group, values in mapping.items():
            if value in values:
                return group
        return None
    
    dataset['pT'] = dataset['pT'].apply(map_to_group)


def response(dataset, ct, pt):
    
    stage_rank = {
        '0': 0, 
        'IS': 1, '1MIC': 2, 
        '1': 3, '1A': 4, '1B': 5, '1C': 6, 
        '2': 7, '3': 8, 
        '4': 9, '4A': 10, '4B': 11, '4C': 12, '4D': 13
    }

    def check_reduction(row):
        ct_rank = stage_rank.get(str(row[ct]).upper(), 0)
        pt_rank = stage_rank.get(str(row[pt]).upper(), 0)
        
        return 1 if pt_rank < ct_rank else 0

    dataset['Neoadjuvant Response'] = dataset.apply(check_reduction, axis=1)

    print('Done')

    