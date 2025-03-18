#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:07:21 2025

@author: jc278357
"""

import mne
import os
import glob
import numpy as np

import pandas as pd
import numpy as np
import seaborn as sns

import pingouin as pg
import statsmodels.api as sm

# Define the folder
eeg_folder = '/volatile/home..../EEG_127TSA'

# Load the list of channel names from the text file
ch_names = np.loadtxt('ch_names.txt', dtype=str)

# Define frequency bands
beta_band = [14, 30]
gamma_band = [30, 49]

# Initialize matrices to store all data
all_data_beta = []
all_data_gamma = []
subjects = []

# Define electrodes for each region of interest
frontal_electrodes = ["E1", "E2", "E3", "E4", "E5", "E8", "E9", "E10", "E11", "E12", "E15", "E16", "E18", "E19", "E20", "E22", "E23", "E24", "E25", "E26", "E27", "E32", "E118", "E123", "E124", "E125", "E128"]
central_electrodes = ["E6", "E7", "E13", "E29", "E30", "E31", "E36", "E37", "E53", "E54", "E55", "E79", "E80", "E86", "E87", "E104", "E105", "E106", "E111", "E112"]
parietal_electrodes = ["E58", "E59", "E60", "E61", "E62", "E66", "E67", "E71", "E72", "E76", "E77", "E78", "E84", "E85", "E91", "E96"]
occipital_electrodes = ["E63", "E64", "E65", "E68", "E69", "E70", "E73", "E74", "E75", "E81", "E82", "E83", "E88", "E89", "E90", "E94", "E95", "E99"]
temporal_right_electrodes = ["E92", "E93", "E97", "E98", "E100", "E101", "E102", "E103", "E107", "E108", "E109", "E110", "E113", "E114", "E115", "E116", "E117", "E120", "E121", "E122"] 
temporal_left_electrodes = ["E28", "E33", "E34", "E35", "E38", "E39", "E40", "E41", "E42", "E43", "E44", "E45", "E46", "E47", "E49", "E50", "E51", "E52", "E56", "E57"] 


def process_and_accumulate_band(eeg_file, ch_names, frequency_band, all_data):
    # Load the EEG file
    raw = mne.io.read_raw_fif(eeg_file, preload=True)

    # Remove channels that are not on the scalp
    raw = raw.drop_channels(['E14', 'E17', 'E21', 'E48', 'E119', 'E126', 'E127'])
    raw.info['bads']  # check that there are no more bads 

    # Remove the first and last second to avoid edge effects
    start_time = raw.times[0] + 1.0
    end_time = raw.times[-1] - 1.0
    raw_band = raw.crop(tmin=start_time, tmax=end_time)

    # Compute the PSD using the Welch method for the specified frequency band
    psd_mean, freqs = raw_band.compute_psd(method='welch', picks='eeg', reject_by_annotation=True, fmin=frequency_band[0], fmax=frequency_band[1]).get_data(return_freqs=True)
   
    psd_meann = np.mean(psd_mean, axis=1)
    
    # Convert from V² to µV²
    psd_meann *= 1e12  
    
    # Transpose the data matrix
    data_matrix_transposed = np.transpose(psd_meann)

    # Accumulate data into the global matrix
    all_data.append([data_matrix_transposed])


# Function to group electrodes and compute the average for each region of interest
def calculate_roi_averages(data_matrix_transposed, electrodes):
    ch_names_list = list(ch_names)
    electrode_indices = [ch_names_list.index(electrode) for electrode in electrodes]
    roi_averages = np.nanmean(data_matrix_transposed[:, electrode_indices], axis=1)
    return roi_averages


# Iterate through all EEG files in the folder
for eeg_file in glob.glob(os.path.join(eeg_folder, '*.fif')):
    
    # Add the file name to the list
    subjects.append(os.path.basename(eeg_file).split('.')[0])
    
    process_and_accumulate_band(eeg_file, ch_names, beta_band, all_data_beta)
    process_and_accumulate_band(eeg_file, ch_names, gamma_band, all_data_gamma)

# Create a DataFrame for each frequency band with accumulated data
df_beta = pd.DataFrame(np.concatenate(all_data_beta, axis=0), columns=ch_names)
df_beta.insert(0, 'subject', subjects)
df_beta['subject'] = df_beta['subject'].str.replace('-RS_eeg', '')


# Add columns for the average of electrodes in each region of interest
df_beta['Frontal'] = calculate_roi_averages(df_beta.values[:, 1:], frontal_electrodes)
df_beta['Central'] = calculate_roi_averages(df_beta.values[:, 1:], central_electrodes)
df_beta['Occipital'] = calculate_roi_averages(df_beta.values[:, 1:], occipital_electrodes)
df_beta['Parietal'] = calculate_roi_averages(df_beta.values[:, 1:], parietal_electrodes)
df_beta['Temporal_Right'] = calculate_roi_averages(df_beta.values[:, 1:], temporal_right_electrodes)
df_beta['Temporal_Left'] = calculate_roi_averages(df_beta.values[:, 1:], temporal_left_electrodes)

# Create a DataFrame for each frequency band with accumulated data
df_gamma = pd.DataFrame(np.concatenate(all_data_gamma, axis=0), columns=ch_names)
df_gamma.insert(0, 'subject', subjects)
df_gamma['subject'] = df_gamma['subject'].str.replace('-RS_eeg', '')


# Add columns for the average of electrodes in each region of interest
df_gamma['Frontal'] = calculate_roi_averages(df_gamma.values[:, 1:], frontal_electrodes)
df_gamma['Central'] = calculate_roi_averages(df_gamma.values[:, 1:], central_electrodes)
df_gamma['Occipital'] = calculate_roi_averages(df_gamma.values[:, 1:], occipital_electrodes)
df_gamma['Parietal'] = calculate_roi_averages(df_gamma.values[:, 1:], parietal_electrodes)
df_gamma['Temporal_Right'] = calculate_roi_averages(df_gamma.values[:, 1:], temporal_right_electrodes) 
df_gamma['Temporal_Left'] = calculate_roi_averages(df_gamma.values[:, 1:], temporal_left_electrodes)

#------------------------------------------------------------------------------------------------------------------------
# Create the full dataset with E/I and clinical variables 
pheno = pd.read_excel('fei_dataset.xlsx')  # Open dataset with clinical variables
pheno = pheno.iloc[:388, :]  # Reshape 
pheno.loc[:, 'sex'] = pheno['sex'].replace({ 1: 'Male', 2: 'Female'})
pheno.loc[:, 'group'] = pheno['group'].replace({0: 'Controls', 1: 'ASD', 2: 'Relatives'})

df_pheno_beta_ROI = pd.merge(df_beta, pheno, on='subject')
df_pheno_beta_ROI = df_pheno_beta_ROI.drop_duplicates(subset='subject', keep='first')
#df_pheno_beta_ROI.to_csv('/Users/julie/OneDrive/..../Dataset_pheno_beta_ROI.csv', index=False)


df_pheno_gamma_ROI = pd.merge(df_gamma, pheno, on='subject')
df_pheno_gamma_ROI = df_pheno_gamma_ROI.drop_duplicates(subset='subject', keep='first')
#df_pheno_gamma_ROI.to_csv('/Users/julie/OneDrive/..../Dataset_pheno_gamma_ROI.csv', index=False)

#------------------------------------------------------------------------------------------------------------------------
columns_to_convert = ['Frontal', 'Central', 'Occipital', 'Parietal', 'Temporal_Right', 'Temporal_Left']
df_pheno_beta_ROI[columns_to_convert] = df_pheno_beta_ROI[columns_to_convert].astype(float)

#-------------------------------------------------------------------------------------------------------------------------
# Define hypo and hyper column indices
hypo_columns_indices = [2, 15, 16, 17, 18, 19, 20, 21, 23, 26, 28, 29, 30, 31, 32, 33]
hyper_columns_indices = [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 22, 24, 25, 34, 35, 36, 37, 38]

# Define frequency bands and DataFrames
frequency_bands = {'beta': [14, 30], 'gamma': [30, 49]}
dataframes = {'beta': df_pheno_beta_ROI, 'gamma': df_pheno_gamma_ROI}

# Process each frequency band
for band, indices in frequency_bands.items():
    hypo_columns_band = [f'DUNN{col}' for col in hypo_columns_indices]
    hyper_columns_band = [f'DUNN{col}' for col in hyper_columns_indices]
    
    dataframes[band][hypo_columns_band] = dataframes[band][hypo_columns_band].apply(pd.to_numeric, errors='coerce')
    dataframes[band][hyper_columns_band] = dataframes[band][hyper_columns_band].apply(pd.to_numeric, errors='coerce')
    
    # Définir le mapping d'inversion des valeurs
    inverse_mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    
    # Appliquer l'inversion aux colonnes hypo et hyper
    dataframes[band][hypo_columns_band] = dataframes[band][hypo_columns_band].replace(inverse_mapping)
    dataframes[band][hyper_columns_band] = dataframes[band][hyper_columns_band].replace(inverse_mapping)
    
    # Process and compute dSSP score for the current band
    dataframes[band]['hypo'] = dataframes[band][hypo_columns_band].mean(axis=1)
    dataframes[band]['hyper'] = dataframes[band][hyper_columns_band].mean(axis=1)

    dataframes[band]['hyper'] = dataframes[band]['hyper'].replace(0, np.nan)
    dataframes[band]['hypo'] = dataframes[band]['hypo'].replace(0, np.nan)

    dataframes[band]['dSSP'] = dataframes[band]['hypo'] / dataframes[band]['hyper']

    mean_dSSP_band = dataframes[band]['dSSP'].mean()
    std_dSSP_band = dataframes[band]['dSSP'].std()

    dataframes[band]['dSSP'] = (dataframes[band]['dSSP'] - mean_dSSP_band) / std_dSSP_band


df_pheno_beta_ROI.loc[:, 'sex'] = df_pheno_beta_ROI['sex'].replace({'Male':0 , 'Female':1})
df_pheno_gamma_ROI.loc[:, 'sex'] = df_pheno_gamma_ROI['sex'].replace({'Male':0 , 'Female':1})


# For BETA = numeric values
df_hB = df_pheno_beta_ROI[df_pheno_beta_ROI['group'] == 'ASD']
df_hB = df_hB.applymap(pd.to_numeric, errors='coerce')  # Converts everything to numeric, replaces non-convertible values with NaN

# For GAMMA = numeric values
df_hG = df_pheno_gamma_ROI[df_pheno_gamma_ROI['group'] == 'ASD']
df_hG = df_hG.applymap(pd.to_numeric, errors='coerce')  # Same


#-----------------------------------------------keep subjects with ADI and ADOS----------------------------------------------------------------

df_hB_filtered = df_hB[
    pd.to_numeric(df_hB['ados_css'], errors='coerce').notna() &
    pd.to_numeric(df_hB['adi_crr'], errors='coerce').notna()
]

# Idem pour GAMMA
df_hG_filtered = df_hG[
    pd.to_numeric(df_hG['ados_css'], errors='coerce').notna() &
    pd.to_numeric(df_hG['adi_crr'], errors='coerce').notna()
]

df_hB.to_excel('/volatile/home/jc278357/Documents/....Dataset_pheno_beta_ROI_127TSA.xlsx',index=True)
df_hG.to_excel('/volatile/home/jc278357/Documents/..../Dataset_pheno_gamma_ROI_127TSA.xlsx',index=True)


#%%------------------------------------------------- Multiple Linear Regression ------------------------------------------------------

#Change the “hypo” variable by the other clinical variables (adi-crr...)

import pandas as pd
import statsmodels.api as sm

def run_Lm_analysis(df, output_filename):
    regions = ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal_Left', 'Temporal_Right']
    results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope_adi_crr', 'Slope_age', 'Slope_sex', 
                                    'p-value_adi_crr', 'p-value_age', 'p-value_sex', 'R-squared', 'AIC'])
    
    for region in regions:
        # Définir les variables indépendantes
        x = df[['adi_crr', 'age_years', 'sex']]  # Inclure hypo, âge et sexe
        x = sm.add_constant(x)  # Ajouter une constante pour l'intercept
        
        y = df[region]
        
        model_multiple = sm.OLS(y, x, missing='drop')
        results_summary_multiple = model_multiple.fit()
        aic_multiple = results_summary_multiple.aic
        
        new_row = pd.DataFrame({
            'Region': [region],
            'Model': ['Multiple'],
            'Intercept': [results_summary_multiple.params[0]],
            'Slope_adi_crr': [results_summary_multiple.params[1]],
            'Slope_age': [results_summary_multiple.params[2]],
            'Slope_sex': [results_summary_multiple.params[3]],
            'p-value_adi_crr': [results_summary_multiple.pvalues[1]],
            'p-value_age': [results_summary_multiple.pvalues[2]],
            'p-value_sex': [results_summary_multiple.pvalues[3]],
            'R-squared': [results_summary_multiple.rsquared],
            'AIC': [aic_multiple]
        })
        
        results = pd.concat([results, new_row], ignore_index=True)

    results.to_excel(output_filename, index=False)

run_Lm_analysis(df_hB, '/volatile/home/jc278357/Documents/Analyse EEG/résultats_127TSA/Résultats_SSP_inverse/Lm_ASD_beta_adi_crr_age_sex.xlsx')
run_Lm_analysis(df_hG, '/volatile/home/jc278357/Documents/Analyse EEG/résultats_127TSA/Résultats_SSP_inverse/Lm_ASD_gamma_adi_crr_age_sex.xlsx')


# FDR correction
pvals = [.20, .002, .64, .16, .20, .66]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)

# %%---------------------------------------------------------- Compute Cohen's f² ----------------------------------------------------
def cohen_f2(r_squared):
    """
    Computes Cohen's f² effect size from R².
    
    :param r_squared: The model's coefficient of determination (R²)
    :return: Cohen's f² effect size
    """
    if r_squared == 1:
        return float('inf')  # Avoid division by zero
    return r_squared / (1 - r_squared)

# Example usage
r2 = 0.15  # Replace with your actual R² value
effect_size = cohen_f2(r2)
print(f"Cohen's f² effect size: {effect_size:.3f}")

# Interpretation
if effect_size < 0.02:
    print("Small effect")
elif effect_size < 0.15:
    print("Moderate effect")
else:
    print("Large effect")
    
# %%----------------------------------------------------- PLOT linear regression -----------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_linear_regression(df, x_col, y_col):

    sns.lmplot(x=x_col, y=y_col, data=df, aspect=1.5, height=5, ci=95, line_kws={'color': 'black'}, scatter_kws={'color': 'black'})
    
    # Add labels and title
    plt.title("Right Temporal", fontweight='bold', fontsize=17)
    plt.xlabel('ADI-R C Score', fontsize=14)
    plt.ylabel('PSD gamma (μV²/Hz)', fontsize=14)
    
    # Show plot
    plt.show()

plot_linear_regression(df_hG, x_col='adi_crr', y_col='Temporal_Right')

# %%----------------------------------------------------- Plot effect size ------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# Define brain regions
#regions = ['Frontal', 'Central', 'Parietal', 'Occipital', 'Right Temporal', 'Left Temporal']

# Effect size f² values (replace with your actual data)
#f2_values = [0.14, 0.35, 0.20, 0.12, 0.22, 0.20]

# Define brain regions
regions = ['Frontal', 'Central', 'Parietal', 'Left Temporal']

# Effect size f² values (replace with your actual data)
f2_values = [0.09, 0.16, 0.12, 0.15]

# Create the plot
plt.figure(figsize=(8, 5))

# Add colored zones for thresholds defined by your criteria
plt.axvspan(0.02, 0.15, color='yellowgreen', alpha=0.5, label='Small effect (f² ≥ 0.02)')
plt.axvspan(0.15, 0.35, color='moccasin', alpha=0.5, label='Medium effect (f² ≥ 0.15)')
plt.axvspan(0.35, max(f2_values) + 0.1, color='sandybrown', alpha=0.5, label='Large effect (f² ≥ 0.35)')

# Plot effect sizes
plt.scatter(f2_values, range(len(regions)), color='black')

# Add a dashed line from the legend to each point
for i, region in enumerate(regions):
    # Coordinates of the point for each region (x, y)
    x = f2_values[i]
    y = i
    # Add the dashed line
    plt.plot([0, x], [y, y], 'k--', lw=1)  # Black dashed line

# Customize the plot
plt.yticks(range(len(regions)), regions, fontsize=14)
plt.xlabel('Effect Size (f²)', fontsize=14)

# Display the legend
plt.legend(loc='lower right')

# Add a grid to improve readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------------------------------

#test MannWitney for sex effect

#BETA
from scipy.stats import mannwhitneyu

regions = ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal_Left', 'Temporal_Right']

for region in regions:

    puissance_beta_hommes = df_pheno_beta_ROI[df_pheno_beta_ROI['sex'] == 0][region].dropna()
    puissance_beta_femmes = df_pheno_beta_ROI[df_pheno_beta_ROI['sex'] == 1][region].dropna()


    stat, p_value = mannwhitneyu(puissance_beta_hommes, puissance_beta_femmes)

    # Affichage des résultats
    print(f"Région : {region}")
    print(f"  Statistique U = {stat:.3f}, p-value = {round(p_value, 3)}")
    print("-" * 40)

#-------------------------plot---------------------------------------------------------------------------------------

df_pheno_beta_ROI.loc[:, 'sex'] = df_pheno_beta_ROI['sex'].replace({0:'Male' , 1:'Female'})
df_pheno_gamma_ROI.loc[:, 'sex'] = df_pheno_gamma_ROI['sex'].replace({0:'Male' , 1:'Female'})

sns.pointplot(x='sex', y='Central', color='black', data=df_pheno_gamma_ROI)
plt.ylabel('Central PSD gamma (μV²/Hz)', fontsize=12)
plt.show()



