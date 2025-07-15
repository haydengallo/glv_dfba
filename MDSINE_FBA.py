import numpy as np
#from dfba import DfbaModel, ExchangeFlux, KineticVariable
import cobra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from numba import njit
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sys
import os
import openpyxl
import gurobipy

import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
import arviz as az

import time
import joblib
import multiprocessing
from scipy.stats import truncnorm
import copy
from pathlib import Path
import subprocess

### script for running glv_dfba inference
from helper_functions import *
import argparse
import re

from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import json

from Bio import Entrez

### Need to activate cobra_agorra conda environment ###

from datetime import datetime
start_time = datetime.now()

################################
### Manual Parameters to set ###
################################

subject_dict = {1953:1948, 1507:1510, 1999:2000}

for key in subject_dict.keys():

    ### subject with abundance data
    subject_to_plot = key#1953#1507#1999
    ### subject to predict, subject with metabolomics data
    subject_to_predict = subject_dict[key]#1948#1510#2000
    ### Set test num
    test_num = 60
    ### set time scaler
    time_scaler = 24
    ### Scaling factor
    scal_fact = 1e12#9.220114e10
    ### Total time steps
    total_time_steps = 415
    ### Simulation notes
    notes = 'just hourly timesteps i guess'



    ## ok now need to reconstruct the time series basically 

    ### need to load in all of the processed_data 
    # Make the data and validation Study objects

    processed_data = Path('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun')

    tsv_files = sorted(processed_data.glob('*.tsv'))
    tsv_files = {f.stem : f for f in tsv_files}

    counts = pd.read_csv(tsv_files['counts'], delimiter='\t', index_col=0)
    metadata = pd.read_csv(tsv_files['metadata'], delimiter='\t', index_col=0)
    perturbations = pd.read_csv(tsv_files['perturbations'], delimiter='\t', index_col=0)
    qpcr = pd.read_csv(tsv_files['qpcr'], delimiter='\t', index_col=0)
    taxonomy = pd.read_csv(tsv_files['taxonomy'], delimiter='\t', index_col=0)


    ### Load up the initial data for subject 1948 with RC diet 

    #init_mets_input_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/sub_' + str(subject_to_predict) + '_init_met.csv'

    #sub_1948_init_mets = pd.read_csv(init_mets_input_path, header=None)
    init_mets_input_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/sub_' + str(subject_to_predict) + '_met_df.tsv'

    sub_1948_init_mets = pd.read_csv(init_mets_input_path,sep='\t')
    sub_1948_init_mets.head()


    ### Ok need to take the sub_1948_init_mets and prepare for input to static_dfba

    sub_1948_init_mets.columns = ['reaction','fluxValue']

    ## change fluxValue to positive and then only keep reaction and fluxvalue columns 

    sub_1948_init_mets = sub_1948_init_mets[['reaction', 'fluxValue']]

    sub_1948_init_mets['fluxValue'] =  (np.double(sub_1948_init_mets['fluxValue']))
    #sub_1948_init_mets.iloc[0,0] = 'EX_12kltchca(e)'

    ### Also need to change out EX_ocdca(e) and EX_erythritol(e)

    ### old 

    #sub_1948_init_mets['reaction'].loc['EX_cis_Oleic acid(e)']


    sub_1948_init_mets['fluxValue'] = sub_1948_init_mets['fluxValue']


    ### Load in diet data that will be applied in intervals 

    RC_diet = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/RC_mm_add.csv', header=None)
    RC_diet.columns = ['reaction','fluxValue','upper_bound']
    RC_diet = RC_diet[['reaction', 'fluxValue']]
    RC_diet['fluxValue'] =  -1.0*(np.double(RC_diet['fluxValue']))

    ## Multiply by 5g over 12 hours
    RC_diet['fluxValue'] = ((5/time_scaler)*RC_diet['fluxValue'])
    RC_diet.head()

    ### Load in metabolomics data 

    metabolomics_data = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/raw_data/complete_mapping_metabolomics.csv', index_col=0)
    ## fill NAs w/ zero 
    metabolomics_data = metabolomics_data.fillna(0)


    ### load in metabolomics metadata

    metabolomics_metadata_raw = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/raw_data/metabolomics_meta.csv', index_col=0)
    # filtered metabolomics metadata for subject 1948 initial sample 
    metabolomics_metadata = metabolomics_metadata_raw.copy()
    metabolomics_metadata = metabolomics_metadata[(metabolomics_metadata['Mouse'] == float(subject_to_predict)) & (metabolomics_metadata['Rec_day_adj'] == -3)]
    metabolomics_metadata.index.tolist()


    ### load in sample metadata 
    sample_metadata = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/metadata.tsv', sep='\t',index_col=0)




    ### load in refseq to agora dataframe


    #refseq_to_agora_df = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/refseq_to_agora_df_all_cohorts.tsv', delimiter='\t', index_col=0)
    refseq_to_agora_df = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/refseq_to_agora_update_07_07_25.csv', index_col=0)


    ### since i am going with daily resolution need to load in that data for determining initial abundances and the rates of change 

    # nevermind bihourly after i figured out the smoothing
    MDSINE_filter_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/filtering_hourly_resolution/Subject_' + str(subject_to_plot) + '/mean_smoothed.tsv'
    bi_hourly_resolution_latent_traj = pd.read_csv(MDSINE_filter_path, delimiter='\t', index_col=0)
    bi_hourly_resolution_latent_traj


    ### filter metabolomics data by initial sample

    metabolomics_data_initial_sub_1948 = metabolomics_data[metabolomics_data['SampleName'] == metabolomics_metadata.index.tolist()[0]]
    #init_metabolite_sample


    rc_diet_data = metabolomics_data[metabolomics_data['SampleName'] == 'RC_001']

    rc_diet_MS_convert = rc_diet_data.copy()
    rc_diet_data


    with open('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/raw_data/BiGG_to_MSID.json') as f:
        bigg_to_modelseed = json.load(f)
        #print(bigg_to_modelseed)

    cmpd_names = []
    for i in rc_diet_data.columns.tolist()[1:]:
        cmpd_names.append(bigg_to_modelseed[i])


    rc_diet_MS_convert = rc_diet_MS_convert.drop(columns=['SampleName'])
    rc_diet_MS_convert.columns = cmpd_names
    rc_diet_MS_convert = rc_diet_MS_convert.T
    rc_diet_MS_convert

    rc_diet_MS_convert['maxflux'] = 100
    rc_diet_MS_convert = rc_diet_MS_convert.reset_index()
    rc_diet_MS_convert.columns =['compounds','maxflux','minflux']
    rc_diet_MS_convert['minflux'] = rc_diet_MS_convert['minflux']*-1.0
    #rc_diet_MS_convert['minflux'] = -25
    rc_diet_MS_convert

    rc_diet_MS_convert = rc_diet_MS_convert[rc_diet_MS_convert['maxflux'] != 0.0]


    def bigg_to_agora_exchange_ids(bigg_ids):
        """
        Convert a list of BiGG metabolite IDs to AGORA-style COBRApy exchange IDs.

        Handles known mismatches between BiGG names and AGORA metabolite IDs.

        Parameters:
        - bigg_ids: list of BiGG metabolite names (e.g. ['glc__D', '12_Ketolithocholic acid'])

        Returns:
        - List of AGORA-style exchange reaction IDs: ['EX_glc_D(e)', 'EX_12kltchca(e)', ...]
        """

        # Dictionary of known non-standard mappings (keys normalized)
        special_cases = {
            '12_ketolithocholic_acid': '12kltchca',
            'cis_oleic_acid': 'ocdca',
            'meso_erythritol': 'erythritol',
            'hc02191': 'hc02191',         # keep as-is but handle case
            'c10164': 'c10164',           # same
            'lnlacp': 'lnlcacp',          # typo fix? depends on your model
            '4hpro_lt': '4hprolt',
        }

        exchange_ids = []
        for met in bigg_ids:
            # Normalize: replace '__' with '_', convert to lowercase, and replace spaces
            norm_met = met.replace('__', '_').replace(' ', '_').lower()

            # Use special mapping if present
            agora_id = special_cases.get(norm_met, norm_met)

            exchange_ids.append(f'EX_{agora_id}(e)')

        return exchange_ids



    agora_ex_ids_list = bigg_to_agora_exchange_ids(rc_diet_data.set_index('SampleName').columns.tolist())

    ### Making dict of initial conditions for 1948 and RC dietary conditions too 

    sub_1948_met_dict = dict(zip(agora_ex_ids_list, metabolomics_data_initial_sub_1948.set_index('SampleName').iloc[0,:].tolist()))
    RC_diet_met_dict = dict(zip(agora_ex_ids_list, rc_diet_data.set_index('SampleName').iloc[0,:].tolist()))


    ### Convert these dicts to dfs for saving and using for analysis in matrix 

    RC_diet_met_df = pd.DataFrame.from_dict(RC_diet_met_dict, orient='index').reset_index()
    RC_diet_met_df.columns = ['metabolite', 'fluxValue']

    sub_1948_met_df = pd.DataFrame.from_dict(sub_1948_met_dict, orient='index').reset_index()
    sub_1948_met_df.columns = ['metabolite', 'fluxValue']


    for i in range(0, len(bi_hourly_resolution_latent_traj.columns)-1):

        if i == 0:
            base_array = np.linspace(bi_hourly_resolution_latent_traj.iloc[:,i], bi_hourly_resolution_latent_traj.iloc[:,i+1], 2)
        else:
            temp_array = np.linspace(bi_hourly_resolution_latent_traj.iloc[:,i], bi_hourly_resolution_latent_traj.iloc[:,i+1], 2)[1:,:]
            base_array = np.concatenate((base_array, temp_array))
    


    abun_df = pd.DataFrame(base_array.T, index=bi_hourly_resolution_latent_traj.index)
    abun_df

    rate_array = np.zeros((15,abun_df.shape[1]-1))


    for i in range(0,(len(abun_df.columns)-1)):

        rate_array[:,i] = (abun_df.iloc[:,i+1]/abun_df.iloc[:,i])-1


    rate_df = pd.DataFrame(rate_array)

    rate_df.index = bi_hourly_resolution_latent_traj.index
    #test_df.index = bi_hourly_resolution_latent_traj.index

    rate_df


    ### load in the taxonomy data that shows which asv is which species

    data_dir = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun'


    taxonomy_path = data_dir + '/taxonomy.tsv'
    RC_taxonomy = pd.read_csv(taxonomy_path, delimiter='\t', index_col=0)
    RC_taxonomy



    ### load the cobra models into memory i guess

    cobra_models_dir = Path('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/test_draft_reconstructions_07_08_25')

    # Make the data and validation Study objects
    cobra_models = sorted(cobra_models_dir.glob('*.sbml'))
    cobra_models = {f.stem : f for f in cobra_models}

    ### Just loading the models needed in 

    loaded_models = {}

    #count = 0

    for key in cobra_models:
        #if count == 1:
        #    break
        #print(key.split('_'))
        model_name = key.split('_')[0] + '_' + key.split('_')[1]
        print(model_name)
        model = cobra.io.read_sbml_model(cobra_models[key])
        loaded_models[model_name] = model
        #count+=1

    # Adding correct names to refseq_to_agora_df

    temp_name_list = []


    for i in range(0, len(refseq_to_agora_df)):
        #print(refseq_to_agora_df['Agora_Name'].iloc[i].split('_'))
        temp_name = refseq_to_agora_df['Agora_Name'].iloc[i].replace('.',"_")
        print(temp_name)
        temp_name_list.append(temp_name.split('_')[0] + '_' + temp_name.split('_')[1])

    refseq_to_agora_df['Model_Names'] = temp_name_list


    ### Manual renaming of two lactobacilllus species ###
    #refseq_to_agora_df.loc['320dfd16200daaf2b0503975d4e68fd5','Model_Names'] = 'Lactobacillus_reuteri'
    #refseq_to_agora_df.loc['94e30534f622e456a683abe4e60fc214','Model_Names'] = 'Lactobacillus_animalis'
    #refseq_to_agora_df.loc['18673193aa6bf30c6a1e71ac504e04df','Model_Names'] = 'Staphylococcus_equorum'
    #refseq_to_agora_df.loc['18673193aa6bf30c6a1e71ac504e04df','Agora_Name'] = 'Staphylococcus_equorum_subsp_equorum_Mu2'

    refseq_to_agora_df

    ASV_string_to_species_names_dict = dict(zip(refseq_to_agora_df.index.tolist(), refseq_to_agora_df['Model_Names']))

    models_for_FBA = {}

    for i in refseq_to_agora_df.index.tolist():
        models_for_FBA[i] = loaded_models[refseq_to_agora_df.loc[i]['Model_Names']].copy()
        #models_for_FBA[i] = refseq_to_agora_df.loc[i]['Model_Names']


    # dict of model names and models 
    models_for_FBA



    bugs_to_filter = list(models_for_FBA.keys())
    bugs_to_filter

    abun_hr_df_filt = bi_hourly_resolution_latent_traj.reindex(bugs_to_filter)

    #abun_hr_df_filt = abun_hr_df.reindex(bugs_to_filter)
    rate_df_filt = rate_df.reindex(bugs_to_filter)


    ## Load in N_massi diet

    test_diet = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/new_diet_test_s_aureus.csv', header=None)
    test_diet.columns = ['reaction','fluxValue','upper_bound']
    test_diet = test_diet[['reaction', 'fluxValue']]
    test_diet['fluxValue'] =  -1*(np.double(test_diet['fluxValue']))

    ### Setting up the things needed for running static_fba


    ### Model names

    model_names = list(models_for_FBA.keys())#refseq_to_agora_df_filt['Model_Names'].tolist()

    ### Get list of models
    ### filter loaded models 


    loaded_models

    #models_dict_filt = {key: loaded_models[key] for key in model_names}
    #models_list = list(models_dict_filt.values())

    models_list = list(models_for_FBA.values())

    ### Conversion factor for absolute abun to gDW

    abun_hr_df_filt = abun_hr_df_filt/scal_fact


    ### init abun 

    init_abun = abun_hr_df_filt.iloc[:,0].tolist()


    basic_range = list(np.arange(6,18,1))

    feeding_schedule = []

    for i in range(0,17):
        feeding_schedule.append([x+24 * i for x in basic_range])

    feeding_schedule = list(np.arange(50,844,1))

    RC_diet_met_df.columns = ['reaction', 'fluxValue']

    ### filter out acgam(e) from RC_diet for concat with init 

    # Specify the columns to compare
    cols_to_check = ['reaction']
    # Filter df2 to keep only rows that are not in df1 based on selected columns
    filtered_RC_diet = RC_diet[~RC_diet.set_index(cols_to_check).index.isin(sub_1948_init_mets.set_index(cols_to_check).index)]

    # Concatenate
    sub_1948_init_mets = pd.concat([sub_1948_init_mets, filtered_RC_diet], ignore_index=True)
    sub_1948_init_mets
    #RC_diet_filt_out_acgam = RC_diet[RC_diet['reaction'] != 'EX_acgam(e)']


    ### need to add required nutrients from gapfilled RC_mm_media to initial conditions for subject but without overriding the 60 metabolite values we want as initial values 


    print(len(np.unique(sub_1948_init_mets['reaction'].tolist())))
    print(sub_1948_init_mets['reaction'].value_counts())

    ### Translate metabolites in diet and initial conditions back to Kbase nomenclature b/c going to use kbase reconstructions instead of translated models
    rc_diet_MS_convert = rc_diet_MS_convert[['compounds', 'maxflux']]
    rc_diet_MS_convert.columns = ['reaction', 'fluxValue']

    ### manually add 18 minimal metabolites to the RC_diet

    mets_to_add = ['cpd00001','cpd00009','cpd00013','cpd00030','cpd00034','cpd00048','cpd00058','cpd00063','cpd00067','cpd00099','cpd00149','cpd00205','cpd00244','cpd00254','cpd00971','cpd10515','cpd10516','cpd11574','cpd00028']
    flux = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


    mets_to_add_df = pd.DataFrame(mets_to_add, flux)
    mets_to_add_df.reset_index(inplace=True)
    mets_to_add_df.columns = ['fluxValue','reaction']
    mets_to_add_df

    rc_diet_MS_convert = pd.concat([rc_diet_MS_convert, mets_to_add_df])

    ### change over init values to correct kbase nomenclature 

    metabolomics_data_initial_sub_1948.iloc[:,1:].T.index.tolist()

    cmpd_names = []
    for i in metabolomics_data_initial_sub_1948.iloc[:,1:].T.index.tolist():
        cmpd_names.append(bigg_to_modelseed[i])

    metabolomics_data_initial_sub_1948 = metabolomics_data_initial_sub_1948.iloc[:,1:].T

    metabolomics_data_initial_sub_1948.index = cmpd_names
    metabolomics_data_initial_sub_1948.reset_index(inplace=True)
    metabolomics_data_initial_sub_1948.columns = ['reaction', 'fluxValue']

    metabolomics_data_initial_sub_1948

    for i in range(0, len(metabolomics_data_initial_sub_1948)):
        metabolomics_data_initial_sub_1948['reaction'].iloc[i] = 'EX_' + metabolomics_data_initial_sub_1948['reaction'].iloc[i]  + '_b'

    for i in range(0, len(rc_diet_MS_convert)):
        rc_diet_MS_convert['reaction'].iloc[i]  = 'EX_' + rc_diet_MS_convert['reaction'].iloc[i]  + '_b'

    diet_scaler = (5/time_scaler)

    rc_diet_MS_convert['fluxValue'] = (diet_scaler*rc_diet_MS_convert['fluxValue'])

    ### Add adjusted RC diet data 

    ### Read in adjusted diet data from mouse GEM mets_to_add_df

    RC_diet_adjusted = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/RC_diet_adjust_mouse_GEM.csv')
    RC_diet_adjusted

    #### Add necessary extra metabolites for bacterial growth 

    mets_to_add = ['EX_cpd00001_b','EX_cpd00009_b','EX_cpd00013_b','EX_cpd00030_b','EX_cpd00034_b','EX_cpd00048_b','EX_cpd00058_b','EX_cpd00063_b','EX_cpd00067_b','EX_cpd00099_b','EX_cpd00149_b','EX_cpd00205_b','EX_cpd00244_b','EX_cpd00254_b','EX_cpd00971_b','EX_cpd10515_b','EX_cpd10516_b','EX_cpd11574_b','EX_cpd00028_b']
    #flux = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    flux = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]


    mets_to_add_df = pd.DataFrame(mets_to_add, flux)
    mets_to_add_df.reset_index(inplace=True)
    mets_to_add_df.columns = ['fluxValue','reaction']
    mets_to_add_df



    RC_diet_adjusted = pd.concat([RC_diet_adjusted, mets_to_add_df])
    RC_diet_adjusted

    RC_diet_adjusted['fluxValue'] = (diet_scaler*RC_diet_adjusted['fluxValue'])


    met_pool_over_time, model_abun_dict = static_dfba(list_model_names=model_names, list_models=models_list, initial_abundance=init_abun, total_sim_time=total_time_steps, num_t_steps=total_time_steps, glv_out=np.array(abun_hr_df_filt.T), glv_params=None, environ_cond=metabolomics_data_initial_sub_1948, pfba=True, MDSINE_rates=rate_df_filt, Diet=RC_diet_adjusted, time_points_feed = feeding_schedule, time_scaler=time_scaler)

    ### Save the results
    output_folder = 'filtering_hourly_resolution'

    plot_dir_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/MDSINE_vs_Exp_plots_' + output_folder + '/test_' + str(test_num) + '/Subject_' + str(subject_to_plot)

    plot_dir = Path(plot_dir_path)
    os.makedirs(plot_dir, exist_ok=True)


    met_save = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_met_pool.npy'
    abun_save = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_abun_save.npy'

    np.save(met_save, met_pool_over_time)
    np.save(abun_save, model_abun_dict)

    #met_pool_over_time = np.load(met_save, allow_pickle=True)
    #model_abun_dict = np.load(abun_save, allow_pickle=True)

    param_save = {'Sub_w_abun_data':subject_to_plot, 'Sub_w_met_data': subject_to_predict, 'Sim_num': test_num, 'time_scaler':time_scaler, 'Scaling_factor':scal_fact, 'Total_time_steps': total_time_steps, 'Diet_scaler':diet_scaler, 'notes':notes}
    param_save_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_params.txt'
    with open(param_save_file_name, 'w') as file:
        file.write(json.dumps(param_save))

    ### Ok now plot bacterial abundance overtime from the FBA output in same order as MDSINE plots 

    ### ok basically remake the stacked plots from MDSINE_FBA_output_analysis.ipynb and add third column for the FBA output abundance data

    ### make empty array for storing FBA abun data for each species and then convert to df for easy plotting 

    FBA_biomass = np.zeros([len(model_abun_dict.keys()), len(model_abun_dict['14ac4eaad5b4e2ff3c071832e0fd4229']['fba_biomass'])])
    FBA_biomass

    # %%
    ### Convert FBA abun output to relative abundance 
    count = 0
    for key in model_abun_dict:
        FBA_biomass[count,:] = model_abun_dict[key]['fba_biomass']
        count+=1

    FBA_biomass_df = pd.DataFrame(FBA_biomass)
    FBA_biomass_df.index = model_abun_dict.keys()

    index_to_filter_by = FBA_biomass_df.index

    ## Tranform to relative abun

    FBA_biomass_df = FBA_biomass_df/FBA_biomass_df.sum(axis=0)

    ### Need to just add the missing bug for now to the FBA output 
    FBA_biomass_df = FBA_biomass_df.T
    FBA_biomass_df['9cf5cb71450a2aa080ff905f89b0a624'] = 0
    FBA_biomass_df = FBA_biomass_df.T

    FBA_biomass_df = FBA_biomass_df.melt(ignore_index=False)
    FBA_biomass_df = FBA_biomass_df.reset_index()
    FBA_biomass_df.columns = ['FeatureID','time', 'count']
    FBA_biomass_df['time'] = (FBA_biomass_df['time']/time_scaler)-3
    FBA_biomass_df

    # %%
    FBA_biomass_df_plot = pd.DataFrame(FBA_biomass)
    FBA_biomass_df_plot.index = model_abun_dict.keys()

    FBA_biomass_df_plot = FBA_biomass_df_plot.melt(ignore_index=False)
    FBA_biomass_df_plot = FBA_biomass_df_plot.reset_index()
    FBA_biomass_df_plot.columns = ['FeatureID','time', 'count']
    FBA_biomass_df_plot['time'] = (FBA_biomass_df_plot['time']/time_scaler)-3
    FBA_biomass_df_plot['count'] = FBA_biomass_df_plot['count']*scal_fact

    # %%
    output_folder = 'filtering_hourly_resolution'

    #plot_dir_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/MDSINE_vs_Exp_plots_' + output_folder + '/test_' + str(test_num)

    plot_dir = Path(plot_dir_path)
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(figsize= (15,10))
    sns.lineplot(data=FBA_biomass_df_plot, x='time', y='count', hue = 'FeatureID')
    plt.yscale('log')
    plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_abundances_over_time_test_' + str(test_num) + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    #plt.show()

    # %%
    FBA_biomass_df_plot

    FBA_biomass_df_plot_unstack = FBA_biomass_df_plot.pivot(index='FeatureID', columns ='time')['count']
    FBA_biomass_df_plot_unstack = pd.DataFrame(FBA_biomass_df_plot_unstack.sum(axis=0)).reset_index()
    FBA_biomass_df_plot_unstack.columns = ['time', 'abun']
    FBA_biomass_df_plot_unstack['time'] = pd.to_numeric(FBA_biomass_df_plot_unstack['time'])
    FBA_biomass_df_plot_unstack['time'] = pd.to_numeric(FBA_biomass_df_plot_unstack['time'])

    # %%
    ### Need to plot metabolite trajectories too

    met_pool_over_time_df = pd.DataFrame(met_pool_over_time)
    met_pool_over_time_df = met_pool_over_time_df.fillna(0)
    met_pool_over_time_df_melt= met_pool_over_time_df.melt(ignore_index=False)
    met_pool_over_time_df_melt = met_pool_over_time_df_melt.reset_index()
    met_pool_over_time_df_melt.columns = ['Time','Metabolite', 'Concentration']
    met_pool_over_time_df_melt

    # %%
    met_pool_over_time_df

    # %%
    met_pool_over_time_df_melt[met_pool_over_time_df_melt['Concentration'] < 0]

    # %%
    met_pool_over_time_df_melt[met_pool_over_time_df_melt['Metabolite'] == 'EX_adn(e)']

    # %%
    total_abun_MDSINE = pd.DataFrame(bi_hourly_resolution_latent_traj.sum(axis=0)).reset_index()
    total_abun_MDSINE.columns = ['time', 'abun']
    total_abun_MDSINE['time'] = pd.to_numeric(total_abun_MDSINE['time'])
    total_abun_MDSINE.head()

    # %%
    fig, axs = plt.subplots(figsize= (15,10))
    sns.lineplot(data=total_abun_MDSINE, x='time', y='abun')
    plt.yscale('log')

    # %%
    output_folder = 'filtering_hourly_resolution'

    #plot_dir_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/MDSINE_vs_Exp_plots_' + output_folder + '/test_' + str(test_num)

    plot_dir = Path(plot_dir_path)
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(figsize= (15,10))
    sns.lineplot(data=met_pool_over_time_df_melt, x='Time', y='Concentration', hue = 'Metabolite')
    plt.yscale('log')
    plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_metabolites_over_time_test_' + str(test_num) + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    #plt.show()


    # %%
    ### Stacked plots 

    ## Directory to save plots to 


    output_folder = 'filtering_hourly_resolution'

    #plot_dir_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/MDSINE_vs_Exp_plots_' + output_folder + '/test_' + str(test_num)

    plot_dir = Path(plot_dir_path)
    os.makedirs(plot_dir, exist_ok=True)

    samps_for_sub ={}


    temp = metadata[metadata['subject'] == subject_to_plot]
    testing = counts.filter(temp.index.tolist(), axis=1)
    print(testing)
    testing = testing.loc[~(testing==0).all(axis=1)]
    qpcr_measurements = qpcr.reindex(testing.columns)
    qpcr_measurements = qpcr_measurements['measurement1'].to_list()
    testing_ra = testing.reindex(index_to_filter_by)
    testing_ra = testing/testing.sum(axis=0)
    testing_abs_abun = testing_ra*qpcr_measurements
    time_dict = dict(zip(temp.index, temp['time'].tolist()))
    testing_melt_ra = testing_ra.melt(ignore_index=False)
    testing_melt_ra.columns = ['sample', 'count']
    testing_melt_ra['time'] = testing_melt_ra['sample'].map(time_dict)
    #print(testing_melt_ra.head())
    testing_melt_abs_abun = testing_abs_abun.melt(ignore_index=False)
    testing_melt_abs_abun.columns = ['sample', 'count']
    testing_melt_abs_abun['time'] = testing_melt_abs_abun['sample'].map(time_dict)

    # load in MDSINE output for each subject 
    output_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/processed_data_filtered_RC_all_cohorts_corrected_abs_abun/' + output_folder + '/Subject_' + str(subject_to_plot) + '/mean_smoothed.tsv'
    MDSINE_output = pd.read_csv(output_path, delimiter='\t', index_col=0)


    MDSINE_output = MDSINE_output.reindex(index_to_filter_by)

    MDSINE_output = MDSINE_output/MDSINE_output.sum(axis=0)
    MDSINE_output = MDSINE_output.melt(ignore_index=False)
    MDSINE_output = MDSINE_output.reset_index()
    MDSINE_output.columns = ['FeatureID','time', 'count']


    ###################
    ### Second plot ###
    ###################

    fig, (ax_top, ax_middle, ax_bottom) = plt.subplots(
    3, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]}
    )

    # Ensure time is numeric
    MDSINE_output['time'] = pd.to_numeric(MDSINE_output['time'])
    testing_melt_ra['time'] = pd.to_numeric(testing_melt_ra['time'])
    FBA_biomass_df['time'] = pd.to_numeric(FBA_biomass_df['time'])

    # Define stacking order
    feature_order = MDSINE_output['FeatureID'].value_counts().index.tolist()
    reversed_order = feature_order[::-1]  # For stackplot

    # Assign consistent colors


    twentysiz = [
        "#690f19", "#b80000", "#d63220", "#d64e20", "#c1693a",  # reds/oranges
        "#d38838", "#AE7219", "#C09038", "#d1ad57", "#d8c33a", "#f1e149",  # orange/yellows
        "#d2e626", "#a0c618", "#7dc119", "#40a903", "#0c9515",  # yellow/green
        "#1da05a", "#1da18d", "#1d91a1", "#1d6ea1", "#1d31a1",  # green/blue
        "#1a1a87", "#5b5bc9", "#8d8df1", "#8670cc", "#9870cc",  # blue/purple
        "#000000"  # black
    ]
    #sns.set_style("dark")  # or "whitegrid", "dark", etc.
    #sns.set_context("notebook")  # or "paper", "talk", "poster"
    sns.set_palette(twentysiz)

    #palette = sns.color_palette("Spectral", n_colors=len(feature_order))  # or use 'husl', 'Set2', etc.
    color_map = dict(zip(feature_order, twentysiz))

    # Pivot for stackplot in reversed stacking order
    continuous_pivot_MDSINE = (
        MDSINE_output
        .pivot(index='time', columns='FeatureID', values='count')
        .fillna(0)
    )[reversed_order]

    # Pivot for stackplot in reversed stacking order
    continuous_pivot_FBA = (
        FBA_biomass_df
        .pivot(index='time', columns='FeatureID', values='count')
        .fillna(0)
    )[reversed_order]



    # Set categorical order for histogram
    testing_melt_ra['FeatureID'] = pd.Categorical(testing_melt_ra.reset_index()['FeatureID'], categories=feature_order, ordered=True)

    # Top: histogram
    sns.histplot(
        data=testing_melt_ra,
        x='time',
        weights='count',
        hue='FeatureID',
        multiple='stack',
        ax=ax_top,
        alpha=0.8,
        binwidth=1,
        palette=color_map
    )
    ax_top.legend_.remove()
    ax_top.set_ylabel('Relative Abundance')
    ax_top.set_title('Experimental Abundance')
    #ax_top.legend(title='FeatureID', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Middle: stackplot using reversed order and matching colors
    ax_middle.stackplot(
        continuous_pivot_MDSINE.index,
        *[continuous_pivot_MDSINE[col] for col in continuous_pivot_MDSINE.columns],
        alpha=0.8,
        colors=[color_map[feat] for feat in reversed_order],
        labels=reversed_order
    )
    ax_middle.set_ylabel('Relative Abundance')
    #ax_middle.set_xlabel('Time')
    ax_middle.set_title('Trajectories from MDSINE')

    # Bottom: stackplot using FBA output 

    # Bottom: stackplot using reversed order and matching colors
    ax_bottom.stackplot(
        continuous_pivot_FBA.index,
        *[continuous_pivot_FBA[col] for col in continuous_pivot_FBA.columns],
        alpha=0.8,
        colors=[color_map[feat] for feat in reversed_order],
        labels=reversed_order
    )
    ax_bottom.set_ylabel('Relative Abundance')
    ax_bottom.set_xlabel('Time')
    ax_bottom.set_title('Trajectories from FBA')



    # Tidy layout
    ax_top.tick_params(labelbottom=False)
    plt.subplots_adjust(hspace=0.1, right=0.75)


    # Create custom legend handles using your color map
    legend_elements = [
        Patch(facecolor=color_map[feat], label=ASV_string_to_species_names_dict[feat])
        for feat in feature_order
    ]

    # Add custom legend to the *figure* (not either axis), outside plot
    fig.legend(
        handles=legend_elements,
        title='Species',
        loc='center left',
        bbox_to_anchor=(.8, 0.5),  # Push legend just outside right edge
        borderaxespad=0,
        frameon=False,
        ncol=1
    )
    # Adjust spacing to make room for the legend
    plt.subplots_adjust(hspace=0.1, right=1)  # Shrink plot width
    overall_title = 'Subject ' + str(subject_to_plot)
    plt.suptitle(overall_title, x=0.6, fontsize = 20)
    plt.subplots_adjust(right=0.75)
    second_plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_stacked_hist_line_plots_FBA_' + str(test_num) + '.pdf'
    plt.savefig(second_plot_file_name, bbox_inches="tight")
    #plt.show()




    # %%
    testing_melt_abs_abun_unstack = testing_melt_abs_abun.reset_index().pivot(index='FeatureID', columns ='time')['count']
    testing_melt_abs_abun_unstack = pd.DataFrame(testing_melt_abs_abun_unstack.sum(axis=0)).reset_index()
    testing_melt_abs_abun_unstack.columns = ['time', 'abun']
    testing_melt_abs_abun_unstack['time'] = pd.to_numeric(testing_melt_abs_abun_unstack['time'])
    testing_melt_abs_abun_unstack

    # %%
    fig, axs = plt.subplots(figsize= (15,10))
    sns.lineplot(data=testing_melt_abs_abun_unstack, x='time', y='abun')
    sns.lineplot(data=total_abun_MDSINE, x='time', y='abun', color = 'red')
    sns.lineplot(data=FBA_biomass_df_plot_unstack, x='time', y='abun', color = 'blue')
    plt.legend(['Exp_data', 'nothing', 'MDSINE', 'nothing', 'MDSINE-FBA'])
    plt.yscale('log')
    plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_comparison_abs_abun_' + str(test_num) + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")

    # %%
    ### Here plot the metabolomics data 
    metabolomics_metadata_sub_1948 = metabolomics_metadata_raw[metabolomics_metadata_raw['Mouse'] == float(subject_to_predict)]
    metabolomics_metadata_sub_1948

    met_days_1948_dict = dict(zip(metabolomics_metadata_sub_1948.index.tolist(),metabolomics_metadata_sub_1948['Rec_day_adj'].tolist() ))
    met_days_1948_dict

    metabolomics_metadata_sub_2000 = metabolomics_metadata_raw[metabolomics_metadata_raw['Mouse'] == 2000.0]
    metabolomics_metadata_sub_2000

    met_days_2000_dict = dict(zip(metabolomics_metadata_sub_2000.index.tolist(),metabolomics_metadata_sub_2000['Rec_day_adj'].tolist() ))
    met_days_2000_dict

    # %%
    ### Filter the metabolomics data by the correct samples

    metabolomics_data_sub_1948 = metabolomics_data[metabolomics_data['SampleName'].isin(met_days_1948_dict.keys())]
    metabolomics_data_sub_1948 = metabolomics_data_sub_1948.set_index('SampleName')
    metabolomics_data_sub_1948_ra = metabolomics_data_sub_1948.T/metabolomics_data_sub_1948.T.sum(axis=0)
    metabolomics_data_sub_1948_ra.columns = np.sort(list(met_days_1948_dict.values()))

    #change_met_ids = bigg_to_agora_exchange_ids(metabolomics_data_sub_1948_ra.index.tolist())
    change_met_ids = []
    for i in metabolomics_data_sub_1948_ra.index.tolist():
        change_met_ids.append('EX_' + bigg_to_modelseed[i] + '_b')
    metabolomics_data_sub_1948_ra.index = change_met_ids
    #metabolomics_data_sub_1948_ra.head()

    metabolomics_data_sub_2000 = metabolomics_data[metabolomics_data['SampleName'].isin(met_days_2000_dict.keys())]
    metabolomics_data_sub_2000 = metabolomics_data_sub_2000.set_index('SampleName')
    metabolomics_data_sub_2000_ra = metabolomics_data_sub_2000.T/metabolomics_data_sub_2000.T.sum(axis=0)
    ### columns were placed incorrectly it seems so need to match 
    change_met_ids = []
    for i in metabolomics_data_sub_2000_ra.index.tolist():
        change_met_ids.append('EX_' + bigg_to_modelseed[i] + '_b')
    metabolomics_data_sub_2000_ra.index = change_met_ids
    metabolomics_data_sub_2000_ra.columns = np.sort(list(met_days_2000_dict.values()))

    change_met_ids = metabolomics_data_sub_2000_ra.index.tolist()


    bigg_to_modelseed
    #metabolomics_data_sub_2000_ra.index = change_met_ids

    # %%
    met_pool_over_time_df_melt_filt = met_pool_over_time_df_melt.set_index('Metabolite').loc[change_met_ids].reset_index()
    met_pool_over_time_df_melt_filt.head()

    # %%
    fig, axs = plt.subplots(figsize= (15,10))
    sns.lineplot(data=met_pool_over_time_df_melt_filt, x='Time', y='Concentration', hue = 'Metabolite')
    plt.yscale('log')
    plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_metabolites_over_time_test_filt' + str(test_num) + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    #plt.show()


    # %%
    metabolomics_data_sub_1948_ra

    # %%

    metabolomics_data_sub_1948_ra_melt = metabolomics_data_sub_1948_ra.melt(ignore_index=False).reset_index()
    metabolomics_data_sub_1948_ra_melt.columns = ['metabolites', 'time', 'concentration']
    metabolomics_data_sub_1948_ra_melt.head()

    metabolomics_data_sub_2000_ra_melt = metabolomics_data_sub_2000_ra.melt(ignore_index=False).reset_index()
    metabolomics_data_sub_2000_ra_melt.columns = ['metabolites', 'time', 'concentration']
    metabolomics_data_sub_2000_ra_melt.head()

    # %%
    metabolomics_data_sub_2000_ra

    cmpd_names_adjust = []
    for i in cmpd_names:
        cmpd_names_adjust.append('EX_' + i + '_b')

    # %%
    ### Prepare the simulation metabolomic data 

    #met_pool_over_time_df_filt = met_pool_over_time_df.T.loc[agora_ex_ids_list]
    met_pool_over_time_df_filt = met_pool_over_time_df.T.loc[cmpd_names_adjust]
    met_pool_over_time_df_filt = met_pool_over_time_df_filt.T.reset_index()
    met_pool_over_time_df_filt['index'] = (met_pool_over_time_df_filt['index']/time_scaler)-3
    met_pool_over_time_df_filt = met_pool_over_time_df_filt.set_index('index').T
    met_pool_over_time_df_filt_ra = met_pool_over_time_df_filt/met_pool_over_time_df_filt.sum(axis=0)
    met_pool_over_time_df_filt_ra_melt = met_pool_over_time_df_filt_ra.melt(ignore_index=False).reset_index()
    met_pool_over_time_df_filt_ra_melt.columns = ['metabolites', 'time', 'concentration']
    met_pool_over_time_df_filt_ra_melt.head()
    # use this to filter, agora_ex_ids_list

    # %%
    fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]}
    )

    # Define stacking order
    feature_order = metabolomics_data_sub_2000_ra_melt['metabolites'].value_counts().index.tolist()
    reversed_order = feature_order[::-1]  # For stackplot



    palette = sns.color_palette("Spectral", n_colors=len(feature_order))  # or use 'husl', 'Set2', etc.
    color_map = dict(zip(feature_order, palette))

    # Pivot for stackplot in reversed stacking order
    met_pool_over_time_df_filt_ra_pivot = (
        met_pool_over_time_df_filt_ra_melt
        .pivot(index='time', columns='metabolites', values='concentration')
        .fillna(0)
    )[reversed_order]

    # Set categorical order for histogram
    metabolomics_data_sub_2000_ra_melt['metabolites'] = pd.Categorical(metabolomics_data_sub_2000_ra_melt.reset_index()['metabolites'], categories=feature_order, ordered=True)

    # Top: histogram
    sns.histplot(
        data=metabolomics_data_sub_2000_ra_melt,
        x='time',
        weights='concentration',
        hue='metabolites',
        multiple='stack',
        ax=ax_top,
        alpha=0.8,
        binwidth=1,
        palette=color_map
    )
    ax_top.legend_.remove()
    ax_top.set_ylabel('Relative Abundance')
    ax_top.set_title('Experimental Abundance')
    #ax_top.legend(title='FeatureID', bbox_to_anchor=(1.05, 1), loc='upper left')


    # %%
    met_pool_over_time_df_filt_ra_pivot.sum(axis=1)

    # %%
    fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]}
    )

    # Define stacking order
    feature_order = metabolomics_data_sub_1948_ra_melt['metabolites'].value_counts().index.tolist()
    reversed_order = feature_order[::-1]  # For stackplot



    palette = sns.color_palette("Spectral", n_colors=len(feature_order))  # or use 'husl', 'Set2', etc.
    color_map = dict(zip(feature_order, palette))

    # Pivot for stackplot in reversed stacking order
    met_pool_over_time_df_filt_ra_pivot = (
        met_pool_over_time_df_filt_ra_melt
        .pivot(index='time', columns='metabolites', values='concentration')
        .fillna(0)
    )[reversed_order]

    # Set categorical order for histogram
    metabolomics_data_sub_1948_ra_melt['metabolites'] = pd.Categorical(metabolomics_data_sub_1948_ra_melt.reset_index()['metabolites'], categories=feature_order, ordered=True)

    # Top: histogram
    sns.histplot(
        data=metabolomics_data_sub_1948_ra_melt,
        x='time',
        weights='concentration',
        hue='metabolites',
        multiple='stack',
        ax=ax_top,
        alpha=0.8,
        binwidth=1,
        palette=color_map
    )
    ax_top.legend_.remove()
    ax_top.set_ylabel('Relative Abundance')
    ax_top.set_title('Experimental Abundance')
    #ax_top.legend(title='FeatureID', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_bottom.stackplot(
        met_pool_over_time_df_filt_ra_pivot.index,
        *[met_pool_over_time_df_filt_ra_pivot[col] for col in met_pool_over_time_df_filt_ra_pivot.columns],
        alpha=0.8,
        colors=[color_map[feat] for feat in reversed_order],
        labels=reversed_order
    )

    # Create custom legend handles using your color map
    legend_elements = [
        Patch(facecolor=color_map[feat], label=feat)
        for feat in feature_order
    ]

    # Add custom legend to the *figure* (not either axis), outside plot
    fig.legend(
        handles=legend_elements,
        title='Metabolite',
        loc='center left',
        bbox_to_anchor=(.8, 0.5),  # Push legend just outside right edge
        borderaxespad=0,
        frameon=False,
        ncol=1
    )
    # Adjust spacing to make room for the legend
    plt.subplots_adjust(hspace=0.1, right=1)  # Shrink plot width
    overall_title = 'Subject ' + str(subject_to_plot)
    plt.suptitle(overall_title, x=0.6, fontsize = 20)
    plt.subplots_adjust(right=0.75)
    plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_mets_exp_vs_sim_over_time_test_' + str(test_num) + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    #plt.show()

    # %%
    metabolomics_data_sub_1948 = metabolomics_data_sub_1948.T

    metabolomics_data_sub_1948.columns = np.sort(list(met_days_1948_dict.values()))
    metabolomics_data_sub_1948


    # %%
    #change_met_ids = bigg_to_agora_exchange_ids(metabolomics_data_sub_1948.index.tolist())
    metabolomics_data_sub_1948.index = change_met_ids
    metabolomics_data_sub_1948.head()

    # %%
    num_plots = 60
    cols = 6  # Number of columns in the grid
    rows = (num_plots + cols - 1) // cols  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=False)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i,met in enumerate(met_pool_over_time_df_filt.index.tolist()):
        temp_sim = met_pool_over_time_df_filt.loc[met,:]
        temp_exp = metabolomics_data_sub_1948.loc[met,:]
            # Scatter plot on the respective subplot
        sns.lineplot(ax=axes[i], 
                        x=temp_sim.index.to_list(), y=temp_sim.to_list())
        sns.scatterplot(ax=axes[i], 
                        x=temp_exp.index.to_list(), y=temp_exp.to_list())

        axes[i].set_title(f"{met}")
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Conc')
        #axes[i].set_yscale('log')
        #axes[i].legend_.remove()

    # Hide any empty subplots if the number of plots is not a perfect square
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])


    plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_mets_exp_vs_sim_individual_scatterplots_' + str(test_num) + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")

    # %%
    metabolomics_data_sub_1948_melt = metabolomics_data_sub_1948.melt(ignore_index=False).reset_index()
    metabolomics_data_sub_1948_melt.columns = ['metabolites', 'time', 'concentration']
    metabolomics_data_sub_1948_melt.head()

    # %%
    ### Prepare the simulation metabolomic data 
    met_pool_over_time_df_filt_melt = met_pool_over_time_df_filt.melt(ignore_index=False).reset_index()
    met_pool_over_time_df_filt_melt.columns = ['metabolites', 'time', 'concentration']
    met_pool_over_time_df_filt_melt.head()
    # use this to filter, agora_ex_ids_list

    # %%
    ### ok now make scatterplot for each time point 

    metabolomics_data_sub_1948_melt['time'] = pd.to_numeric(metabolomics_data_sub_1948_melt['time'], downcast='float') 
    merged_exp_sim_met_data = metabolomics_data_sub_1948_melt.merge(met_pool_over_time_df_filt_melt, on=['time', 'metabolites'])
    merged_exp_sim_met_data.columns = ['metabolites', 'time', 'exp_conc', 'sim_conc']
    #met_pool_over_time_df_filt_ra

    # %%
    merged_exp_sim_met_data

    # %%
    x = y = np.linspace(0,25,50)

    # %%
    num_plots = 8
    cols = 2  # Number of columns in the grid
    rows = (num_plots + cols - 1) // cols  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=False)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i,time in enumerate(merged_exp_sim_met_data['time'].unique().tolist()):
        temp = merged_exp_sim_met_data[merged_exp_sim_met_data['time'] == time]
        
            # Scatter plot on the respective subplot
        sns.scatterplot(ax=axes[i], data=temp, 
                        x=temp['exp_conc'], y=temp['sim_conc'], hue='metabolites', palette=color_map)

        sns.lineplot(ax=axes[i], x=x, y = y)

        axes[i].set_xlim(0,25)
        axes[i].set_ylim(0,25)
        axes[i].set_title(f"Day {time}")
        axes[i].set_xlabel('Exp Conc')
        axes[i].set_ylabel('Sim Conc')
        axes[i].legend_.remove()

    # Hide any empty subplots if the number of plots is not a perfect square
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Add custom legend to the *figure* (not either axis), outside plot
    fig.legend(
        handles=legend_elements,
        title='Metabolite',
        loc='center left',
        bbox_to_anchor=(.8, 0.5),  # Push legend just outside right edge
        borderaxespad=0,
        frameon=False,
        ncol=1
    )
    # Adjust spacing to make room for the legend
    plt.subplots_adjust(hspace=.2, right=1)  # Shrink plot width
    overall_title = 'Subject ' + str(subject_to_plot)
    plt.suptitle(overall_title, x=0.6, fontsize = 20)
    plt.subplots_adjust(right=0.75)
    plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_mets_scatter_exp_vs_sim_' + str(test_num) + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    #plt.show()

end_time = datetime.now()

total_time = end_time-start_time
print(total_time)