### Hayden Gallo
### 10/20/25
### Bucci Lab
### Running gLV-FBA on Venturelli data, simulating all co-culture combos 
### This an edit to the script gLV_FBA_Venturelli_sims_panGenus where the goal is to parallelize the simulations via 
### making deep copies of the GEMs in memory and also leveraging joblib.parallel


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
import argparse
import re

from helper_functions import *

from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import json
import logging
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# export GRB_LICENSE_FILE=/Users/haydengallo/.gurobi/gurobi.lic


#################################################################################################################################################################
### this is used to surpress all logging from loading in the kbase models with cobra, such that they don't get added to the glv_fba log file and overcrowd it ### 
logging.getLogger("cobra").setLevel(logging.ERROR)
#################################################################################################################################################################

#############################################
### Parameters to set prior to simulation ###
#############################################

#data_dir = '/Users/haydengallo/UMass_Dropbox/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data'
data_dir = '/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data'
cobra_models_dir_path = data_dir + '/panGenusModels_Venturelli_corrected'
scal_fac = 50
t_steps = 48

test_name = 'panGenusmodel_sims'
test_num = 22


 ### Simulation notes
notes = 'Running all sims on desktop, not allowing uptake of but, succ, also think i solved the issue of having incorrect order of GEMs sometimes, also putting scaling factor to 50, maybe that will help, now testing parallelization of simulations'

#plot_dir_path_base = '/Users/haydengallo/UMass_Dropbox/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data/' + test_name + '/test_' + str(test_num)

plot_dir_path_base = '/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data/' + test_name + '/test_' + str(test_num)

plot_dir = Path(plot_dir_path_base)
os.makedirs(plot_dir, exist_ok=True)


### define generalized Lotka-Volterra function 
def gLV(t, init_abun, paired_growth_matrix, basal_grow):

    return ((np.dot(init_abun, paired_growth_matrix) + basal_grow) * init_abun)


### Setting up some metadata from metadata_2019_06_17.py Clark et al 2021 


#Define the descriptors involved in this experiment
numspecies=25
allspecies=['ER','FP','AC','CC','RI','EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA']
phylogeny=['PC','PJ','BV','BF','BO','BT','BC','BY','BU','DP','BL','BA','BP','CA','EL','FP','CH','AC','BH','CG','ER','RI','CC','DL','DF']
phylogeny_nobpb=['PC','PJ','BV','BF','BO','BT','BC','BY','BU','DP','BL','BA','BP','CA','EL','CH','BH','CG','DL','DF']
phylogeny_nogaps=['PC','PJ','BF','BO','BT','DP','BA','BP','CA','EL','CH','BH','CG','DL','DF']
speciesvectorDict={}
n=1
for species in allspecies:
	speciesvectorDict[species]=n
	n+=1

k=0
phylogenyvectorDict={}
for species in phylogeny:
	phylogenyvectorDict[species]=k
	k+=1

bpbindices=[15,17,20,21,22]
bpbspecies=['ER','FP','AC','CC','RI']
spbspecies=['PJ','BT','BF','BC','BO','BV','PC']
others=['EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA']
comms=['COMM0','COMM1','COMM2','COMM3','COMM4']

commsDict={
	'COMM0':['DP','BH','CA','PC','EL','CH','BO','BT','BU','BV'],
	'COMM1':['ER','FP','DP','BH','CA','PC','EL','CH','BO','BT','BU','BV'],
	'COMM2':['ER','FP','AC','HB','CC','RI','DP','BH','CA','PC','EL','CH','BO','BT','BU','BV'],
	'COMM3':['ER','FP','AC','HB','CC','RI','EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA'],
	'COMM4':['EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA'],
	'COMM5':['ER','FP','AC','CC','RI','DP','BH','CA','PC','EL','CH','BO','BT','BU','BV'],
	'COMM6':['ER','FP','AC','CC','RI','EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA'],
	'COMM7':['ER','FP','RI','BH','DP','PJ','AC','CC','BV','DL','BY','BL','DF','BA','EL'],
	'COMM8':['ER','FP','RI','AC','CC']
}

COMM0=['DP','BH','CA','PC','EL','CH','BO','BT','BU','BV']
COMM1=['ER','FP','DP','BH','CA','PC','EL','CH','BO','BT','BU','BV']
COMM2=['ER','FP','AC','HB','CC','RI','DP','BH','CA','PC','EL','CH','BO','BT','BU','BV']
COMM3=['ER','FP','AC','HB','CC','RI','EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA']
COMM4=['EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA']
COMM5=['ER','FP','AC','CC','RI','DP','BH','CA','PC','EL','CH','BO','BT','BU','BV']
COMM6=['ER','FP','AC','CC','RI','EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA']
COMM7=['ER','FP','RI','BH','DP','PJ','AC','CC','BV','DL','BY','BL','DF','BA','EL']
COMM8=['ER','FP','RI','AC','CC']

LOOComms=['COMM6']
for species in phylogeny:
	LOOComms.append('COMM6*'+species)

		
namedict={
   'BA': 'Bifidobacterium_adolescentis_ATCC_15703_NC_008618',
   'CA': 'Collinsella_aerofaciens_ATCC_25986',
   'BT': 'Bacteroides_thetaiotaomicron_VPI-5482_NC_004663',
   'BU': 'Bacteroides_uniformis_ATCC_8492',
   'PC': 'Prevotella_copri_DSM_18205',
   'AC': 'Anaerostipes_caccae_DSM_14662_4',
   'BH': 'Blautia_hydrogenotrophica_DSM_10507',
   'CC': 'Coprococcus_comes_1.0.1_Cont2276_NZ_ABVR01000038',
   'CG': 'Clostridium_asparagiforme_DSM_15981_C_asparagiforme_1.0_Cont7.2_NZ_ACCJ01000522',
   'ER': 'Eubacterium_rectale_ATCC_33656_NC_012781',
   'DP': 'Desulfovibrio_piger_ATCC_29098',
   'EL': 'Eggerthella_lenta_DSM_2243_NC_013204',
   'BY': 'Bacteroides_cellulosilyticus_DSM_14838_1.0_Cont4.1_NZ_ACCH01000108',
   'BF': 'Bacteroides_fragilis_NCTC_9343',
   'CD': 'Clostridioides_difficile',
   'RI': 'Roseburia_intestinalis_L1_82',
   'BP': 'Bifidobacterium_pseudocatenulatum_DSM20438',
   'BV': 'Bacteroides_vulgatus_ATCC_8482_NC_009614',
   'CH': 'Clostridium_hiranonis_DSM_13275',
   'DF': 'Dorea_formicigenerans_ATCC_27755',
   'CS': 'Clostridium_scindens_ATCC_35704',
   'PJ': 'Parabacteroides_johnsonii_DSM_18315_NZ_ABYH01000014',
   'FP': 'Faecalibacterium_prausnitzii_A2_165_NZ',
   'EH': 'Eubacterium_hallii_DSM_3353_1.0_Cont383.1_NZ_ACEP01000116',
   'EC': 'Escherichia_coli',
   'BC': 'Bacteroides_caccae_ATCC_43185',
   'HB': 'Holdemanella_biformis_DSM_3989',
   'BO': 'Bacteroides_ovatus_ATCC_8483',
   'DL': 'Dorea_longicatena_DSM_13814',
   'BL': 'Bifidobacterium_longum_subsp_infantis',
   'B.cereus':'Bacillus_cereus'
   }




## load in the data needed for running simulations

int_matrix_path = data_dir + '/interaction_matrix.npy'
int_matrix = np.load(int_matrix_path, allow_pickle=True)

growth_rates_path = data_dir + '/growth_rates.npy'
growth_rates = np.load(growth_rates_path, allow_pickle=True)

m2_init_abun_path = data_dir + '/m2_time_series_init_no_mets.csv'
m2_init_abun = pd.read_csv(m2_init_abun_path)

m2_init_abun = m2_init_abun.set_index('Experiments')


m2_final_abun_path = data_dir + '/m2_time_series_final_w_mets.csv'
m2_final_abun = pd.read_csv(m2_final_abun_path, index_col = 0)

m2_final_abun = m2_final_abun.set_index('Experiments')

m2_final_abun = m2_final_abun.drop(columns = ['Time'])
m2_final_abun_met  = m2_final_abun[['Butyrate', 'Acetate', 'Lactate', 'Succinate']]
m2_final_abun_bac = m2_final_abun.drop(columns = ['Butyrate', 'Acetate', 'Lactate', 'Succinate'])


time = np.linspace(0,t_steps,t_steps+1)
args = (int_matrix, growth_rates)



### load in all of the GEMs

### load the cobra models into memory i guess

#cobra_models_dir_path = data_dir + '/AGORA_GEMs'


cobra_models_dir = Path(cobra_models_dir_path)

# Make the data and validation Study objects



bac_abun_predict_final_df = m2_final_abun_bac.copy()

met_abun_predict_final_df = m2_final_abun_met.copy()

bac_abun_predict_final_df[:] = 0
met_abun_predict_final_df[:] = 0 

met_abun_predict_final_df.columns = ['EX_but(e)','EX_ac(e)', 'EX_lac_L(e)',  'EX_succ(e)']



### Set media conditions 

# Example medium definition for COBRApy + AGORA
# exchange reactions : uptake flux (mmol/gDW/h)
defined_media = {
    "EX_ca2(e)":  -1.290526021,
    "EX_na1(e)": -47.34272139,
    "EX_cu2(e)": -0.021265311,
    "Ex_cu(e)":  -0.021265311,
    "EX_so4(e)": -10.22721245,
    "EX_pydx(e)": -0.009822218,
    "EX_thymd(e)": -0.021,
    "EX_xan(e)":  -0.024981921,
    "EX_fol(e)":  -0.002402832,
    "EX_orot(e)": -0.064061499,
    "EX_k(e)":    -6.617592674,
    "EX_cobalt2(e)": -0.055,
    "EX_no3(e)":  -0.109323669,
    "EX_fe3(e)":  -0.066,
    "EX_fe2(e)":  -0.066,
    "EX_mg2(e)":  -4.380495746,
    "EX_mn2(e)":  -0.33,
    "EX_mobd(e)": -0.004856255,
    "EX_slnt(e)": -0.000578243,
    "EX_tungs(e)": -0.003403444,
    "EX_cl(e)":  -14.45186396,
    "EX_ni2(e)": -0.01543217,
    "EX_zn2(e)": -0.061931009,
    "EX_pnto_R(e)": -0.000419695,
    "EX_cbl1(e)": -1.47561E-06,
    "EX_pydxn(e)": -0.000972583,
    "EX_h(e)": -0.001565646,
    "EX_ribflv(e)": -1.000531406,
    "EX_thf(e)": -2.80628E-06,
    "EX_thm(e)": -0.000593064,
    "EX_4abz(e)": -0.073072918,
    "EX_pydam(e)": -0.021,
    "EX_nh4(e)": -9.347366847,
    "EX_pi(e)": -6.61327063,
    "EX_ncam(e)": -0.034392401,
    "EX_pheme(e)": -0.015338835,
    "EX_csn(e)": -5.94059E-05,
    "EX_gua(e)": -5.95514E-05,
    "EX_ade(e)": -5.99423E-05,
    "EX_ura(e)": -5.88829E-05,
    "EX_inost(e)": -6.272202487,
    "EX_btn(e)": -0.040952069,
    "EX_ala_L(e)": -5.3,
    "EX_arg_L(e)": -21.81400689,
    "EX_asn_L(e)": -2.6,
    "EX_asp_L(e)": -0.4,
    "EX_cys_L(e)": -8.4,
    "EX_glu_L(e)": -0.662721893,
    "EX_gln_L(e)": -2.7,
    "EX_his_L(e)": -1,
    "EX_ile_L(e)": -1.6,
    "EX_leu_L(e)": -3.6,
    "EX_lys_L(e)": -2.4,
    "EX_met_L(e)": -0.84,
    "EX_phe_L(e)": -4.5,
    "EX_pro_L(e)": -5.9,
    "EX_ser_L(e)": -6.4,
    "EX_thr_L(e)": -1.9,
    "EX_trp_L(e)": -0.73,
    "EX_val_L(e)": -2.8,
    "EX_tyr_L(e)": -3.201059661,
    "EX_mops(e)": -71.68003181,
    "EX_hco3_L(e)": -47.6150797,
    "EX_arab_L(e)": -21.31486045,
    "EX_glc_D(e)": -24.97835209,
    "EX_lac_L(e)": -28.30817052,
    "EX_malt(e)": -4.382120947,
    "EX_h2o(e)": -55.50645091,
    # Trace/small additions set to -0.001
    "EX_12dgr180(e)": -1,
    "EX_acgam(e)": -1,
    "EX_adn(e)": -1,
    "EX_mqn7(e)": -1,
    "EX_mqn8(e)": -1,
    "EX_nac(e)": -1,
    "EX_nmn(e)": -1,
    "EX_ocdca(e)": -1,
    "EX_q8(e)": -1,
    "EX_sheme(e)": -1,
    "EX_spmd(e)": -1,
    "EX_o2(e)": -1,
    "EX_26dap_M(e)": -1,
    "EX_bglc(e)": -1,
    "EX_cgly(e)": -1}


defined_media_df = pd.DataFrame.from_dict(defined_media, orient='index')
defined_media_df = defined_media_df.reset_index()
defined_media_df.columns = ['reaction', 'fluxValue']
defined_media_df['fluxValue'] = -1.0*defined_media_df['fluxValue']




cobra_models = sorted(cobra_models_dir.glob('*.mat'))
cobra_models = {f.stem : f for f in cobra_models}

### Just loading the models needed in 

loaded_models = {}

#count = 0

for key in cobra_models:
    #if count == 1:
    #    break
    #print(key.split('_'))
    model_name = key.split('_')[2] + '_' + key.split('_')[3]
    #model_name = key.split('.')[0]
    #model_name = model_name[3:]
    #print(model_name)
    model = cobra.io.load_matlab_model(cobra_models[key])
    loaded_models[model_name] = model
    #count+=1


adjusted_names = []

for key in namedict:
    temp = namedict[key].split('_')
    temp_name = temp[0] + '_' + temp[1]
    adjusted_names.append(temp_name)

adjusted_names_dict = dict(zip(namedict.keys(), adjusted_names))


correct_model_dict_order = {}
correct_model_name_order = []


for i in allspecies:
    model_to_grab = adjusted_names_dict[i]
    #model_to_grab_genus = model_to_grab.split('_')[0]
    print(model_to_grab)
    correct_model_dict_order[model_to_grab] = loaded_models[model_to_grab]
    correct_model_name_order.append(model_to_grab)

print(correct_model_name_order)


param_save = {'Sim_num': test_num, 'Scaling_factor':scal_fac, 'Total_time_steps': t_steps, 'notes':notes}
param_save_file_name = plot_dir_path_base + '/params.txt'
with open(param_save_file_name, 'w') as file:
    file.write(json.dumps(param_save))




#for sim in tqdm(range(0, len(m2_init_abun))):
#for sim in tqdm(range(39, 42)):
def run_simulations(i):

    
      
      ### Testing one simulation 

    #sim_to_grab = sim
    sim_to_grab = i

    spec_exp_name = m2_init_abun.index.to_list()[sim_to_grab]

    plot_dir_path_spec_exp = plot_dir_path_base + '/' + spec_exp_name

    plot_dir = Path(plot_dir_path_spec_exp)
    os.makedirs(plot_dir, exist_ok=True)


    init_abun = np.array(m2_init_abun.iloc[sim_to_grab,:].to_list())
    sol = odeint(gLV,  init_abun,time, args = args, tfirst = True)
    sol.shape


    m2_final_abun_bac_filt = m2_final_abun_bac.iloc[sim_to_grab,:]
    m2_final_abun_met_filt = m2_final_abun_met.iloc[sim_to_grab,:]

    bac_to_keep_for_inf = list(np.where(init_abun != 0)[0])

    glv_derived_growth_rates = np.zeros([t_steps+1,25])

    for i in range(0,t_steps+1):
        glv_derived_growth_rates[i,:] = gLV(i, sol[i,:], paired_growth_matrix=int_matrix, basal_grow=growth_rates)

        
    glv_derived_growth_rates_df = pd.DataFrame(glv_derived_growth_rates)

    glv_abun_df = pd.DataFrame(sol)

    models_list = list(correct_model_dict_order.values())


    ### Set media conditions 



    defined_media_df_filt = defined_media_df[defined_media_df['fluxValue'] != -0.001]

    rate_array = np.zeros((25,glv_abun_df.shape[0]-1))

    for i in range(0,(len(glv_abun_df.T.columns)-1)):

        rate_array[:,i] = (glv_abun_df.T.iloc[:,i+1]/glv_abun_df.T.iloc[:,i])-1


    #for i in range(0, len(models_list)):


    #    test_media = make_media(models_list[i], defined_media_df)
    #    models_list[i].medium = test_media
        #models_list[i].optimize()
    #    print(models_list[i].slim_optimize())


    rate_df = pd.DataFrame(rate_array).fillna(0)

    glv_abun_df = glv_abun_df.iloc[:,bac_to_keep_for_inf]
    rate_df = rate_df.iloc[bac_to_keep_for_inf,:]
    init_abun = init_abun/scal_fac
    init_abun = list(init_abun[bac_to_keep_for_inf])
    correct_model_name_order_current_sim=list(np.array(correct_model_name_order)[bac_to_keep_for_inf])
    models_list_current_sim=list(np.array(models_list)[bac_to_keep_for_inf])
    glv_abun_df = glv_abun_df/scal_fac

    #print(models_list_current_sim)
    #print(bac_to_keep_for_inf)

    for i in range(0, len(models_list_current_sim)):


        test_media = make_media(models_list_current_sim[i], defined_media_df)
        models_list_current_sim[i].medium = test_media
        #models_list[i].optimize()
        print(models_list_current_sim[i].slim_optimize())

    ## Here check to make sure we have chosen the correct GEMs to run the inference on 
        
    filt_columns_list_names = m2_init_abun.iloc[:,bac_to_keep_for_inf].columns.to_list()
    checking_order_list = []

    for i in filt_columns_list_names:

        checking_order_list.append(adjusted_names_dict[i])
    
    if checking_order_list == correct_model_name_order_current_sim:
        print('yes, the order and identity of GEMs selected for inference is correct')
    else:
        print('no good')#break


    ### run actual inference
    met_pool_over_time, model_abun_dict, mets_used_for_constraint = static_dfba(list_model_names=correct_model_name_order_current_sim, list_models=models_list_current_sim, initial_abundance=init_abun, total_sim_time=t_steps, num_t_steps=t_steps, glv_out=np.array(glv_abun_df), glv_params=None, environ_cond=defined_media_df, pfba=False, MDSINE_rates=rate_df, Diet=None, output_file_path = plot_dir_path_spec_exp, flux_sampling=False, host = None, random_constraints = 'No', AGORA_models = 'yes')#interpolated_met_values)


    FBA_biomass = np.zeros([len(model_abun_dict.keys()), t_steps+1])
    glv_biomass = np.zeros([len(model_abun_dict.keys()), t_steps+1])

    
    ### Convert FBA abun output to relative abundance 
    count = 0
    for key in model_abun_dict:
        FBA_biomass[count,:] = model_abun_dict[key]['fba_biomass']
        glv_biomass[count,:] = model_abun_dict[key]['glv_out']
        count+=1

    FBA_biomass_df = pd.DataFrame(FBA_biomass)
    FBA_biomass_df.index = model_abun_dict.keys()

    index_to_filter_by = FBA_biomass_df.index


    glv_biomass_df = pd.DataFrame(glv_biomass)
    glv_biomass_df.index = model_abun_dict.keys()

    index_to_filter_by = glv_biomass_df.index

    
    FBA_biomass_df_plot = pd.DataFrame(FBA_biomass)
    FBA_biomass_df_plot.index = model_abun_dict.keys()




    FBA_biomass_df_plot = FBA_biomass_df_plot.melt(ignore_index=False)
    FBA_biomass_df_plot = FBA_biomass_df_plot.reset_index()
    FBA_biomass_df_plot.columns = ['FeatureID','time', 'count']
    #FBA_biomass_df_plot['time'] = (FBA_biomass_df_plot['time']/time_scaler)-3
    FBA_biomass_df_plot['count'] = FBA_biomass_df_plot['count']*scal_fac




    glv_biomass_df_plot = glv_biomass_df.melt(ignore_index=False)
    glv_biomass_df_plot = glv_biomass_df_plot.reset_index()
    glv_biomass_df_plot.columns = ['FeatureID','time', 'count']
    #FBA_biomass_df_plot['time'] = (FBA_biomass_df_plot['time']/time_scaler)-3
    glv_biomass_df_plot['count'] = glv_biomass_df_plot['count']*scal_fac

    fig, axs = plt.subplots(figsize= (15,10))
    sns.lineplot(data=FBA_biomass_df_plot, x='time', y='count', hue = 'FeatureID')
    #plt.yscale('log')
    plot_file_name = plot_dir_path_spec_exp + '/biomass_single_plot_' + spec_exp_name + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")


    m2_final_abun_bac_filt = list(m2_final_abun_bac_filt[bac_to_keep_for_inf])


    num_plots = len(model_abun_dict)
    cols = 5  # Number of columns in the grid
    rows = (num_plots + cols - 1) // cols  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=False)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i,key in enumerate(model_abun_dict):
        temp_glv = glv_biomass_df_plot[glv_biomass_df_plot['FeatureID'] == key]
        temp_FBA = FBA_biomass_df_plot[glv_biomass_df_plot['FeatureID'] == key]
            # Scatter plot on the respective subplot
        #sns.lineplot(ax=axes[i], x=temp_sim.index.to_list(), y=temp_sim.to_list(), color = 'red', lw = 2)
        sns.lineplot(ax = axes[i], data=temp_glv, x = 'time', y = 'count', color = 'blue', lw = 3)
        
        sns.lineplot(ax = axes[i], data=temp_FBA, x = 'time', y = 'count', color = 'red', lw = 3)
        sns.scatterplot(ax=axes[i], x=[t_steps], y=m2_final_abun_bac_filt[i], color = 'green', s = 100)

        axes[i].set_title(f"{key}")
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Conc')
        #axes[i].set_yscale('log')
        #axes[i].legend_.remove()

    # Hide any empty subplots if the number of plots is not a perfect square
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plot_file_name = plot_dir_path_spec_exp + '/biomass_exp_vs_sim_single_species_' + spec_exp_name + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")


        
    FBA_biomass_df_plot_unstack = FBA_biomass_df_plot.pivot(index='FeatureID', columns ='time')['count']
    FBA_biomass_df_plot_unstack = pd.DataFrame(FBA_biomass_df_plot_unstack.sum(axis=0)).reset_index()
    FBA_biomass_df_plot_unstack.columns = ['time', 'abun']
    FBA_biomass_df_plot_unstack['time'] = pd.to_numeric(FBA_biomass_df_plot_unstack['time'])
    FBA_biomass_df_plot_unstack['time'] = pd.to_numeric(FBA_biomass_df_plot_unstack['time'])

    # 
    ### Need to plot metabolite trajectories too

    met_pool_over_time_df = pd.DataFrame(met_pool_over_time)
    met_pool_over_time_df = met_pool_over_time_df.fillna(0)
    met_pool_over_time_df_melt= met_pool_over_time_df.melt(ignore_index=False)
    met_pool_over_time_df_melt = met_pool_over_time_df_melt.reset_index()
    met_pool_over_time_df_melt.columns = ['Time','Metabolite', 'Concentration']




    fig, axs = plt.subplots(figsize= (15,10))
    sns.lineplot(data=met_pool_over_time_df_melt, x='Time', y='Concentration', hue = 'Metabolite')
    plt.yscale('log')
    plot_file_name = plot_dir_path_spec_exp + '/met_concentration' + spec_exp_name + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    

    filt_mets = ['EX_lac_L(e)', 'EX_succ(e)','EX_but(e)', 'EX_ac(e)']

    mets_present = list(set(np.unique(met_pool_over_time_df_melt['Metabolite'].to_list())).intersection(set(filt_mets)))
    #filt_mets = ['EX_lac_L(e)','EX_but(e)', 'EX_ac(e)']
    #filt_mets = ['EX_cpd00029_b', 'EX_cpd00159_b']

    #filt_mets = ['EX_lac_L(e)']

    met_pool_over_time_df_melt_filt = met_pool_over_time_df_melt.set_index('Metabolite').loc[mets_present].reset_index()


    met_pool_over_time_df_melt_filt





    fig, ax = plt.subplots(figsize=(15, 10))

    # Define fixed color palette for your 4 key metabolites
    met_palette = {
        "EX_but(e)": "tab:orange",
        "EX_ac(e)": "tab:green",
        "EX_lac_L(e)": "tab:blue",
        "EX_succ(e)": "tab:red"
    }

    # Plot only metabolites present in the simulation
    sns.lineplot(
        data=met_pool_over_time_df_melt_filt,
        x='Time',
        y='Concentration',
        hue='Metabolite',
        palette=met_palette,
        ax=ax
    )

    # Mapping between readable names and BiGG IDs
    met_mapping = {
        "Butyrate": "EX_but(e)",
        "Acetate": "EX_ac(e)",
        "Lactate": "EX_lac_L(e)",
        "Succinate": "EX_succ(e)"
    }

    # Add scatter points for *all 4 experimental* metabolites
    for name, bigg_id in met_mapping.items():
        y_val = m2_final_abun_met_filt.get(name, None)
        if y_val is not None:
            color = met_palette[bigg_id]
            sns.scatterplot(x=[t_steps], y=[y_val], color=color, s=100, edgecolor='black', label=f"{name} (exp)", ax=ax)

    # Combine legend nicely
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # remove duplicates
    ax.legend(unique.values(), unique.keys(), title="Metabolite", fontsize=12)

    ax.set_title("Simulated vs Experimental Metabolite Concentrations", fontsize=14)
    ax.set_ylabel("Concentration")
    ax.set_xlabel("Time")





    plot_file_name = plot_dir_path_spec_exp + '/mets_of_int_exp_vs_sim' + spec_exp_name + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")


    bac_abun_predict_final_df.iloc[sim_to_grab,bac_to_keep_for_inf] = FBA_biomass_df_plot[FBA_biomass_df_plot['time'] == t_steps]['count'].to_list()
    met_pool_over_time_df_melt_filt_final_t_pt = met_pool_over_time_df_melt_filt[met_pool_over_time_df_melt_filt['Time'] == t_steps].set_index('Metabolite').reindex(met_abun_predict_final_df.columns)
    met_abun_predict_final_df.iloc[sim_to_grab,:] = met_pool_over_time_df_melt_filt_final_t_pt['Concentration'].to_list()

print(joblib.Parallel(n_jobs = -1, prefer='threads')(joblib.delayed(run_simulations)(i) for i in tqdm(range(0, len(m2_init_abun)))))


met_final_pre_path = plot_dir_path_base + '/final_met_predict_df.csv'
abun_final_pre_path = plot_dir_path_base + '/final_abun_predict_df.csv'

met_abun_predict_final_df.to_csv(met_final_pre_path)
bac_abun_predict_final_df.to_csv(abun_final_pre_path)



