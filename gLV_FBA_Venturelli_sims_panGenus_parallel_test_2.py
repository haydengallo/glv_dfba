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
# directory for pan genus models 
cobra_models_dir_path = data_dir + '/panGenusModels_Venturelli_corrected'
### directory for pan species models
#cobra_models_dir_path = data_dir + '/panSpeciesModels_Venturelli'
### directory for strain models
#cobra_models_dir_path = data_dir + '/AGORA_GEMs'
scal_fac = 1
t_steps = 48

test_num = 34
### are we using single timepoint dataset or multi?
multi_t_pt_dataset = 'yes'

if multi_t_pt_dataset == 'yes':
    test_name = 'panGenusmodel_sims_multi_t_pt'
    #test_name = 'panSpeciesmodel_sims_multi_t_pt'
    #test_name = 'Strainmodel_sims_multi_t_pt'
    
else:
    test_name = 'panGenusmodel_sims'
    #test_name = 'panSpeciesmodel_sims'




 ### Simulation notes
notes = 'doing forward simulations using settings from test_27 of long dataset, so regular fba with cgly for a caccae growth and some oxygen, also now trying to get all time points to save for the multi t point dataset, still trying to get this to work '
#notes = 'fba now,for some reason seems in cobrapy need cgly for a caccae to grow, pan genus models, adding compounds required for each model to reach growth rate of 1, these compounds added at 1.0, regular fba, extra oxygen at 10'
#notes = 'had to redo full sims b/c of miss ordering in index of glv params from mdsine not aligning with allspecies master species order list, still correcting indexing issues, think this sim will be corrected, hopefully'
#notes = 'also added EX_alagln(e), and more fol, trying to get multi timepoint sim to work, fixing some saving issues can at least save final time point now i think, adding EX_rbflvrd(e) and EX_pnto_R(e) at basal levels to get dorea species to grow'
#notes = 'Running all sims on desktop, not allowing uptake of but, succ, also think i solved the issue of having incorrect order of GEMs sometimes, also putting scaling factor to 10, maybe that will help, now testing parallelization of simulations,' \
#'ok now this is also running with the parameters from MDSINE2 which was inferred on all of the longitudinal data. For multi timepoint simulatins, using the initial abundance values from MDSINE2 sims. Trying to get ordering of indexes correct '



#plot_dir_path_base = '/Users/haydengallo/UMass_Dropbox/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data/' + test_name + '/test_' + str(test_num)

plot_dir_path_base = '/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data/' + test_name + '/test_' + str(test_num)

plot_dir = Path(plot_dir_path_base)
os.makedirs(plot_dir, exist_ok=True)


### define generalized Lotka-Volterra function 
def gLV(t, init_abun, paired_growth_matrix, basal_grow):
    return init_abun * (basal_grow + np.dot(paired_growth_matrix, init_abun))
#def gLV(t, init_abun, paired_growth_matrix, basal_grow):

    #return ((np.dot(init_abun, paired_growth_matrix) + basal_grow) * init_abun)


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

### this is int matrix from MDSINE2
### old do not use
#int_matrix = np.load('/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_long_MDSINE_fit/merged_studies_test_3/int_matrix.npy', allow_pickle=True)
# New, do use
int_matrix = np.load('/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_long_MDSINE_fit/merged_studies_test_3/int_matrix_reindexed_12_11_25.npy', allow_pickle=True)
### this is int matrix from Venturelli 
#int_matrix_path = data_dir + '/interaction_matrix.npy'
#int_matrix = np.load(int_matrix_path, allow_pickle=True)

### this is growth rates from MDSINE2
### Old do not use
#growth_rates = np.load('/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_long_MDSINE_fit/merged_studies_test_3/growth_rates.npy', allow_pickle=True)
### new, do use
growth_rates = np.load('/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_long_MDSINE_fit/merged_studies_test_3/growth_rates_12_11_25.npy', allow_pickle=True)
### this is growth rates from Venturelli 
#growth_rates_path = data_dir + '/growth_rates.npy'
#growth_rates = np.load(growth_rates_path, allow_pickle=True)

### index of growth rates and int_matrix ordering is based on original experimental data ordering not on the allspecies list that is being used as the master order here on out



m2_init_abun_path = data_dir + '/m2_time_series_init_no_mets.csv'
m2_init_abun = pd.read_csv(m2_init_abun_path)

m2_init_abun = m2_init_abun.set_index('Experiments')

### must use the m2 final with no mets b/c species are ordered correctly here and not in m2 final with mets
m2_final_abun_bac_path = data_dir + '/m2_time_series_final_no_mets.csv'
m2_final_abun_bac = pd.read_csv(m2_final_abun_bac_path, index_col = 0)

m2_final_abun_met_path = data_dir + '/m2_time_series_final_w_mets.csv'
m2_final_abun_met = pd.read_csv(m2_final_abun_met_path, index_col = 0)

m2_final_abun_met = m2_final_abun_met.set_index('Experiments')

## loading in the multifunctional dataset here for easy simulation and switching between the two

### load in mutlifunctional dataset

multi_func_data = pd.read_csv('/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data/Baranwal_et_al_2022/2021_02_19_MultifunctionalDynamicData.csv', index_col = 0)
multi_func_data.head()

multi_func_data_filt = multi_func_data.filter(regex = '_OD|Community|Time|Butyrate|Succinate|Acetate|Lactate', axis =1)
multi_func_data_filt.columns = list(multi_func_data_filt.columns.str.split('_').str[0])
multi_func_data_filt = multi_func_data_filt.rename(columns={'Community':'Experiments'})
multi_func_data_filt_reorder = multi_func_data_filt.set_index('Experiments')
multi_func_data_filt_reorder = multi_func_data_filt_reorder.drop(['Time','Butyrate','Succinate','Acetate','Lactate'], axis=1)[m2_init_abun.columns.to_list()]
multi_func_data_filt_reorder['Time'] = multi_func_data_filt['Time'].to_list()
multi_func_data_filt_reorder['Butyrate'] = multi_func_data_filt['Butyrate'].to_list()
multi_func_data_filt_reorder['Succinate'] = multi_func_data_filt['Succinate'].to_list()
multi_func_data_filt_reorder['Acetate'] = multi_func_data_filt['Acetate'].to_list()
multi_func_data_filt_reorder['Lactate'] = multi_func_data_filt['Lactate'].to_list()
multi_func_data_filt_reorder_initial = multi_func_data_filt_reorder[multi_func_data_filt_reorder['Time'] == 0.0].drop(['Time','Butyrate','Succinate','Acetate','Lactate'], axis=1)
multi_func_data_filt_reorder_final = multi_func_data_filt_reorder[multi_func_data_filt_reorder['Time'] != 0.0]



m2_final_abun_met = m2_final_abun_met.drop(columns = ['Time'])

### here if using the multifunctional multi time point dataset switch m2_final_abun_bac and m2_final_abun_met to said dataset

if multi_t_pt_dataset == 'yes':

    #### need to reorder columns of the various dataframes here to align with 

    #m2_init_abun = multi_func_data_filt_reorder_initial
    # need to use the initial values from MDSINE2 
    m2_init_abun = pd.read_csv('/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data/multi_t_pt_for_glv_fba_init_vals_MDSINE2.csv', index_col=0)
    ### need to reindex the mdsine2 data to align with the experimental data
    ### VERY IMPORTANT ###
    m2_init_abun = m2_init_abun.reindex(multi_func_data_filt_reorder_initial.index.to_list())
    ### need to reorder the columns too 
    m2_init_abun = m2_init_abun[allspecies]
    print(m2_init_abun.index.to_list())


    m2_final_abun_met = multi_func_data_filt_reorder_final[['Time','Butyrate', 'Acetate', 'Lactate', 'Succinate']]
    ### keep all time points b/c using interpolated values from mdsine2
    m2_final_abun_bac = multi_func_data_filt_reorder.drop(columns = ['Butyrate', 'Acetate', 'Lactate', 'Succinate'])


    #### load in the interpolated abundance values from mdsine2
    Venturelli_long_MDSINE2_interpolated_data = np.load('/Users/haydengallo/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data/Venturelli_long_MDSINE2_interpolated_data.npy', allow_pickle=True).item()


else:
    m2_final_abun_met  = m2_final_abun_met[['Butyrate', 'Acetate', 'Lactate', 'Succinate']]
    #m2_final_abun_bac = m2_final_abun_bac.drop(columns = ['Butyrate', 'Acetate', 'Lactate', 'Succinate'])


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


if multi_t_pt_dataset == 'yes':
    met_abun_predict_final_df = met_abun_predict_final_df.drop(columns = ['Time'])
    bac_abun_predict_final_df = bac_abun_predict_final_df.drop(columns= ['Time'])
    #met_abun_predict_final_df.columns = ['Time','EX_but(e)','EX_ac(e)', 'EX_lac_L(e)',  'EX_succ(e)']
else:
    met_abun_predict_final_df.columns = ['EX_but(e)','EX_ac(e)', 'EX_lac_L(e)',  'EX_succ(e)']

### Set media conditions 

# Example medium definition for COBRApy + AGORA
# exchange reactions : uptake flux (mmol/gDW/h)
    ### this is the defined media with additions for the pan genus models


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
    "EX_2dmmq8(e)": -1,
    "EX_acgam(e)": -1,
    "EX_adn(e)": -1,
    "EX_alagln(e)": -1,
    "EX_cobalt2(e)": -1,
    "EX_cu2(e)": -1,
    "EX_fol(e)": -1,
    "EX_galmannan(e)": -1,
    "EX_glygln(e)": -1,
    "EX_glyglu(e)": -1,
    "EX_mqn7(e)": -1,
    "EX_mqn8(e)": -1,
    "EX_nac(e)": -1,
    "EX_nmn(e)": -1,
    "EX_ocdca(e)": -1,
    "EX_pheme(e)": -1,
    "EX_pnto_R(e)": -1,
    "EX_ptrc(e)": -1,
    "EX_q8(e)": -1,
    "EX_sheme(e)": -1,
    "EX_spmd(e)": -1,
    "EX_o2(e)": -10,
    "EX_rbflvrd(e)": -1,
    "EX_zn2(e)": -1,
    "EX_cgly(e)": -1}


### this is the defined media for the pan species models 
'''
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
    "EX_12dgr180(e)": -.1,
    "EX_26dap_M(e)": -.1,
    "EX_adn(e)": -.1,
    "EX_bglc(e)": -.1,
    "EX_cgly(e)": -.1,
    #"EX_chor(e)": -1,
    "EX_cytd(e)": -.1,
    "EX_fol(e)": -.1,
    #"EX_glc3meacp(e)": -1,
    #"EX_glycys(e)": -1,
    "EX_glyphy(e)": -.1,
    "EX_glytyr(e)": -.1,
    "EX_hxan(e)": -.1,
    "EX_mqn7(e)": -.1,
    "EX_mqn8(e)": -.1,
    "EX_nac(e)": -.1,
    "EX_nmn(e)": -.1,
    #"EX_ocdca(e)": -1,
    "EX_q8(e)": -.1,
    "EX_sheme(e)": -.1,
    "EX_spmd(e)": -.1,
    "EX_o2(e)": -.1,
    "EX_rbflv(e)": -.1,
    #"EX_pnto_R(e)": -1,
    #"EX_stys(e)": -1,
    "EX_thm(e)": -.1
    }
'''
### this is the defined media for the strain models 
'''
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
    "EX_26dap_M(e)": -1,
    "EX_2dmmq8(e)": -1,
    "EX_2obut(e)": -1,
    "EX_ade(e)": -1,
    "EX_adn(e)": -1,
    "EX_alagln(e)": -1,
    "EX_alaglu(e)": -1,
    "EX_alahis(e)": -1,
    "EX_alathr(e)": -1,
    "EX_arab_D(e)": -1,
    "EX_bglc(e)": -1,
    "EX_cgly(e)": -1,
    "EX_cit(e)": -1,
    "EX_cytd(e)": -1,
    #"EX_glc3meacp(e)": -1,
    "EX_fol(e)": -1,
    "EX_glycys(e)": -1,
    "EX_glymet(e)": -1,
    "EX_glyphe(e)": -1,
    "EX_glytyr(e)": -1,
    "EX_h2s(e)": -1,
    "EX_hxan(e)": -1,
    "EX_indole(e)": -1,
    "EX_lanost(e)": -1,
    "EX_mqn8(e)": -1,
    "EX_nac(e)": -1,
    "EX_nmn(e)": -1,
    "EX_ocdca(e)": -1,
    "EX_ptrc(e)": -1,
    "EX_q8(e)": -1,
    "EX_sheme(e)": -1,
    "EX_spmd(e)": -1,
    "EX_o2(e)": -10,
    "EX_ura(e)": -1,
    "EX_pnto_R(e)": -1,
    "EX_rib_D(e)": -1,
    "EX_ribflv(e)": -1,
    "EX_thm(e)": -1
    }
'''

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
reverse_order_name_dict = dict(zip(adjusted_names, namedict.keys()))

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




### REPLACE YOUR run_simulations FUNCTION AND EXECUTION CODE WITH THIS ###
### Keep everything else in your script the same ###

def create_plots(FBA_biomass_df, glv_biomass_df, met_pool_over_time, 
                 m2_final_abun_bac_filt, m2_final_abun_met_filt,
                 bac_to_keep_for_inf, scal_fac, t_steps, 
                 plot_dir_path_spec_exp, spec_exp_name, reverse_order_name_dict):
    """Create all plots for a simulation"""
    
    # Prepare plotting dataframes
    FBA_biomass_df_plot = FBA_biomass_df.copy()
    FBA_biomass_df_plot = FBA_biomass_df_plot.melt(ignore_index=False)
    FBA_biomass_df_plot = FBA_biomass_df_plot.reset_index()
    FBA_biomass_df_plot.columns = ['FeatureID', 'time', 'count']
    FBA_biomass_df_plot['count'] = FBA_biomass_df_plot['count'] * scal_fac
    
    glv_biomass_df_plot = glv_biomass_df.copy()
    glv_biomass_df_plot = glv_biomass_df_plot.melt(ignore_index=False)
    glv_biomass_df_plot = glv_biomass_df_plot.reset_index()
    glv_biomass_df_plot.columns = ['FeatureID', 'time', 'count']
    glv_biomass_df_plot['count'] = glv_biomass_df_plot['count'] * scal_fac
    
    # Plot 1: Combined biomass trajectories
    fig, axs = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=FBA_biomass_df_plot, x='time', y='count', hue='FeatureID')
    plot_file_name = plot_dir_path_spec_exp + '/biomass_single_plot_' + spec_exp_name + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Individual species plots
    if multi_t_pt_dataset == 'yes':
        temp_filtered_final_abun = m2_final_abun_bac_filt.drop(columns = ['Time'])
        temp_filtered_final_abun = temp_filtered_final_abun.T.iloc[bac_to_keep_for_inf,:].T
        temp_filtered_final_abun['Time'] = m2_final_abun_bac_filt['Time'].to_list()
        temp_filtered_final_abun_melt = temp_filtered_final_abun.melt(id_vars = ['Time'])
        temp_filtered_final_abun_melt.columns = ['Time', 'Species','Abun']
        #temp_filtered_final_abun_melt['Time'] = pd.to_numeric(temp_filtered_final_abun_melt['Time'], errors='coerce')
        #temp_filtered_final_abun_melt['Abun'] = pd.to_numeric(temp_filtered_final_abun_melt['Abun'], errors='coerce')
        print(temp_filtered_final_abun_melt)
        print(temp_filtered_final_abun_melt.dtypes)

    else:
        m2_final_abun_bac_filt_list = list(m2_final_abun_bac_filt[bac_to_keep_for_inf])
    num_plots = len(FBA_biomass_df.index)
    cols = 5
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=False, sharey = True)
    axes = axes.flatten()
    
    for i, key in enumerate(FBA_biomass_df.index):
        temp_glv = glv_biomass_df_plot[glv_biomass_df_plot['FeatureID'] == key]
        temp_FBA = FBA_biomass_df_plot[FBA_biomass_df_plot['FeatureID'] == key]
        
        sns.lineplot(ax=axes[i], data=temp_glv, x='time', y='count', color='blue', lw=3)
        sns.lineplot(ax=axes[i], data=temp_FBA, x='time', y='count', color='red', lw=3)

        ### here account for multi timepoint dataset 
        if multi_t_pt_dataset == 'yes':
            species_name = reverse_order_name_dict[key]
            temp_for_plot = temp_filtered_final_abun_melt[temp_filtered_final_abun_melt['Species'] == species_name]
            print(temp_for_plot)
            sns.scatterplot(ax=axes[i],data=temp_for_plot, x='Time', y='Abun', 
                       color='green', s=100)
        else:  
            sns.scatterplot(ax=axes[i], x=[t_steps], y=[m2_final_abun_bac_filt_list[i]], 
                       color='green', s=100)
        
        axes[i].set_title(f"{key}")
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Conc')
    
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    
    plot_file_name = plot_dir_path_spec_exp + '/biomass_exp_vs_sim_single_species_' + spec_exp_name + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    plt.close()
    
    # Plot 3: Metabolite concentrations
    met_pool_over_time_df = pd.DataFrame(met_pool_over_time).fillna(0)
    met_pool_over_time_df_melt = met_pool_over_time_df.melt(ignore_index=False)
    met_pool_over_time_df_melt = met_pool_over_time_df_melt.reset_index()
    met_pool_over_time_df_melt.columns = ['Time', 'Metabolite', 'Concentration']
    
    fig, axs = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=met_pool_over_time_df_melt, x='Time', y='Concentration', hue='Metabolite')
    plt.yscale('log')
    plot_file_name = plot_dir_path_spec_exp + '/met_concentration' + spec_exp_name + '.pdf'
    plt.savefig(plot_file_name, bbox_inches="tight")
    plt.close()
    
    # Plot 4: Key metabolites
    filt_mets = ['EX_lac_L(e)', 'EX_succ(e)', 'EX_but(e)', 'EX_ac(e)']
    mets_present = list(set(np.unique(met_pool_over_time_df_melt['Metabolite'].to_list())).intersection(set(filt_mets)))
    
    if len(mets_present) > 0:
        met_pool_over_time_df_melt_filt = met_pool_over_time_df_melt.set_index('Metabolite').loc[mets_present].reset_index()
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        met_palette = {
            "EX_but(e)": "tab:orange",
            "EX_ac(e)": "tab:green",
            "EX_lac_L(e)": "tab:blue",
            "EX_succ(e)": "tab:red"
        }
        
        sns.lineplot(data=met_pool_over_time_df_melt_filt, x='Time', y='Concentration',
                    hue='Metabolite', palette=met_palette, ax=ax)
        
        met_mapping = {
            "Butyrate": "EX_but(e)",
            "Acetate": "EX_ac(e)",
            "Lactate": "EX_lac_L(e)",
            "Succinate": "EX_succ(e)"
        }
        
        for name, bigg_id in met_mapping.items():
            if multi_t_pt_dataset == 'yes':
                # m2_final_abun_met_filt is a DataFrame with 'Time' column
                if name in m2_final_abun_met_filt.columns and bigg_id in met_palette:
                    # Extract time and metabolite values
                    exp_data = m2_final_abun_met_filt[['Time', name]].dropna()
                    if len(exp_data) > 0:
                        color = met_palette[bigg_id]
                        time_points = exp_data['Time'].values
                        values = exp_data[name].values
                        sns.scatterplot(x=time_points, y=values, color=color, s=100, 
                                    edgecolor='black', label=f"{name} (exp)", ax=ax)
            else:
                # m2_final_abun_met_filt is a Series with single endpoint values
                y_val = m2_final_abun_met_filt.get(name, None)
                if y_val is not None and bigg_id in met_palette:
                    color = met_palette[bigg_id]
                    sns.scatterplot(x=[t_steps], y=[y_val], color=color, s=100, 
                                edgecolor='black', label=f"{name} (exp)", ax=ax)
        
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Metabolite", fontsize=12)
        ax.set_title("Simulated vs Experimental Metabolite Concentrations", fontsize=14)
        
        plot_file_name = plot_dir_path_spec_exp + '/mets_of_int_exp_vs_sim' + spec_exp_name + '.pdf'
        plt.savefig(plot_file_name, bbox_inches="tight")
        plt.close()


    ### Plot 5: Species relative abundance overtime in multitimepoint data, exp vs glv vs glvfba
    #temp_filtered_final_abun_melt
    if multi_t_pt_dataset == 'yes':

        temp_wide = temp_filtered_final_abun_melt.pivot(
            index='Species',
            columns='Time',
            values='Abun'
        )


        temp_wide_RA = temp_wide/temp_wide.sum(axis=0)

        glv_biomass_df_RA = glv_biomass_df/glv_biomass_df.sum(axis=0)
        FBA_biomass_df_RA = FBA_biomass_df/FBA_biomass_df.sum(axis=0)

        temp_wide_RA_plot = temp_wide_RA.melt(ignore_index=False)
        temp_wide_RA_plot = temp_wide_RA_plot.reset_index()
        temp_wide_RA_plot.columns = ['FeatureID', 'time', 'count']
        temp_wide_RA_plot['count'] = temp_wide_RA_plot['count'] * scal_fac


        FBA_biomass_df_RA_plot = FBA_biomass_df_RA.melt(ignore_index=False)
        FBA_biomass_df_RA_plot = FBA_biomass_df_RA_plot.reset_index()
        FBA_biomass_df_RA_plot.columns = ['FeatureID', 'time', 'count']
        FBA_biomass_df_RA_plot['count'] = FBA_biomass_df_RA_plot['count'] * scal_fac
        
        glv_biomass_df_RA_plot = glv_biomass_df_RA.melt(ignore_index=False)
        glv_biomass_df_RA_plot = glv_biomass_df_RA_plot.reset_index()
        glv_biomass_df_RA_plot.columns = ['FeatureID', 'time', 'count']
        glv_biomass_df_RA_plot['count'] = glv_biomass_df_RA_plot['count'] * scal_fac

            # Ensure time is numeric
        glv_biomass_df_RA_plot['time'] = pd.to_numeric(glv_biomass_df_RA_plot['time'])
        temp_wide_RA_plot['time'] = pd.to_numeric(temp_wide_RA_plot['time'])
        FBA_biomass_df_RA_plot['time'] = pd.to_numeric(FBA_biomass_df_RA_plot['time'])


        feature_order = glv_biomass_df_RA_plot['FeatureID'].value_counts().index.tolist()
        reversed_order = feature_order[::-1]  # For stackplot

        #palette = sns.color_palette("Set2", n_colors=len(feature_order))  # or use 'husl', 'Set2', etc.
        #color_map = dict(zip(feature_order, palette))

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
            glv_biomass_df_RA_plot
            .pivot(index='time', columns='FeatureID', values='count')
            .fillna(0)
        )[reversed_order]

        # Pivot for stackplot in reversed stacking order
        continuous_pivot_FBA = (
            FBA_biomass_df_RA_plot
            .pivot(index='time', columns='FeatureID', values='count')
            .fillna(0)
        )[reversed_order]

        featureid_translate_list = []
        for i in range(0, len(temp_wide_RA_plot['FeatureID'].to_list())):
            species_name = adjusted_names_dict[temp_wide_RA_plot['FeatureID'].to_list()[i]]
            featureid_translate_list.append(species_name)
        temp_wide_RA_plot['FeatureID'] = featureid_translate_list

        fig, (ax_top, ax_middle, ax_bottom) = plt.subplots(
        3, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]}
        )
        print(temp_wide_RA_plot)
        # Set categorical order for histogram
        temp_wide_RA_plot['FeatureID'] = pd.Categorical(temp_wide_RA_plot['FeatureID'], categories=feature_order, ordered=True)
        print(color_map)
        print(temp_wide_RA_plot)

        # Top: histogram
        sns.histplot(
            data=temp_wide_RA_plot,
            x='time',
            weights='count',
            hue='FeatureID',
            multiple='stack',
            ax=ax_top,
            alpha=0.8,
            binwidth=4,
            palette=color_map
        )
        ax_top.legend_.remove()
        ax_top.set_ylabel('Relative Abundance')
        ax_top.set_title('Experimental Abundance')


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

        legend_elements = [
            Patch(facecolor=color_map[feat], label=feat)
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
        overall_title = 'Relative Species Abun \n' + str(spec_exp_name)
        plt.suptitle(overall_title, x=0.6, fontsize = 20)
        plt.subplots_adjust(right=0.75)
        plot_file_name = plot_dir_path_spec_exp + '/species_RA_' + spec_exp_name + '.pdf'
        plt.savefig(plot_file_name, bbox_inches="tight")
        plt.close()






    ### Plot 6: Metabolites of interest relative abundance overtime in multitimepoint data exp vs glvfba


def run_simulations(i):
    """
    Run a single simulation - modified for proper parallelization
    """
    
    # Make deep copies of models for this worker to avoid conflicts
    models_list = [copy.deepcopy(model) for model in list(correct_model_dict_order.values())]
    
    sim_to_grab = i
    spec_exp_name = m2_init_abun.index.to_list()[sim_to_grab]
    
    plot_dir_path_spec_exp = plot_dir_path_base + '/' + spec_exp_name
    plot_dir = Path(plot_dir_path_spec_exp)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get initial abundance
    ### if using mdsine2 must multiply by whatever constant I used to scale
    init_abun = np.array(m2_init_abun.iloc[sim_to_grab,:].to_list())*10e8
    #init_abun = np.array(m2_init_abun.iloc[sim_to_grab,:].to_list())
    
    # Run gLV simulation
    sol = odeint(gLV, init_abun, time, args=args, tfirst=True)
    
    ### if using mdsine2 must then rescale the output
    sol = sol/10e8
    init_abun = init_abun/10e8

    # Filter data
    if multi_t_pt_dataset == 'yes':
        m2_final_abun_bac_filt = m2_final_abun_bac.loc[spec_exp_name,:]
        m2_final_abun_met_filt = m2_final_abun_met.loc[spec_exp_name,:]
        sol = np.array(Venturelli_long_MDSINE2_interpolated_data[spec_exp_name])
    else:    
        m2_final_abun_bac_filt = m2_final_abun_bac.iloc[sim_to_grab,:]
        m2_final_abun_met_filt = m2_final_abun_met.iloc[sim_to_grab,:]
    
    ### basically can't just filter by bacteria at zero, b/c MDSINE2 doesn't start at values of zero, rather very small numbers like 1e-7 for species that aren't techincally present, so filter by things greater than 1e-5
    if multi_t_pt_dataset == 'yes':
        bac_to_keep_for_inf = list(np.where(init_abun >= 1e-5)[0])
        print('bac to keep',bac_to_keep_for_inf)
        print('index of init abun', spec_exp_name)

    else:
        bac_to_keep_for_inf = list(np.where(init_abun != 0)[0])
    
    # Calculate growth rates
    glv_derived_growth_rates = np.zeros([t_steps+1, 25])
    for j in range(0, t_steps+1):
        glv_derived_growth_rates[j,:] = gLV(j, sol[j,:], paired_growth_matrix=int_matrix, basal_grow=growth_rates)
    
    glv_abun_df = pd.DataFrame(sol)
    
    # Calculate rate array
    rate_array = np.zeros((25, glv_abun_df.shape[0]-1))
    for j in range(0, len(glv_abun_df.T.columns)-1):
        rate_array[:,j] = (glv_abun_df.T.iloc[:,j+1]/glv_abun_df.T.iloc[:,j])-1
    
    rate_df = pd.DataFrame(rate_array).fillna(0)
    
    # Filter for current simulation
    glv_abun_df = glv_abun_df.iloc[:,bac_to_keep_for_inf]
    rate_df = rate_df.iloc[bac_to_keep_for_inf,:]
    init_abun_scaled = init_abun/scal_fac
    init_abun_scaled = list(init_abun_scaled[bac_to_keep_for_inf])
    correct_model_name_order_current_sim = list(np.array(correct_model_name_order)[bac_to_keep_for_inf])
    models_list_current_sim = list(np.array(models_list)[bac_to_keep_for_inf])
    glv_abun_df = glv_abun_df/scal_fac
    
    # Set media conditions for each model
    for j in range(0, len(models_list_current_sim)):
        test_media = make_media(models_list_current_sim[j], defined_media_df)
        models_list_current_sim[j].medium = test_media
        print(models_list_current_sim[j].slim_optimize())
        print(models_list_current_sim[j].medium)
        print(correct_model_name_order_current_sim[j])
    
    # Verify model order
    filt_columns_list_names = m2_init_abun.iloc[:,bac_to_keep_for_inf].columns.to_list()
    checking_order_list = [adjusted_names_dict[name] for name in filt_columns_list_names]
    
    if checking_order_list == correct_model_name_order_current_sim:
        print('yes, the order and identity of GEMs selected for inference is correct')
    else:
        print('no good')
        print('This is correct order of models: ',correct_model_name_order_current_sim)
        print('This is the order given: ', checking_order_list)
        raise ValueError(f"Model order mismatch for simulation {i}")
    
    # Run FBA simulation
    met_pool_over_time, model_abun_dict, mets_used_for_constraint = static_dfba(
        list_model_names=correct_model_name_order_current_sim,
        list_models=models_list_current_sim,
        initial_abundance=init_abun_scaled,
        total_sim_time=t_steps,
        num_t_steps=t_steps,
        glv_out=np.array(glv_abun_df),
        glv_params=None,
        environ_cond=defined_media_df,
        pfba=False,
        MDSINE_rates=rate_df,
        Diet='None',
        output_file_path=plot_dir_path_spec_exp,
        flux_sampling=False,
        host=None,
        random_constraints='No',
        AGORA_models='yes',
	    calc_neg_consumption='yes'
    )

    met_save = plot_dir_path_spec_exp + '/met_pool.npy'
    abun_save = plot_dir_path_spec_exp + '/abun_save.npy'

    np.save(met_save, met_pool_over_time)
    np.save(abun_save, model_abun_dict)
    
    # Extract biomass data
    FBA_biomass = np.zeros([len(model_abun_dict.keys()), t_steps+1])
    glv_biomass = np.zeros([len(model_abun_dict.keys()), t_steps+1])
    
    count = 0
    for key in model_abun_dict:
        FBA_biomass[count,:] = model_abun_dict[key]['fba_biomass']
        glv_biomass[count,:] = model_abun_dict[key]['glv_out'][:t_steps+1]
        count += 1
    
    FBA_biomass_df = pd.DataFrame(FBA_biomass)
    FBA_biomass_df.index = model_abun_dict.keys()
    
    glv_biomass_df = pd.DataFrame(glv_biomass)
    glv_biomass_df.index = model_abun_dict.keys()
    
    # Create all plots
    create_plots(FBA_biomass_df, glv_biomass_df, met_pool_over_time, 
                 m2_final_abun_bac_filt, m2_final_abun_met_filt,
                 bac_to_keep_for_inf, scal_fac, t_steps, 
                 plot_dir_path_spec_exp, spec_exp_name,reverse_order_name_dict)
    
    # Prepare return values - bacteria predictions
    bac_final_predictions = np.zeros(25)  # All 25 species
    FBA_biomass_df_plot = FBA_biomass_df.copy()
    FBA_biomass_df_plot = FBA_biomass_df_plot * scal_fac
    final_abundances = FBA_biomass_df_plot.iloc[:, -1].values
    bac_final_predictions[bac_to_keep_for_inf] = final_abundances
    
    # Get final metabolite concentrations
    met_pool_over_time_df = pd.DataFrame(met_pool_over_time).fillna(0)
    met_pool_over_time_df_melt = met_pool_over_time_df.melt(ignore_index=False)
    met_pool_over_time_df_melt = met_pool_over_time_df_melt.reset_index()
    met_pool_over_time_df_melt.columns = ['Time', 'Metabolite', 'Concentration']
    
    filt_mets = ['EX_lac_L(e)', 'EX_succ(e)', 'EX_but(e)', 'EX_ac(e)']
    # NEW: For multi-timepoint, return DataFrame with specific timepoints only
    if multi_t_pt_dataset == 'yes':
        # Only keep timepoints 16, 32, 48
        timepoints_to_keep = [16, 32, 48]
        
        print(f"\nDEBUG - Experiment {spec_exp_name}:")
        print(f"met_pool_over_time_df shape: {met_pool_over_time_df.shape}")
        print(f"met_pool_over_time_df index (all): {met_pool_over_time_df.index.tolist()}")
        print(f"met_pool_over_time_df columns: {met_pool_over_time_df.columns.tolist()}")
        
        # Filter to only the metabolites of interest
        mets_present = list(set(met_pool_over_time_df.columns).intersection(set(filt_mets)))
        if len(mets_present) > 0:
            # Filter to only the specified timepoints
            met_final = met_pool_over_time_df.loc[timepoints_to_keep, mets_present].copy()
            print(f"Returning {len(met_final)} timepoints ({timepoints_to_keep}) for metabolites: {mets_present}")
            if len(mets_present) > 0:
                print(f"Values for {mets_present[0]}:")
                print(met_final[mets_present[0]])
        else:
            met_final = pd.DataFrame()
    else:
        # Single timepoint - return final value only as before
        met_pool_over_time_df_melt = met_pool_over_time_df.melt(ignore_index=False)
        met_pool_over_time_df_melt = met_pool_over_time_df_melt.reset_index()
        met_pool_over_time_df_melt.columns = ['Time', 'Metabolite', 'Concentration']
        
        mets_present = list(set(np.unique(met_pool_over_time_df_melt['Metabolite'].to_list())).intersection(set(filt_mets)))
        
        if len(mets_present) > 0:
            met_pool_over_time_df_melt_filt = met_pool_over_time_df_melt.set_index('Metabolite').loc[mets_present].reset_index()
            met_pool_over_time_df_melt_filt_final_t_pt = met_pool_over_time_df_melt_filt[met_pool_over_time_df_melt_filt['Time'] == t_steps]
            met_final = met_pool_over_time_df_melt_filt_final_t_pt.set_index('Metabolite')['Concentration']
        else:
            met_final = pd.Series(dtype=float)
    '''
    if len(mets_present) > 0:
        met_pool_over_time_df_melt_filt = met_pool_over_time_df_melt.set_index('Metabolite').loc[mets_present].reset_index()
        met_pool_over_time_df_melt_filt_final_t_pt = met_pool_over_time_df_melt_filt[met_pool_over_time_df_melt_filt['Time'] == t_steps]
        met_final = met_pool_over_time_df_melt_filt_final_t_pt.set_index('Metabolite')['Concentration']
    else:
        met_final = pd.Series(dtype=float)
    '''
    return {
        'sim_idx': i,
        'bac_predictions': bac_final_predictions,
        'met_predictions': met_final,
        'spec_exp_name': spec_exp_name
    }


### MAIN EXECUTION - REPLACE YOUR CURRENT joblib.Parallel CALL WITH THIS ###

if __name__ == "__main__":
    # Determine number of workers (leave 1-2 cores free for system)
    n_jobs = max(1, multiprocessing.cpu_count()) - 2
    
    print(f"Starting {len(m2_init_abun)} simulations using {n_jobs} parallel workers...")
    print(f"Output directory: {plot_dir_path_base}")
    
    # Run parallel simulations with progress bar
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(run_simulations)(i) 
        #for i in tqdm(range(len(m2_init_abun)))
        for i in tqdm(range(2))
    )
        # Ensure metabolite prediction columns exist
    for col_name in met_cols_mapping.values():
        if col_name not in met_abun_predict_final_df.columns:
            met_abun_predict_final_df[col_name] = 0.0

    # ---- metabolite column mapping ----
    met_cols_mapping = {
        "EX_but(e)": "Butyrate",
        "EX_ac(e)": "Acetate",
        "EX_lac_L(e)": "Lactate",
        "EX_succ(e)": "Succinate"
    }

    # Aggregate results into the final DataFrames
    print("\nAggregating results...")
    for result in results:
        sim_idx = result['sim_idx']
        
        # Store bacteria predictions
        bac_abun_predict_final_df.iloc[sim_idx, :] = result['bac_predictions']
        
        # Store metabolite predictions
        met_pred = result['met_predictions']
        met_cols_mapping = {
            'EX_but(e)': 'EX_but(e)',
            'EX_ac(e)': 'EX_ac(e)', 
            'EX_lac_L(e)': 'EX_lac_L(e)',
            'EX_succ(e)': 'EX_succ(e)'
        }
        if multi_t_pt_dataset == 'yes':
            exp_name = result['spec_exp_name']
            met_pred = result['met_predictions']  # DataFrame indexed by time

            # 🔒 get the row POSITIONS that already exist for this experiment
            exp_row_positions = np.where(
                met_abun_predict_final_df.index == exp_name
            )[0]

            # Optional safety check (remove later if you want)
            assert len(exp_row_positions) == len(met_pred), \
                f"Row count mismatch for {exp_name}"

            # 🔹 write one timepoint per row (no broadcasting possible)
            for pos, (time_val, pred_values) in zip(exp_row_positions, met_pred.iterrows()):
                for bigg_id, col_name in met_cols_mapping.items():
                    if bigg_id in pred_values.index:
                        met_abun_predict_final_df.iloc[
                            pos,
                            met_abun_predict_final_df.columns.get_loc(col_name)
                        ] = pred_values[bigg_id]

        else:
            # Single timepoint - met_pred is a Series with final values
            for bigg_id, col_name in met_cols_mapping.items():
                if bigg_id in met_pred.index:
                    met_abun_predict_final_df.loc[result['spec_exp_name'], col_name] = met_pred[bigg_id]


    # Save final results
    met_final_pre_path = plot_dir_path_base + '/final_met_predict_df.csv'
    abun_final_pre_path = plot_dir_path_base + '/final_abun_predict_df.csv'
    
    met_abun_predict_final_df.to_csv(met_final_pre_path)
    bac_abun_predict_final_df.to_csv(abun_final_pre_path)
    
    print(f"\nSimulation complete!")
    print(f"Results saved to:")
    print(f"  - {abun_final_pre_path}")
    print(f"  - {met_final_pre_path}")
