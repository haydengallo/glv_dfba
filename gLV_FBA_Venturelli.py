#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Hayden Gallo
### 9/29/25
### Bucci Lab
### Running gLV-FBA on Venturelli data


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


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


#################################################################################################################################################################
### this is used to surpress all logging from loading in the kbase models with cobra, such that they don't get added to the glv_fba log file and overcrowd it ### 
logging.getLogger("cobra").setLevel(logging.ERROR)
#################################################################################################################################################################


# In[4]:


### define generalized Lotka-Volterra function 
def gLV(t, init_abun, paired_growth_matrix, basal_grow):

    return ((np.dot(init_abun, paired_growth_matrix) + basal_grow) * init_abun)


# In[5]:


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


# In[6]:


data_dir = '/Users/haydengallo/UMass_Dropbox/UMass Medical School Dropbox/Hayden Gallo/Bucci_Lab/glv_FBA/Venturelli_data'


# In[7]:


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

m2_final_abun


# In[8]:


m2_final_abun = m2_final_abun.drop(columns = ['Time'])
m2_final_abun_met  = m2_final_abun[['Butyrate', 'Acetate', 'Lactate', 'Succinate']]
m2_final_abun_bac = m2_final_abun.drop(columns = ['Butyrate', 'Acetate', 'Lactate', 'Succinate'])


# In[9]:


time = np.linspace(0,48,49)
args = (int_matrix, growth_rates)


# In[10]:


### Sanity check to make sure things are good to start simulating 


final_prediction = np.zeros([len(m2_init_abun), 25])

for i in range(0, len(m2_init_abun)):

    init_abun = np.array(m2_init_abun.iloc[i,:].to_list())
    sol = odeint(gLV,  init_abun,time, args = args, tfirst = True)
    final_prediction[i,:] = sol[-1,:]

final_prediction


# In[11]:


#list(cobra_models.keys())

count = 0

for i in range(0, len(m2_init_abun)):

    index = list(m2_init_abun.index)[i]
    if index == 'CA':
        print(count)
    count +=1

m2_init_abun.loc['CA']


# In[12]:


m2_init_abun


# In[13]:


### load in all of the GEMs

### load the cobra models into memory i guess

#cobra_models_dir_path = data_dir + '/AGORA_GEMs'
cobra_models_dir_path = data_dir + '/panGenusModels_Venturelli'

cobra_models_dir = Path(cobra_models_dir_path)

# Make the data and validation Study objects
cobra_models = sorted(cobra_models_dir.glob('*.mat'))
cobra_models = {f.stem : f for f in cobra_models}

### Just loading the models needed in 

loaded_models = {}

#count = 0

for key in cobra_models:
    #if count == 1:
    #    break
    #print(key.split('_'))
    #model_name = key.split('_')[0] + '_' + key.split('_')[1]
    model_name = key.split('.')[0]
    model_name = model_name[3:]
    #print(model_name)
    model = cobra.io.load_matlab_model(cobra_models[key])
    loaded_models[model_name] = model
    #count+=1



# In[14]:


#loaded_models['Anaerostipes_caccae'].slim_optimize()


# In[15]:


loaded_models


# In[16]:


loaded_models['Anaerostipes'].reactions


# In[17]:


adjusted_names = []

for key in namedict:
    temp = namedict[key].split('_')
    temp_name = temp[0] + '_' + temp[1]
    adjusted_names.append(temp_name)

adjusted_names_dict = dict(zip(namedict.keys(), adjusted_names))


# In[18]:


adjusted_names_dict


# In[19]:


correct_model_dict_order = {}
correct_model_name_order = []


for i in allspecies:
    model_to_grab = adjusted_names_dict[i]
    model_to_grab_genus = model_to_grab.split('_')[0]
    print(model_to_grab_genus)
    correct_model_dict_order[model_to_grab] = loaded_models[model_to_grab_genus]
    correct_model_name_order.append(model_to_grab)


# In[20]:


correct_model_dict_order['Eubacterium_rectale']


# In[21]:


correct_model_dict_order


# In[22]:


### Testing one simulation 

sim_to_grab = 1


init_abun = np.array(m2_init_abun.iloc[sim_to_grab,:].to_list())
sol = odeint(gLV,  init_abun,time, args = args, tfirst = True)
sol.shape


m2_final_abun_bac_filt = m2_final_abun_bac.iloc[sim_to_grab,:]
m2_final_abun_met_filt = m2_final_abun_met.iloc[sim_to_grab,:]


# In[23]:


len(m2_init_abun)


# In[24]:


bac_to_keep_for_inf = list(np.where(init_abun != 0)[0])
bac_to_keep_for_inf


# In[25]:


glv_derived_growth_rates = np.zeros([49,25])

for i in range(0,49):
    glv_derived_growth_rates[i,:] = gLV(i, sol[i,:], paired_growth_matrix=int_matrix, basal_grow=growth_rates)

glv_derived_growth_rates


# In[26]:


glv_derived_growth_rates_df = pd.DataFrame(glv_derived_growth_rates)
glv_derived_growth_rates_df


# In[27]:


glv_abun_df = pd.DataFrame(sol)
glv_abun_df


# In[28]:


models_list = list(correct_model_dict_order.values())


# In[29]:


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





# In[30]:


defined_media_df = pd.DataFrame.from_dict(defined_media, orient='index')
defined_media_df = defined_media_df.reset_index()
defined_media_df.columns = ['reaction', 'fluxValue']
defined_media_df['fluxValue'] = -1.0*defined_media_df['fluxValue']
defined_media_df


# In[31]:


defined_media_df_filt = defined_media_df[defined_media_df['fluxValue'] != -0.001]
defined_media_df_filt


# In[32]:


with open('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/gLV_FBA_test_Kennedy_et_al_2025/raw_data/BiGG_to_MSID.json') as f:
    bigg_to_modelseed = json.load(f)
    print(bigg_to_modelseed)


# In[33]:


defined_media_bigg = {
    "ca2":  -1.290526021,
    "na1": -47.34272139,
    "cu2": -0.021265311,
    "cu":  -0.021265311,
    "so4": -10.22721245,
    "pydx": -0.009822218,
    "thymd": -0.021,
    "xan":  -0.024981921,
    "fol":  -0.002402832,
    "orot": -0.064061499,
    "k":    -6.617592674,
    "cobalt2": -0.055,
    "no3":  -0.109323669,
    "fe3":  -0.066,
    "fe2":  -0.066,
    "mg2":  -4.380495746,
    "mn2":  -0.33,
    "mobd": -0.004856255,
    "slnt": -0.000578243,
    "tungs": -0.003403444,
    "cl":  -14.45186396,
    "ni2": -0.01543217,
    "zn2": -0.061931009,
    "pnto__R": -0.000419695,
    "cbl1": -1.47561E-06,
    "pydxn": -0.000972583,
    "h": -0.001565646,
    "ribflv": -0.000531406,
    "thf": -2.80628E-06,
    "thm": -0.000593064,
    "4abz": -0.073072918,
    "pydam": -0.021,
    "nh4": -9.347366847,
    "pi": -6.61327063,
    "ncam": -0.034392401,
    "pheme": -0.015338835,
    "csn": -5.94059E-05,
    "gua": -5.95514E-05,
    "ade": -5.99423E-05,
    "ura": -5.88829E-05,
    "inost": -6.272202487,
    "btn": -0.040952069,
    "ala__L": -5.3,
    "arg__L": -21.81400689,
    "asn__L": -2.6,
    "asp__L": -0.4,
    "cys__L": -8.4,
    "glu__L": -0.662721893,
    "gln__L": -2.7,
    "his__L": -1,
    "ile__L": -1.6,
    "leu__L": -3.6,
    "lys__L": -2.4,
    "met__L": -0.84,
    "phe__L": -4.5,
    "pro__L": -5.9,
    "ser__L": -6.4,
    "thr__L": -1.9,
    "trp__L": -0.73,
    "val__L": -2.8,
    "tyr__L": -3.201059661,
    "mops": -71.68003181,
    "hco3": -47.6150797,
    "arab__L": -21.31486045,
    "glc__D": -24.97835209,
    "lac__L": -28.30817052,
    "malt": -4.382120947,
    "h2o": -55.50645091}


# In[34]:


defined_media_bigg_df = pd.DataFrame.from_dict(defined_media_bigg, orient='index')
defined_media_bigg_df = defined_media_bigg_df.reset_index()
defined_media_bigg_df.columns = ['reaction', 'fluxValue']
#defined_media_df['fluxValue'] = -100.0*defined_media_df['fluxValue']
defined_media_bigg_df


# In[35]:


kbase_cpds = []

for i in range(0, len(defined_media_bigg_df)):
    if defined_media_bigg_df.loc[i,'reaction'] == 'mops':
        kbase_cpds.append('cpd11575')
    else:    
        kbase_cpds.append(bigg_to_modelseed[defined_media_bigg_df.loc[i,'reaction']])


defined_media_bigg_df['reaction'] = kbase_cpds
defined_media_bigg_df


# In[36]:


defined_media_bigg_df['minflux'] = -100.0
defined_media_bigg_df['fluxValue'] = defined_media_bigg_df['fluxValue']*-1.0
defined_media_bigg_df.columns = ['compounds', 'maxflux', 'minflux']
defined_media_bigg_df


# In[37]:


#defined_media_bigg_df.to_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/Venturelli_data/kbase_media.tsv')


# In[38]:


plot_dir_path = '/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/Venturelli_data/testing'

plot_dir = Path(plot_dir_path)
os.makedirs(plot_dir, exist_ok=True)


# In[39]:


rate_array = np.zeros((25,glv_abun_df.shape[0]-1))
rate_array.shape


# In[40]:


for i in range(0,(len(glv_abun_df.T.columns)-1)):

    rate_array[:,i] = (glv_abun_df.T.iloc[:,i+1]/glv_abun_df.T.iloc[:,i])-1


rate_df = pd.DataFrame(rate_array).fillna(0)


# In[41]:


A_caccae = cobra.io.read_sbml_model('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/Venturelli_data/AGORA_GEMs_alt/Anaerostipes_caccae_ASM2514600v1_genomic.fna_assembly.RAST.mdl.sbml')
B_adolescentis = cobra.io.read_sbml_model('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/Venturelli_data/AGORA_GEMs_alt/Bifidobacterium_adolescentis_ASM1042v1_genomic.fna_assembly.RAST.mdl.sbml')

MS_defined_media = pd.read_csv('/Users/haydengallo/UMass_Dropbox/Dropbox (UMass Medical School)/Bucci_Lab/glv_FBA/Venturelli_data/kbase_media.tsv', index_col = 0)
MS_defined_media


# In[42]:


reaction = []

for i in range(0, len(MS_defined_media)):
    reaction.append('EX_' + MS_defined_media['compounds'][i] + '_b')


# In[43]:


MS_defined_media['compounds'] = reaction
MS_defined_media['maxflux'] = MS_defined_media['maxflux']

MS_defined_media = MS_defined_media[['compounds','maxflux']]
MS_defined_media.columns = ['reaction', 'fluxValue']
MS_defined_media


# In[44]:


glv_abun_df = glv_abun_df.iloc[:,bac_to_keep_for_inf]
glv_abun_df


# In[45]:


rate_df = rate_df.iloc[bac_to_keep_for_inf,:]


# In[46]:


#models_list = [A_caccae, B_adolescentis]

models_list


# In[47]:


#correct_model_name_order = ['A_caccae', 'B_adolescentis']


# In[48]:


init_abun[bac_to_keep_for_inf]


# In[49]:


init_abun = init_abun/10
init_abun = list(init_abun[bac_to_keep_for_inf])
init_abun


# In[50]:


list(np.array(correct_model_name_order)[bac_to_keep_for_inf])


# In[51]:


correct_model_name_order=list(np.array(correct_model_name_order)[bac_to_keep_for_inf])


# In[52]:


for i in range(0, len(models_list)):


    test_media = make_media(models_list[i], defined_media_df)
    models_list[i].medium = test_media
    #models_list[i].optimize()
    print(models_list[i].slim_optimize())


# In[53]:


models_list=list(np.array(models_list)[bac_to_keep_for_inf])


# In[54]:


bac_to_keep_for_inf


# In[55]:


glv_abun_df = glv_abun_df/10


# In[56]:


defined_media_df.head(25)


# In[57]:


#met_pool_over_time, model_abun_dict, mets_used_for_constraint = static_dfba(list_model_names=correct_model_name_order, list_models=models_list, initial_abundance=init_abun, total_sim_time=48, num_t_steps=48, glv_out=np.array(glv_abun_df), glv_params=None, environ_cond=MS_defined_media, pfba=True, MDSINE_rates=rate_df, Diet=None, output_file_path = plot_dir_path, flux_sampling=False, host = None, random_constraints = 'No', AGORA_models = 'yes')#interpolated_met_values)
met_pool_over_time, model_abun_dict, mets_used_for_constraint = static_dfba(list_model_names=correct_model_name_order, list_models=models_list, initial_abundance=init_abun, total_sim_time=48, num_t_steps=48, glv_out=np.array(glv_abun_df), glv_params=None, environ_cond=defined_media_df, pfba=False, MDSINE_rates=rate_df, Diet=None, output_file_path = plot_dir_path, flux_sampling=False, host = None, random_constraints = 'No', AGORA_models = 'yes')#interpolated_met_values)


# In[ ]:


model_abun_dict['Anaerostipes_caccae']['glv_out']


# In[ ]:


len(model_abun_dict['Anaerostipes_caccae']['fba_biomass'])


# In[ ]:


FBA_biomass = np.zeros([len(model_abun_dict.keys()), len(model_abun_dict['Anaerostipes_caccae']['fba_biomass'])])
glv_biomass = np.zeros([len(model_abun_dict.keys()), len(model_abun_dict['Anaerostipes_caccae']['glv_out'])])

# %%
### Convert FBA abun output to relative abundance 
count = 0
for key in model_abun_dict:
    FBA_biomass[count,:] = model_abun_dict[key]['fba_biomass']
    glv_biomass[count,:] = model_abun_dict[key]['glv_out']
    count+=1

FBA_biomass_df = pd.DataFrame(FBA_biomass)
FBA_biomass_df.index = model_abun_dict.keys()

index_to_filter_by = FBA_biomass_df.index
FBA_biomass_df

glv_biomass_df = pd.DataFrame(glv_biomass)
glv_biomass_df.index = model_abun_dict.keys()

index_to_filter_by = glv_biomass_df.index
glv_biomass_df



# In[ ]:


FBA_biomass_df_plot = pd.DataFrame(FBA_biomass)
FBA_biomass_df_plot.index = model_abun_dict.keys()




FBA_biomass_df_plot = FBA_biomass_df_plot.melt(ignore_index=False)
FBA_biomass_df_plot = FBA_biomass_df_plot.reset_index()
FBA_biomass_df_plot.columns = ['FeatureID','time', 'count']
#FBA_biomass_df_plot['time'] = (FBA_biomass_df_plot['time']/time_scaler)-3
FBA_biomass_df_plot['count'] = FBA_biomass_df_plot['count']*10




glv_biomass_df_plot = glv_biomass_df.melt(ignore_index=False)
glv_biomass_df_plot = glv_biomass_df_plot.reset_index()
glv_biomass_df_plot.columns = ['FeatureID','time', 'count']
#FBA_biomass_df_plot['time'] = (FBA_biomass_df_plot['time']/time_scaler)-3
glv_biomass_df_plot['count'] = glv_biomass_df_plot['count']*10

fig, axs = plt.subplots(figsize= (15,10))
sns.lineplot(data=FBA_biomass_df_plot, x='time', y='count', hue = 'FeatureID')
#plt.yscale('log')
#plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_abundances_over_time_test_' + str(test_num) + '.pdf'
#plt.savefig(plot_file_name, bbox_inches="tight")
plt.show()


# In[ ]:


glv_biomass_df_plot[glv_biomass_df_plot['time'] == 48]


# In[ ]:


m2_final_abun_bac_filt = list(m2_final_abun_bac_filt[bac_to_keep_for_inf])
m2_final_abun_bac_filt


# In[ ]:


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
    sns.lineplot(ax = axes[i], data=temp_glv, x = 'time', y = 'count', color = 'blue')
    
    sns.lineplot(ax = axes[i], data=temp_FBA, x = 'time', y = 'count', color = 'red')
    sns.scatterplot(ax=axes[i], x=[48], y=m2_final_abun_bac_filt[i], color = 'green', s = 100)

    axes[i].set_title(f"{key}")
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Conc')
    #axes[i].set_yscale('log')
    #axes[i].legend_.remove()

# Hide any empty subplots if the number of plots is not a perfect square
for j in range(num_plots, len(axes)):
    fig.delaxes(axes[j])


#plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_mets_exp_vs_sim_individual_scatterplots_smoothed_predictions' + str(test_num) + '.pdf'
#plt.savefig(plot_file_name, bbox_inches="tight")


# In[ ]:


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

# %
'''
total_abun_MDSINE = pd.DataFrame(bi_hourly_resolution_latent_traj.sum(axis=0)).reset_index()
total_abun_MDSINE.columns = ['time', 'abun']
total_abun_MDSINE['time'] = pd.to_numeric(total_abun_MDSINE['time'])-3
total_abun_MDSINE.head()

# %%
fig, axs = plt.subplots(figsize= (15,10))
sns.lineplot(data=total_abun_MDSINE, x='time', y='abun')
plt.yscale('log')
plt.show()
'''


# In[ ]:


np.unique(met_pool_over_time_df_melt['Metabolite'].to_list())


# In[ ]:


plot_dir = Path(plot_dir_path)
os.makedirs(plot_dir, exist_ok=True)

fig, axs = plt.subplots(figsize= (15,10))
sns.lineplot(data=met_pool_over_time_df_melt, x='Time', y='Concentration', hue = 'Metabolite')
plt.yscale('log')
#plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_metabolites_over_time_test_' + str(test_num) + '.pdf'
#plt.savefig(plot_file_name, bbox_inches="tight")
plt.show()


# In[ ]:


filt_mets = ['EX_lac_L(e)', 'EX_succ(e)','EX_but(e)', 'EX_ac(e)']

mets_present = list(set(np.unique(met_pool_over_time_df_melt['Metabolite'].to_list())).intersection(set(filt_mets)))
#filt_mets = ['EX_lac_L(e)','EX_but(e)', 'EX_ac(e)']
#filt_mets = ['EX_cpd00029_b', 'EX_cpd00159_b']

#filt_mets = ['EX_lac_L(e)']

met_pool_over_time_df_melt_filt = met_pool_over_time_df_melt.set_index('Metabolite').loc[mets_present].reset_index()


met_pool_over_time_df_melt_filt



# In[ ]:


#m2_final_abun_bac_filt = list(m2_final_abun_bac_filt[bac_to_keep_for_inf])

m2_final_abun_met_filt


# In[ ]:


met_pool_over_time_df_melt_filt[met_pool_over_time_df_melt_filt['Time'] == 48]


# In[ ]:


# Create the base line plot
fig, ax = plt.subplots(figsize=(15, 10))
palette = sns.color_palette("tab10")  # or your preferred palette

# Let seaborn assign colors automatically by hue
sns.lineplot(
    data=met_pool_over_time_df_melt_filt,
    x='Time',
    y='Concentration',
    hue='Metabolite',
    palette=palette,
    ax=ax
)

# Extract color mapping from the lineplot legend
handles, labels = ax.get_legend_handles_labels()
color_map = dict(zip(labels, [h.get_color() for h in handles]))

# Example: map your metabolites (Butyrate, Acetate, etc.) to BiGG IDs
met_mapping = {
    "Butyrate": "EX_but(e)",
    "Acetate": "EX_ac(e)",
    "Lactate": "EX_lac_L(e)",
    "Succinate": "EX_succ(e)"
}

# Scatter for each metabolite using the same color as the corresponding line
for i, (name, bigg_id) in enumerate(met_mapping.items()):
    color = color_map[bigg_id]
    y_val = m2_final_abun_met_filt.iloc[i]
    sns.scatterplot(x=[48], y=[y_val], color=color, s=100, label=name, ax=ax)

# Adjust legend if needed
ax.legend(title="Metabolite", fontsize=12)


# In[ ]:


fig, axs = plt.subplots(figsize= (15,10))
sns.lineplot(data=met_pool_over_time_df_melt_filt, x='Time', y='Concentration', hue = 'Metabolite')


sns.scatterplot(x = [48], y = m2_final_abun_met_filt.to_list()[0], color = 'orange', label = 'Butyrate')
sns.scatterplot(x = [48], y = m2_final_abun_met_filt.to_list()[1], color = 'green', label = 'Acetate')
sns.scatterplot(x = [48], y = m2_final_abun_met_filt.to_list()[2], color = 'blue', label = 'Lactate')
sns.scatterplot(x = [48], y = m2_final_abun_met_filt.to_list()[3], color = 'red', label = 'Succinate')

#plot_file_name = plot_dir_path + '/Subject_' + str(subject_to_plot) + '_metabolites_over_time_test_filt' + str(test_num) + '.pdf'
#plt.savefig(plot_file_name, bbox_inches="tight")
plt.yscale('log')
plt.show()


# In[ ]:


met_endpoint_dict = {}
met_endpoint_dict['butyrate'] = {}
met_endpoint_dict['succinate'] = {}
met_endpoint_dict['lactate'] = {}
met_endpoint_dict['acetate'] = {}

met_endpoint_dict['butyrate']['predict'] = []
met_endpoint_dict['succinate']['predict'] = []
met_endpoint_dict['lactate']['predict'] = []
met_endpoint_dict['acetate']['predict'] = []

met_endpoint_dict['butyrate']['exp'] = []
met_endpoint_dict['succinate']['exp'] = []
met_endpoint_dict['lactate']['exp'] = []
met_endpoint_dict['acetate']['exp'] = []


met_endpoint_dict


# In[ ]:


m2_final_abun_met_filt


# In[ ]:


bac_abun_predict_final_df = m2_final_abun_bac.copy()
bac_abun_predict_final_df

met_abun_predict_final_df = m2_final_abun_met.copy()
met_abun_predict_final_df


# In[ ]:


bac_abun_predict_final_df[:] = 0
met_abun_predict_final_df[:] = 0 


# In[ ]:


bac_abun_predict_final_df.iloc[7,bac_to_keep_for_inf] = FBA_biomass_df_plot[FBA_biomass_df_plot['time'] == 48]['count'].to_list()

bac_abun_predict_final_df.head(10)


# In[ ]:


met_abun_predict_final_df.columns = ['EX_but(e)','EX_ac(e)', 'EX_lac_L(e)',  'EX_succ(e)']
met_abun_predict_final_df


# In[ ]:


met_pool_over_time_df_melt_filt_final_t_pt = met_pool_over_time_df_melt_filt[met_pool_over_time_df_melt_filt['Time'] == 48].set_index('Metabolite').reindex(met_abun_predict_final_df.columns)
met_pool_over_time_df_melt_filt_final_t_pt


# In[ ]:


met_abun_predict_final_df.iloc[7,:] = met_pool_over_time_df_melt_filt_final_t_pt['Concentration'].to_list()
met_abun_predict_final_df.head(10)


# In[ ]:


m2_init_abun.index.to_list()[sim_to_grab]


# In[ ]:




