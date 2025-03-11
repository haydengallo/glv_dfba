### Hayden Gallo
### Bucci Lab
### 10/10/24

### Submitting Batch Jobs on HPC

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

'''
### For now need to manually load models
p_copri_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/iPc610.mat')

### For some reason need to change upper bounds of a few reactions of p_copri, just going to set upperbounds to 100, something that wouldln't limit predicted fluxes
### changing how upper bound of glucose exchange because can't change media conditions otherwise, i.e. what the lower bound would be then 
p_copri_model.reactions.get_by_id(id='EX_glc_D(e)').upper_bound = 100
### change upper bound of formate too
p_copri_model.reactions.get_by_id(id='EX_for(e)').upper_bound = 100
### change upper bound of carbon dioxide too
p_copri_model.reactions.get_by_id(id='EX_co2(e)').upper_bound = 100
### Need to change objective function id for this p_copri_model to biomassPan
p_copri_model.reactions.Biomass_BT_v2_norm.id = 'biomassPan'

'''

p_copri_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/dfba_glv/panSpeciesModels_AMANHI_P/panPrevotella_copri.mat')  
eb_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/dfba_glv/panSpeciesModels_AMANHI_P/panEubacterium_limosum.mat') 
dorea_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/panDorea_longicatena.mat')

models = [eb_model, p_copri_model, dorea_model]
#models = [dorea_model, p_copri_model]


rcm_add = [['EX_adocbl(e)','0.01'],['EX_btn(e)','0.01'],['EX_h2s(e)','0.01'],['EX_ribflv(e)', '0.01'],['EX_thymd(e)', '0.01'],['EX_thm(e)', '0.01'],['EX_spmd(e)', '0.01'],['EX_sheme(e)', '0.01'],['EX_q8(e)', '0.01'],['EX_pheme(e)', '0.01'],['EX_fol(e)', '0.01'],['EX_2dmmq8(e)', '0.01'],['EX_26dap_M(e)', '0.01'],['EX_cobalt2(e)', '0.01'],['EX_cu2(e)', '0.01'],['EX_fe3(e)', '0.01'],['EX_mn2(e)', '0.01'],['EX_zn2(e)', '0.01'],['EX_na1(e)', '49.13878039'],['EX_cl(e)', '35.42367951'],['EX_n2(e)', '221.9285714'],['EX_ca2(e)', '0.083087978'],['EX_fe2(e)', '0.009273883'],['EX_mg2(e)', '0.216827813'],['EX_k(e)', '10.45186108'],['EX_so4(e)', '0.913009078'],['EX_pi(e)', '4.844687796'],['EX_ala_L(e)', '16.70146138'],['EX_arg_L(e)', '5.384491745'],['EX_asn_L(e)', '0.908272088'],['EX_asp_L(e)', '9.083395943'],['EX_glu_L(e)', '17.32518669'],['EX_gly(e)', '25.44393675'],['EX_his_L(e)', '2.120432852'],['EX_ile_L(e)', '6.174957118'],['EX_leu_L(e)', '9.323422908'],['EX_lys_L(e)', '7.168752993'],['EX_met_L(e)', '1.702321591'],['EX_phe_L(e)', '5.193956124'],['EX_pro_L(e)', '11.3782441'],['EX_ser_L(e)', '3.882275699'],['EX_thr_L(e)', '2.837474815'],['EX_trp_L(e)', '0.56309339'],['EX_tyr_L(e)', '1.357683329'],['EX_val_L(e)', '7.89599481'], ['EX_Lcystin(e)', '0.108201688'], ['EX_gln_L(e)', '0.109479562'],['EX_glc_D(e)', '27.75372455'],['EX_cys_L(e)', '2.846894039'],['EX_M02144(e)', '2.846894039'],['EX_h2o(e)', '55509.29781'],['EX_h(e)', '0.000158489']]

rcm_add = pd.DataFrame(rcm_add)

rcm_add.columns = ['reaction','fluxValue']
rcm_add['fluxValue'] =  np.double(rcm_add['fluxValue'])

# Create an argument parser
parser = argparse.ArgumentParser(description='Process job parameters and save path.')
parser.add_argument('--params', type=str, required=True, help='Comma-separated list of parameters')
parser.add_argument('--job_save_dir', type=str, required=True, help='Path to save the results')
parser.add_argument('--model_names', type=str, required=True, help='List of model names')
parser.add_argument('--init_abun', type=str, required=True, help='Species initial abundances')
#parser.add_argument('--time', type=str, required=True, help='Species initial abundances')

# Parse the arguments
args = parser.parse_args()

# Convert the comma-separated string back into a NumPy array
params = np.fromstring(args.params, sep=',')
print(params)
init_abun = np.array(args.init_abun.split(','), dtype=np.float64)
model_names = args.model_names.split(',')
#int_time = args.time.split(',')

### when transporting init_abun and time_steps from the run.py script, it is reencoded incorrectly as UTF-5 instead of float64 for some reason so need to figure that out 
#init_abun = [.002, .002]

#time_integration = np.arange(0, 461, 1)
## time integration over 24 hours, but 1440 minutes
### need to change time_integration from using np.arange to np.linspace

#time_integration = np.arange(0, 289, 1)

time_integration = np.linspace(0, 24,577)


# total sim time should be in hours....
#glv_out = odeint(generalized_gLV, y0 = init_abun, t=time_integration, args = (params,))
glv_out = odeint(multi_spec_gLV, y0 = init_abun, t=time_integration, args  =(params,))
#print(glv_out)

#### here I need to specify the index because DO would be in [:,2]

met_pool_over_time, model_abun_dict = static_dfba(list_model_names=model_names,list_models=models, initial_abundance=init_abun, total_sim_time=(24), num_t_steps=(576), glv_out=glv_out, glv_params=params, environ_cond= rcm_add, pfba=True)

met_pool_filename = "met_pool_over_time.npy"
model_abun_filename = "model_abun_dict.npy"

# Save the results to the specified save path
np.save(os.path.join(args.job_save_dir, met_pool_filename), met_pool_over_time)
np.save(os.path.join(args.job_save_dir, model_abun_filename), model_abun_dict)
