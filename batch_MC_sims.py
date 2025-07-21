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
from sklearn.metrics import *



p_copri_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/dfba_glv/panSpeciesModels_AMANHI_P/panPrevotella_copri.mat')  
eb_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/dfba_glv/panSpeciesModels_AMANHI_P/panEubacterium_limosum.mat') 
dorea_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/panDorea_longicatena.mat')

models = [eb_model, p_copri_model, dorea_model]


# Create an argument parser
parser = argparse.ArgumentParser(description='Process job parameters and save path.')
parser.add_argument('--params', type=str, required=True, help='Comma-separated list of parameters')
parser.add_argument('--sim_save_dir', type=str, required=True, help='Path to save the results')
parser.add_argument('--model_names', type=str, required=True, help='List of model names')
parser.add_argument('--init_abun', type=str, required=True, help='Species initial abundances')
parser.add_argument('--sim_num', type=str, required=True, help='Sim number')
parser.add_argument('--param_df_path', type=str, required=True, help='Path to df containing met values')
parser.add_argument('--rmse_path', type=str, required=True, help='Path to rmse folder')

# Parse the arguments
args = parser.parse_args()

## load in mets_draw_df

#mets_draw_df = pd.DataFrame(mets_draw_array, index = mets_draw_array['index'])
mets_draw_df = pd.read_pickle(args.param_df_path)


# Convert the comma-separated string back into a NumPy array
params = np.fromstring(args.params, sep=',')
print(params)
init_abun = np.array(args.init_abun.split(','), dtype=np.float64)
model_names = args.model_names.split(',')

sim_num = int(args.sim_num)

rcm_add = pd.DataFrame(zip(mets_draw_df.index, mets_draw_df.iloc[:,sim_num]))
rcm_add.columns = ['reaction','fluxValue']
rcm_add['fluxValue'] =  np.double(rcm_add['fluxValue'])

time_integration = np.linspace(0, 24,577)

glv_out = odeint(multi_spec_gLV, y0 = init_abun, t=time_integration, args  =(params,))

met_pool_over_time, model_abun_dict = static_dfba(list_model_names=model_names,list_models=models, initial_abundance=init_abun, total_sim_time=(24), num_t_steps=(576), glv_out=glv_out, glv_params=params, environ_cond= rcm_add, pfba=True)

met_pool_filename = "met_pool_over_time.npy"
model_abun_filename = "model_abun_dict.npy"







# Save the results to the specified save path
np.save(os.path.join(args.sim_save_dir, met_pool_filename), met_pool_over_time)
np.save(os.path.join(args.sim_save_dir, model_abun_filename), model_abun_dict)




### ok here load cerillo data and get set for batching out 

cerillo_test_data = pd.read_csv('/home/hayden.gallo-umw/data/dfba_glv/cerillo_data_comp_analysis.csv', index_col=0)
averaged_cerillo_df_for_testing = cerillo_test_data.unstack().reset_index()
averaged_cerillo_df_for_testing.columns = ['Group_together', 'Time', 'OD']
cerillo_test_data = averaged_cerillo_df_for_testing

#mono culture data of EB

EB_mono_culture_df = cerillo_test_data[(cerillo_test_data['Group_together'] == 'EUB/EUB')]
EB_co_culture_PC = cerillo_test_data[(cerillo_test_data['Group_together'] == 'EUBwPC')]
PC_co_culture_EB = cerillo_test_data[(cerillo_test_data['Group_together'] == 'PCwEUB')]

counter = 0 

test_list = []
pc_list = []

if np.count_nonzero(init_abun == 0) == 1:
    for i in range(0, len(model_abun_dict['eb']['fba_biomass'])):
        if counter == 280:
            break
        if i%2 == 0:
            test_list.append(model_abun_dict['eb']['fba_biomass'][i])
            pc_list.append(model_abun_dict['p_copri']['fba_biomass'][i])
            counter += 1

    rmse_list =[]
    rmse_eb = root_mean_squared_error(EB_co_culture_PC['OD'].to_list(), test_list)
    rmse_list.append(rmse_eb)        
    rmse_pc = root_mean_squared_error(PC_co_culture_EB['OD'].to_list(), pc_list)
    rmse_list.append(rmse_pc)   

else:

    for i in range(0, len(model_abun_dict['eb']['fba_biomass'])):
        if counter == 280:
            break
        if i%2 == 0:
            test_list.append(model_abun_dict['eb']['fba_biomass'][i])
            counter += 1

    rmse_list =[]
    rmse = root_mean_squared_error(EB_mono_culture_df['OD'].to_list(), test_list)
    rmse_list.append(rmse)


rmse_file_name = 'rmse_' + 'sim_' + str(sim_num) + '.csv'


np.savetxt(os.path.join(args.rmse_path, rmse_file_name), rmse_list, delimiter=',')