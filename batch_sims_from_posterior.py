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

### For now need to manually load models
p_copri_model = cobra.io.load_matlab_model('/Users/haydengallo/cobratoolbox/panSpeciesModels/panPrevotella_copri.mat')  
eb_model = cobra.io.load_matlab_model('/Users/haydengallo/cobratoolbox/panSpeciesModels/panEubacterium_limosum.mat') 

models = [eb_model, p_copri_model]

# Create an argument parser
parser = argparse.ArgumentParser(description='Process job parameters and save path.')
parser.add_argument('--params', type=str, required=True, help='Comma-separated list of parameters')
parser.add_argument('--job_save_dir', type=str, required=True, help='Path to save the results')
parser.add_argument('--model_names', type=str, required=True, help='List of model names')
parser.add_argument('--init_abun', type=str, required=True, help='Species initial abundances')

# Parse the arguments
args = parser.parse_args()

# Convert the comma-separated string back into a NumPy array
params = np.fromstring(args.params, sep=',')
init_abun = args.init_abun.split(',')
model_names = args.model_names.split(',')

glv_out = odeint(generalized_gLV, y0 = init_abun, t=time, args = (params,))


met_pool_over_time, model_abun_dict = static_dfba(list_model_names=model_names,list_models=models, initial_abundance=init_abun, total_sim_time=(460), num_t_steps=(460), glv_out=glv_out, glv_params=params_samp, environ_cond= rcm_add, pfba=True)

met_pool_filename = "met_pool_over_time.npy"
model_abun_filename = "model_abun_dict.npy"

# Save the results to the specified save path
np.save(os.path.join(args.job_save_dir, met_pool_filename), met_pool_over_time)
np.save(os.path.join(args.job_save_dir, model_abun_filename), model_abun_dict)
