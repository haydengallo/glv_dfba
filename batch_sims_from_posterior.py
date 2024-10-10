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

# Create an argument parser
parser = argparse.ArgumentParser(description='Process job parameters and save path.')
parser.add_argument('--params', type=str, required=True, help='Comma-separated list of parameters')
parser.add_argument('--job_save_dir', type=str, required=True, help='Path to save the results')

# Parse the arguments
args = parser.parse_args()

# Convert the comma-separated string back into a NumPy array
params = np.fromstring(args.params, sep=',')

glv_out = odeint(generalized_gLV, y0 = init_abun, t=time, args = (params_samp,))

models = [eb_models[i], p_copri_models[i]]    
met_pool_over_time, model_abun_dict = static_dfba(list_model_names=model_names,list_models=models, initial_abundance=init_abun, total_sim_time=(460), num_t_steps=(460), glv_out=glv_out, glv_params=params_samp, environ_cond= rcm_add, pfba=True)


# Save the results to the specified save path
np.save(args.job_save_dir, results)
print(f"Results saved to {args.save_path}")
