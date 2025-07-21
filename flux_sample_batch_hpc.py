### Hayden Gallo
### Bucci Lab
### 7/17/25
### Batching flux sample runs 

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
from scipy.optimize import minimize
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

test_num = 3


# Job and file paths
job_name = "sim"
base_output_dir = "/home/hayden.gallo-umw/MDSINE_Flux_Sampling/flux_sampling_simulations/Test_" + str(test_num)
output_dir = "/home/hayden.gallo-umw/job_output/out_logs/flux_sampling_simulations/Test_" + str(test_num)
os.makedirs(output_dir, exist_ok=True)
error_dir = "/home/hayden.gallo-umw/job_output/error_logs/flux_sampling_simulations/Test_" + str(test_num)
os.makedirs(error_dir, exist_ok=True)
python_script_path = "/home/hayden.gallo-umw/glv_dfba_implement/glv_dfba/MDSINE_flux_sampling_hpc.py"

os.makedirs(base_output_dir, exist_ok=True)




for i in range(0, 500):

    sim_num = i
    unique_job_name = f"{job_name}_{i}"
    job_save_dir = os.path.join(base_output_dir, unique_job_name)
    os.makedirs(job_save_dir, exist_ok=True)

    batch_script = f"{base_output_dir}/{unique_job_name}.lsf"



#batch_script = f"{base_output_dir}/{unique_job_name}.lsf"

    # Create the batch script content
    batch_content = f"""#!/bin/bash
    #BSUB -J {unique_job_name}
    #BSUB -o {output_dir}/{unique_job_name}.%J.out
    #BSUB -e {error_dir}/{unique_job_name}.%J.err
    #BSUB -q long
    #BSUB -W 72:00
    #BSUB -n 1
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=2GB]"

    # Execute the Python script with the parameters
    python {python_script_path} --sim_num {sim_num} --test_num {test_num}
    """

    # Write the batch script to a file
    with open(batch_script, 'w') as file:
        file.write(batch_content)

    # Submit the job using 'bsub' by reading the batch script file
    try:
        with open(batch_script) as f:
            subprocess.run(["bsub"], stdin=f, check=True)
        print(f"Submitted job, sim num: {sim_num}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e}")