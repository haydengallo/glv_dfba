### Hayden Gallo
### Bucci Lab
### 10/31/24

### trying to batch out glv dfba with 3 species 


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

os.environ["GRB_LICENSE_FILE"] = "/share/pkg/gurobi/11.0.2/lib/gurobi.lic"

wd = '/Users/haydengallo/Documents/Bucci_Lab'

### Load in growth curve data 
### load in hanks data 

cerillo_test_data = pd.read_csv('/home/hayden.gallo-umw/data/dfba_glv/cerillo_data_comp_analysis.csv', index_col=0)
averaged_cerillo_df_for_testing = cerillo_test_data.unstack().reset_index()
averaged_cerillo_df_for_testing.columns = ['Group_together', 'Time', 'OD']
cerillo_test_data = averaged_cerillo_df_for_testing

#cerillo_test_data = pd.read_csv('/home/hayden.gallo-umw/data/dfba_glv/averaged_cerillo.csv', index_col=0)

### load in non-averaged data 
non_avg_data = 'no'
exp_data_dict = np.load('/home/hayden.gallo-umw/data/dfba_glv/exp_data_dict.npy', allow_pickle=True).item()


### Creating the 5 different sets of data EB_PC_co, PC_DO_co, EB_mono, PC_mono, DO_mono

### 1. co culture data of EB and PC

co_culture_df = cerillo_test_data[(cerillo_test_data['Group_together'] == 'PCwEUB') | (cerillo_test_data['Group_together'] == 'EUBwPC')]
EB_PC_co_culture_data = co_culture_df.pivot(index='Time', columns= 'Group_together', values = 'OD').reset_index()

### 2. mono culture data of EB

EB_mono_culture_df = cerillo_test_data[(cerillo_test_data['Group_together'] == 'EUB/EUB')]
EB_mono_culture_data = EB_mono_culture_df.pivot(index='Time', columns= 'Group_together', values = 'OD').reset_index()
EB_mono_culture_data['PC'] = 0

list_zero = EB_mono_culture_data['PC']
EB_mono_culture_data.insert(3, 'DO', list_zero)

### 3. mono culture data of PC

PC_mono_culture_df = cerillo_test_data[(cerillo_test_data['Group_together'] == 'PC/PC')]
PC_mono_culture_data = PC_mono_culture_df.pivot(index='Time', columns= 'Group_together', values = 'OD').reset_index()
PC_mono_culture_data.insert(1, 'EB', list_zero)
PC_mono_culture_data.insert(3, 'DO', list_zero)

### 4. mono culture data of DO

mono_culture_df_DO = cerillo_test_data[(cerillo_test_data['Group_together'] == 'DO/DO')]
DO_mono_culture_data = mono_culture_df_DO.pivot(index='Time', columns= 'Group_together', values = 'OD').reset_index()
DO_mono_culture_data.insert(1, 'EB', list_zero)
DO_mono_culture_data.insert(2, 'PC', list_zero)

### 5. co culture data of PC and DO 

co_culture_df_DO_PC = cerillo_test_data[(cerillo_test_data['Group_together'] == 'PCwDO') | (cerillo_test_data['Group_together'] == 'DOwPC')]
DO_PC_co_culture_data = co_culture_df_DO_PC.pivot(index='Time', columns= 'Group_together', values = 'OD').reset_index()
DO_PC_co_culture_data.insert(1, 'EB', list_zero)
DO_PC_co_culture_data = DO_PC_co_culture_data[['Time', 'EB', 'PCwDO', 'DOwPC']]

### Add DO data to coculture of PC and EB

EB_PC_co_culture_data.insert(3, 'DO', list_zero)


### setting up the initial abundances

init_abun_co_EB_PC = [.005, .005, 0]
init_abun_co_PC_DO = [0, .005, 0.005]
EB_init_abun_mono = [.005, 0, 0]
PC_init_abun_mono = [0, 0.005, 0]
DO_init_abun_mono = [0, 0, 0.005]


### setup for the averaged data, use this data for getting weakly informative priors, then use non-averaged data in mcmc so we can get idea of uncertainty on predictions which can be propagated to 
### predictions about metabolites etc. 
abun_list_priors = [init_abun_co_EB_PC, init_abun_co_PC_DO, EB_init_abun_mono, PC_init_abun_mono, DO_init_abun_mono]
microbe_data_list_priors = [EB_PC_co_culture_data,DO_PC_co_culture_data, EB_mono_culture_data, PC_mono_culture_data, DO_mono_culture_data]
dataset_names = ['EB_PC_co', 'DO_PC_co', 'EB_mono', 'PC_mono', 'DO_mono']


'''
### should probably make a directory to store inference 
data_dir = str(wd + '/glv_dfba_testing_data/testing')

if os.path.exists(data_dir):
    print('already exists')
else:
    os.mkdir(data_dir)
'''

model_names = ['eb', 'p_copri', 'do']


### Set initial parameter values 

'''
# monoculture growth rate of EB
r_1 = .15
# monoculture growth rate of P. copri
r_2 = .15
# monoculture growth rate of DO
r_3 = .5
# co culture growth rate of EB with PC
gamma_EP = .05
# co culture growth rate of EB with DO
gamma_ED = 0
# co culture growth rate of PC with EB
gamma_PE = .001
# co culture growth rate of PC with DO
gamma_PD = .001
# co culture growth rate of DO with EB
gamma_DE = 0
# co culture growth rate of DO with PC
gamma_DP = -.05
# intraspecies competition of EB
a_1 = -.05
# intraspecies competition of P. copri
a_2 = -.05
# intraspecies competition of DO
a_3 = 0
'''

# monoculture growth rate of EB
r_1 = .275
# monoculture growth rate of P. copri
r_2 = .26
# monoculture growth rate of DO
r_3 = .28
# co culture growth rate of EB with PC
gamma_EP = .05
# co culture growth rate of EB with DO
gamma_ED = 0
# co culture growth rate of PC with EB
gamma_PE = .01
# co culture growth rate of PC with DO
gamma_PD = .01
# co culture growth rate of DO with EB
gamma_DE = 0
# co culture growth rate of DO with PC
gamma_DP = -.12
# intraspecies competition of EB
a_1 = -.05
# intraspecies competition of P. copri
a_2 = -.05
# intraspecies competition of DO
a_3 = 0


bnds = ((0, 1), (0, 1), (0, 1), (-1, 1), (0, 0), (-1, 1), (-1, 1), (0, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0))
params = np.array([r_1, r_2, r_3, gamma_EP, gamma_ED, gamma_PE, gamma_PD, gamma_DE, gamma_DP, a_1, a_2, a_3])

total_sim_time = 500
num_t_steps = 500

#print(np.finfo(float))

num_samples_post = 50
mcmc_samps = 500000
#mcmc_samps = 5000

results = minimize(total_loss_multi, params, args=(microbe_data_list_priors, abun_list_priors), bounds=bnds)    
new_params = results.x
print(new_params)

#new_params = np.array([0.01506192,0.01410974,0.0151906,0.04874128,0.,0.00151483,0.00170373,0.,-0.03227239,-0.04587137,-0.02085214,-0.01264632])



### first perform original least squares fit of glv

#glv_out, params_ls, time = ls_glv_fit(init_abun = init_abun, params = params, total_sim_time=total_sim_time, time_steps=num_t_steps, microbe_data=microbe_data)
### next perform bayesian glv fit using params_ls
#print(params_ls)

### setup data to be used as input for MCMC 

if non_avg_data == 'yes':

    microbe_data_list_mcmc = []
    for key in exp_data_dict.keys():
        for j in range(0, len(exp_data_dict[key])):
            microbe_data_list_mcmc.append(exp_data_dict[key][j])

    ### this is quite inefficient, so would be good to figure out a better way to do this later on 

    abun_list_mcmc = [init_abun_co_EB_PC,init_abun_co_EB_PC,init_abun_co_EB_PC,init_abun_co_PC_DO,init_abun_co_PC_DO,init_abun_co_PC_DO,
                EB_init_abun_mono,EB_init_abun_mono,EB_init_abun_mono,EB_init_abun_mono,EB_init_abun_mono,EB_init_abun_mono,
                PC_init_abun_mono,PC_init_abun_mono,PC_init_abun_mono,PC_init_abun_mono,PC_init_abun_mono,PC_init_abun_mono,
                DO_init_abun_mono,DO_init_abun_mono,DO_init_abun_mono,DO_init_abun_mono,DO_init_abun_mono,DO_init_abun_mono] 

    model = bayesian_glv_setup_three_spec(params_init=new_params, microbe_data_list=microbe_data_list_mcmc, abun_list=abun_list_mcmc)

else:

    model = bayesian_glv_setup_three_spec(params_init=new_params, microbe_data_list=microbe_data_list_priors, abun_list=abun_list_priors)

trace = bayesian_glv_run_three_spec(model=model, num_samples=mcmc_samps, chains = 10)

#trace_save_name = str(data_dir + '/trace.nc')
#trace.to_netcdf(trace_save_name)

### next sample from posterior of bayesian fit 

#trace = az.from_netcdf(trace_save_name)

#param_dict = posterior_param_samps_multi(num_samples=num_samples_post, glv_trace=trace) 
#print(param_dict)
### Submitting batch jobs to the cluster


# Job and file paths
job_name = "glv_dfba_testing"
base_output_dir = "/home/hayden.gallo-umw/glv_dfba_testing/test_54"
output_dir = "/home/hayden.gallo-umw/job_output/out_logs/glv_dfba_testing/test_54"
os.makedirs(output_dir, exist_ok=True)
error_dir = "/home/hayden.gallo-umw/job_output/error_logs/glv_dfba_testing/test_54"
os.makedirs(error_dir, exist_ok=True)
python_script_path = "/home/hayden.gallo-umw/glv_dfba_implement/glv_dfba/batch_sims_from_posterior.py"

os.makedirs(base_output_dir, exist_ok=True)

trace_save_name = str(base_output_dir + '/trace.nc')

#trace = az.from_netcdf(trace_save_name)

param_dict = posterior_param_samps_multi(num_samples=num_samples_post, glv_trace=trace) 

trace.to_netcdf(trace_save_name)



# looping through all of the datasets and batch submitting all jobs so N 
for j, init_abun in enumerate(abun_list_priors):
    ###                                                                                        ###
    ### think i want to loop over the single instances of init_abun not what was fed into mcmc ###
    ###                                                                                        ###

    subdirectory = base_output_dir + '/' + dataset_names[j]
    os.makedirs(subdirectory, exist_ok=True)

    # Iterate over seeds and submit jobs
    for i in range(0, num_samples_post):
        # build parameter np.array
        params = np.array([param_dict['r_1']['samples'][i], param_dict['r_2']['samples'][i], param_dict['r_3']['samples'][i],
                            param_dict['gamma_EP']['samples'][i],0, param_dict['gamma_PE']['samples'][i], param_dict['gamma_PD']['samples'][i] ,0, param_dict['gamma_DP']['samples'][i],
                              param_dict['a_1']['samples'][i], param_dict['a_2']['samples'][i], param_dict['a_3']['samples'][i]])    
        # convert np.array to comma separated list for later parsing
        params_str = ','.join(map(str, params))
        # take model_names list and make comma sep list
        model_names_str = ','.join(model_names)
        # take init_abun list and make comma sep list
        init_abun_str = ','.join(map(str, init_abun))
        # take time array and send to each job
        #time_str = ','.join(map(str, time))


        unique_job_name = f"{job_name}_{i}"
        job_save_dir = os.path.join(subdirectory, unique_job_name)
        os.makedirs(job_save_dir, exist_ok=True)

        batch_script = f"{subdirectory}/{unique_job_name}.lsf"



    #batch_script = f"{base_output_dir}/{unique_job_name}.lsf"

        # Create the batch script content
        batch_content = f"""#!/bin/bash
        #BSUB -J {unique_job_name}
        #BSUB -o {output_dir}/{unique_job_name}.%J.out
        #BSUB -e {error_dir}/{unique_job_name}.%J.err
        #BSUB -q short
        #BSUB -W 4:00
        #BSUB -n 1
        #BSUB -R "span[hosts=1]"
        #BSUB -R "rusage[mem=500]"

        # Execute the Python script with the parameters
        python {python_script_path} --params {params_str}  --model_names {model_names_str} --init_abun {init_abun_str} --job_save_dir {job_save_dir} 
        """

        # Write the batch script to a file
        with open(batch_script, 'w') as file:
            file.write(batch_content)

        # Submit the job using 'bsub' by reading the batch script file
        try:
            with open(batch_script) as f:
                subprocess.run(["bsub"], stdin=f, check=True)
            print(f"Submitted job for params {params}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit job for params {params}: {e}")


### Ok now here after all of the batched out jobs have finished, automatically make plots
### to do this
