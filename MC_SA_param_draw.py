### Hayden Gallo
### Bucci Lab
### 3/13/25

### Implementation of Monte Carlo Sensitivity analysis for dfba
### here we construct parameter distributions for metabolites and then draw from said distributions 
### then batch out jobs with parameter values and run dfba simulations  


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
import time
### script for running glv_dfba inference
from helper_functions import *

os.environ["GRB_LICENSE_FILE"] = "/share/pkg/gurobi/11.0.2/lib/gurobi.lic"

#############################
### Set number of samples ###
#############################

num_samples = 5000
print(num_samples)

################
### Job name ###
################


sim_job_name = 'co_culture_MC_pareto_frontier'

########################
### Media to utilize ###
########################

rcm_add = [['EX_adocbl(e)','0.01'],['EX_btn(e)','0.01'],['EX_h2s(e)','0.01'],['EX_ribflv(e)', '0.01'],['EX_thymd(e)', '0.01'],['EX_thm(e)', '0.01'],['EX_spmd(e)', '0.01'],['EX_sheme(e)', '0.01'],['EX_q8(e)', '0.01'],['EX_pheme(e)', '0.01'],['EX_fol(e)', '0.01'],['EX_2dmmq8(e)', '0.01'],['EX_26dap_M(e)', '0.01'],['EX_cobalt2(e)', '0.01'],['EX_cu2(e)', '0.01'],['EX_fe3(e)', '0.01'],['EX_mn2(e)', '0.01'],['EX_zn2(e)', '0.01'],['EX_na1(e)', '49.13878039'],['EX_cl(e)', '35.42367951'],['EX_n2(e)', '221.9285714'],['EX_ca2(e)', '0.083087978'],['EX_fe2(e)', '0.009273883'],['EX_mg2(e)', '0.216827813'],['EX_k(e)', '10.45186108'],['EX_so4(e)', '0.913009078'],['EX_pi(e)', '4.844687796'],['EX_ala_L(e)', '16.70146138'],['EX_arg_L(e)', '5.384491745'],['EX_asn_L(e)', '0.908272088'],['EX_asp_L(e)', '9.083395943'],['EX_glu_L(e)', '17.32518669'],['EX_gly(e)', '25.44393675'],['EX_his_L(e)', '2.120432852'],['EX_ile_L(e)', '6.174957118'],['EX_leu_L(e)', '9.323422908'],['EX_lys_L(e)', '7.168752993'],['EX_met_L(e)', '1.702321591'],['EX_phe_L(e)', '5.193956124'],['EX_pro_L(e)', '11.3782441'],['EX_ser_L(e)', '3.882275699'],['EX_thr_L(e)', '2.837474815'],['EX_trp_L(e)', '0.56309339'],['EX_tyr_L(e)', '1.357683329'],['EX_val_L(e)', '7.89599481'], ['EX_Lcystin(e)', '0.108201688'], ['EX_gln_L(e)', '0.109479562'],['EX_glc_D(e)', '27.75372455'],['EX_cys_L(e)', '2.846894039'],['EX_M02144(e)', '2.846894039'],['EX_h2o(e)', '55509.29781'],['EX_h(e)', '0.000158489']]

rcm_add = pd.DataFrame(rcm_add)

rcm_add.columns = ['reaction','fluxValue']
rcm_add['fluxValue'] =  np.double(rcm_add['fluxValue'])



### Base directory

main_directory = '/home/hayden.gallo-umw/glv_dfba_testing'

### Make directory for MC_sensitivity_analysis 
job_name = "MC_sensitivity_analysis"
save_dir = main_directory + '/' + job_name
os.makedirs(save_dir, exist_ok=True)

### load in trace of gLV to use 
trace_file_path = '/home/hayden.gallo-umw/glv_dfba_testing/test_54/trace.nc'
trace = az.from_netcdf(trace_file_path)
trace_summary = az.summary(trace)


### ok here load cerillo data and get set for batching out 

cerillo_test_data = pd.read_csv('/home/hayden.gallo-umw/data/dfba_glv/cerillo_data_comp_analysis.csv', index_col=0)
averaged_cerillo_df_for_testing = cerillo_test_data.unstack().reset_index()
averaged_cerillo_df_for_testing.columns = ['Group_together', 'Time', 'OD']
cerillo_test_data = averaged_cerillo_df_for_testing

#mono culture data of EB

EB_mono_culture_df = cerillo_test_data[(cerillo_test_data['Group_together'] == 'EUB/EUB')]
EB_mono_culture_data = EB_mono_culture_df.pivot(index='Time', columns= 'Group_together', values = 'OD').reset_index()
EB_mono_culture_data['PC'] = 0

list_zero = EB_mono_culture_data['PC']
EB_mono_culture_data.insert(3, 'DO', list_zero)



### 1. co culture data of EB and PC

co_culture_df = cerillo_test_data[(cerillo_test_data['Group_together'] == 'PCwEUB') | (cerillo_test_data['Group_together'] == 'EUBwPC')]
EB_PC_co_culture_data = co_culture_df.pivot(index='Time', columns= 'Group_together', values = 'OD').reset_index()

### Add DO data to coculture of PC and EB

EB_PC_co_culture_data.insert(3, 'DO', list_zero)








EB_init_abun_mono = [.005, 0, 0]
init_abun_co_EB_PC = [.005, .005, 0]

init_abun = EB_init_abun_mono

abun_list_priors = [ EB_init_abun_mono, init_abun_co_EB_PC]
microbe_data_list_priors = [ EB_mono_culture_data, EB_PC_co_culture_data]
dataset_names = [ 'EB_mono', 'EB_PC_co']
model_names = ['eb', 'p_copri', 'do']





### Parameter array 

params = np.array([trace_summary['mean'].loc['r_1'], trace_summary['mean'].loc['r_2'], trace_summary['mean'].loc['r_3'], trace_summary['mean'].loc['gamma_EP'], 0, trace_summary['mean'].loc['gamma_PE'],
                    trace_summary['mean'].loc['gamma_PD'], 0, trace_summary['mean'].loc['gamma_DP'], trace_summary['mean'].loc['a_1'], trace_summary['mean'].loc['a_2'], trace_summary['mean'].loc['a_3']])


# here need to specify in a list the mets to vary

mets_to_vary = ['EX_fe2(e)', 'EX_ribflv(e)', 'EX_thymd(e)', 'EX_fol(e)', 'EX_2dmmq8(e)']
pct_to_vary = 0.20
#mets_to_vary = rcm_add['reaction'].to_list()



### function to draw met conc, can probably just add this to helper functions at some point
def draw_met_conc(media_df, num_samples, pct_to_vary):

    mets_draws_dict = {}
    mets_draws_array = np.zeros([len(media_df),num_samples])

    for i in range(0, len(media_df)):
        
        rxn_name = media_df.iloc[i,0]
        rxn_flux = media_df.iloc[i,1]
        # for now lets just make the sd 5% of what was supplied 
        # nvm just specify the amount you want to vary with pct_to_vary

        if rxn_name in mets_to_vary:
            five_pct_rxn_flux = rxn_flux * pct_to_vary

            if rxn_flux == 0.01:
                samples = np.random.uniform(0.01,0.1,num_samples)
            else:
                samples = truncnorm.rvs((0-rxn_flux)/five_pct_rxn_flux,(np.inf-rxn_flux)/five_pct_rxn_flux,loc = rxn_flux, scale=five_pct_rxn_flux, size = num_samples)

            # convert samples array to a list to be stored in the dict   
            samples = samples.tolist()
            mets_draws_dict[rxn_name] = samples
            mets_draws_array[i,:] = samples
        
        else:

            samples = [rxn_flux] * num_samples
            mets_draws_dict[rxn_name] = samples
            mets_draws_array[i,:] = samples


    mets_draws_df = pd.DataFrame(mets_draws_array)
    mets_draws_df.index = media_df['reaction']

    return mets_draws_dict, mets_draws_df



### draw 
mets_draws_dict, mets_draw_df = draw_met_conc(rcm_add,num_samples, pct_to_vary)




job_save_dir = os.path.join(save_dir, sim_job_name)
os.makedirs(job_save_dir, exist_ok=True)
print(job_save_dir)

param_df_path = job_save_dir + '/MC_param_df.pkl'

mets_draw_df.to_pickle(param_df_path)

#### Make output directories and error directories
output_dir = "/home/hayden.gallo-umw/job_output/out_logs/MC_sensitivity_analysis/" + sim_job_name
os.makedirs(output_dir, exist_ok=True)
error_dir = "/home/hayden.gallo-umw/job_output/error_logs/MC_sensitivity_analysis/" + sim_job_name
os.makedirs(error_dir, exist_ok=True)
python_script_path = "/home/hayden.gallo-umw/glv_dfba_implement/glv_dfba/batch_MC_sims.py"



for j, init_abun in enumerate(abun_list_priors):

    ### ok so here now need to batch jobs similarly to how it was done with glv-dfba
    subdirectory = job_save_dir + '/' + dataset_names[j]
    os.makedirs(subdirectory, exist_ok=True)

    rmse_path = subdirectory + '/rmse'
    os.makedirs(rmse_path, exist_ok=True)
    #np.save(os.path.join(param_df_path), mets_draw_df)

    # looping through all of the datasets and batch submitting all jobs so N

        # Iterate over seeds and submit jobs
    for i in range(0, num_samples):

        # every 250 jobs batched, sleep for 1000 seconds so as not to overload job submission
        if (i%500 == 0 and i != 0):
            time.sleep(500)

        sim_name = 'sim'
        unique_job_name = f"{sim_name}_{i}"
        sim_save_dir = subdirectory + '/' + unique_job_name
        os.makedirs(sim_save_dir, exist_ok=True)

        batch_script = f"{subdirectory}/{unique_job_name}.lsf"


        # convert np.array to comma separated list for later parsing
        params_str = ','.join(map(str, params))
        # take model_names list and make comma sep list
        model_names_str = ','.join(model_names)
        # take init_abun list and make comma sep list
        init_abun_str = ','.join(map(str, init_abun))
        # take time array and send to each job
        #time_str = ','.join(map(str, time))
        # take sim number and submit as params for job, so that you grab correct column from met_draw_df
        sim_num = str(i)






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
        python {python_script_path} --params {params_str}  --model_names {model_names_str} --init_abun {init_abun_str} --sim_save_dir {sim_save_dir} --sim_num {sim_num} --param_df_path {param_df_path} --rmse_path {rmse_path}
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