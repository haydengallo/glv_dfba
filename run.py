### Hayden Gallo
### Bucci Lab
### 10/9/24

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

wd = '/Users/haydengallo/Documents/Bucci_Lab'




### loading cobra models

### for some reason, sbml/xml models give different initial growth rates than same model but in .mat format???? why 

p_copri_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/dfba_glv/panSpeciesModels_AMANHI_P/panPrevotella_copri.mat')  
eb_model = cobra.io.load_matlab_model('/home/hayden.gallo-umw/data/dfba_glv/panSpeciesModels_AMANHI_P/panEubacterium_limosum.mat') 


#fp_model = cobra.io.load_matlab_model('/Users/haydengallo/cobratoolbox/panSpeciesModels/panFaecalibacterium_prausnitzii.mat') 

#fp_strain_model = cobra.io.load_matlab_model('/Users/haydengallo/cobratoolbox/AGORA-2/AGORA_2_mat/Faecalibacterium_prausnitzii_ERR1022327.mat')


### loading initial metabolite data

#agora_media = pd.read_csv('/Users/haydengallo/Documents/Bucci_Lab/metconsin/AGORA_Media/EU_average_AGORA.tsv', sep = '\t', index_col=0)

### with additions for EB to grow

pyg_test_all_add = [['EX_26dap_M(e)','1'],['EX_2dmmq8(e)','1'],['EX_cgly(e)','1'],['EX_mqn8(e)','1'],['EX_o2(e)','1'],['EX_orn(e)','1'],['EX_sheme(e)','1'],['EX_spmd(e)','1'],['EX_q8(e)','1'],['EX_pydx(e)','1'],['EX_nac(e)','1'],['EX_lys_L(e)','1'],['EX_hxan(e)','1'],['EX_ade(e)','1'],['EX_thymd(e)','1'],['EX_thm(e)','1'],['EX_ribflv(e)','1'],['EX_pnto_R(e)','1'],['EX_nac(e)','1'],['EX_fol(e)','1'],['EX_zn2(e)','1'],['EX_mn2(e)','1'],['EX_fe3(e)','1'],['EX_cu2(e)','1'],['EX_cobalt2(e)','1'],['EX_n2(e)','227.8571429'],['EX_na1(e)','20.51163798'],['EX_cl(e)','5.941065976'],['EX_ca2(e)','0.173941043'],['EX_fe2(e)','0.016053362'],['EX_mg2(e)','0.474477191'],['EX_k(e)','35.39748582'],['EX_so4(e)','0.710983412'],['EX_pi(e)','18.29826648'],['EX_ala_L(e)','16.89227108'],['EX_arg_L(e)','5.338568575'],['EX_asn_L(e)','1.286718791'],['EX_asp_L(e)','10.81893313'],['EX_Lcystin(e)','0.187272152'],['EX_glu_L(e)','18.56754922'],['EX_gln_L(e)','0.205274178'],['EX_gly(e)','20.3151851'],['EX_his_L(e)','3.319218598'],['EX_ile_L(e)','8.195159139'],['EX_leu_L(e)','11.2826377'],['EX_lys_l(e)','9.884397018'],['EX_met_L(e)','2.144657123'],['EX_phe_L(e)','6.083829725'],['EX_pro_L(e)','11.89938505'],['EX_ser_L(e)','4.424652451'],['EX_thr_L(e)','3.567830759'],['EX_trp_L(e)','0.685504997'],['EX_tyr_L(e)','1.683306566'],['EX_val_L(e)','10.37149589'],['EX_glc_D(e)','27.7537245498346'],['EX_hco3(e)','1.190379826'],['EX_phyQ(e)','0.002172408'],['EX_etoh(e)','3.237913066'],['EX_pheme(e)','0.0076693'],['EX_oh1(e)','0.099987502'],['EX_cys_L(e)','2.846894039'],['EX_M02144(e)','2.846894039'],['EX_h2o(e)','55013.2623'],['EX_h(e)','6.30957E-05']]
#pyg_test_all_add = [['EX_26dap_M[e]','1'],['EX_2dmmq8[e]','1'],['EX_cgly[e]','1'],['EX_mqn8[e]','1'],['EX_o2[e]','1'],['EX_orn[e]','1'],['EX_sheme[e]','1'],['EX_spmd[e]','1'],['EX_q8[e]','1'],['EX_pydx[e]','1'],['EX_nac[e]','1'],['EX_lys_L[e]','1'],['EX_hxan[e]','1'],['EX_ade[e]','1'],['EX_thymd[e]','1'],['EX_thm[e]','1'],['EX_ribflv[e]','1'],['EX_pnto_R[e]','1'],['EX_nac[e]','1'],['EX_fol[e]','1'],['EX_zn2[e]','1'],['EX_mn2[e]','1'],['EX_fe3[e]','1'],['EX_cu2[e]','1'],['EX_cobalt2[e]','1'],['EX_n2[e]','227.8571429'],['EX_na1[e]','20.51163798'],['EX_cl[e]','5.941065976'],['EX_ca2[e]','0.173941043'],['EX_fe2[e]','0.016053362'],['EX_mg2[e]','0.474477191'],['EX_k[e]','35.39748582'],['EX_so4[e]','0.710983412'],['EX_pi[e]','18.29826648'],['EX_ala_L[e]','16.89227108'],['EX_arg_L[e]','5.338568575'],['EX_asn_L[e]','1.286718791'],['EX_asp_L[e]','10.81893313'],['EX_Lcystin[e]','0.187272152'],['EX_glu_L[e]','18.56754922'],['EX_gln_L[e]','0.205274178'],['EX_gly[e]','20.3151851'],['EX_his_L[e]','3.319218598'],['EX_ile_L[e]','8.195159139'],['EX_leu_L[e]','11.2826377'],['EX_lys_l[e]','9.884397018'],['EX_met_L[e]','2.144657123'],['EX_phe_L[e]','6.083829725'],['EX_pro_L[e]','11.89938505'],['EX_ser_L[e]','4.424652451'],['EX_thr_L[e]','3.567830759'],['EX_trp_L[e]','0.685504997'],['EX_tyr_L[e]','1.683306566'],['EX_val_L[e]','10.37149589'],['EX_glc_D[e]','27.7537245498346'],['EX_hco3[e]','1.190379826'],['EX_phyQ[e]','0.002172408'],['EX_etoh[e]','3.237913066'],['EX_pheme[e]','0.0076693'],['EX_oh1[e]','0.099987502'],['EX_cys_L[e]','2.846894039'],['EX_M02144[e]','2.846894039'],['EX_h2o[e]','55013.2623'],['EX_h[e]','6.30957E-05']]
#pyg_test_all_add = [['EX_q8[e]','1'],['EX_pydx[e]','1'],['EX_nac[e]','1'],['EX_lys_L[e]','1'],['EX_hxan[e]','1'],['EX_ade[e]','1'],['EX_thymd[e]','1'],['EX_thm[e]','1'],['EX_ribflv[e]','1'],['EX_pnto_R[e]','1'],['EX_nac[e]','1'],['EX_fol[e]','1'],['EX_zn2[e]','1'],['EX_mn2[e]','1'],['EX_fe3[e]','1'],['EX_cu2[e]','1'],['EX_cobalt2[e]','1'],['EX_n2[e]','227.8571429'],['EX_na1[e]','20.51163798'],['EX_cl[e]','5.941065976'],['EX_ca2[e]','0.173941043'],['EX_fe2[e]','0.016053362'],['EX_mg2[e]','0.474477191'],['EX_k[e]','35.39748582'],['EX_so4[e]','0.710983412'],['EX_pi[e]','18.29826648'],['EX_ala_L[e]','16.89227108'],['EX_arg_L[e]','5.338568575'],['EX_asn_L[e]','1.286718791'],['EX_asp_L[e]','10.81893313'],['EX_Lcystin[e]','0.187272152'],['EX_glu_L[e]','18.56754922'],['EX_gln_L[e]','0.205274178'],['EX_gly[e]','20.3151851'],['EX_his_L[e]','3.319218598'],['EX_ile_L[e]','8.195159139'],['EX_leu_L[e]','11.2826377'],['EX_lys_l[e]','9.884397018'],['EX_met_L[e]','2.144657123'],['EX_phe_L[e]','6.083829725'],['EX_pro_L[e]','11.89938505'],['EX_ser_L[e]','4.424652451'],['EX_thr_L[e]','3.567830759'],['EX_trp_L[e]','0.685504997'],['EX_tyr_L[e]','1.683306566'],['EX_val_L[e]','10.37149589'],['EX_glc_D[e]','27.7537245498346'],['EX_hco3[e]','1.190379826'],['EX_phyQ[e]','0.002172408'],['EX_etoh[e]','3.237913066'],['EX_pheme[e]','0.0076693'],['EX_oh1[e]','0.099987502'],['EX_cys_L[e]','2.846894039'],['EX_M02144[e]','2.846894039'],['EX_h2o[e]','55013.2623'],['EX_h[e]','6.30957E-05']]
pyg_test_all_add = np.array(pyg_test_all_add)

pyg_test_all_add = pd.DataFrame(pyg_test_all_add)

pyg_test_all_add.columns = ['reaction','fluxValue']
pyg_test_all_add['fluxValue'] =  np.double(pyg_test_all_add['fluxValue'])


rcm_add = [['EX_ribflv(e)', '0.01'],['EX_thymd(e)', '0.01'],['EX_thm(e)', '0.01'],['EX_spmd(e)', '0.01'],['EX_sheme(e)', '0.01'],['EX_q8(e)', '0.01'],['EX_pheme(e)', '0.01'],['EX_fol(e)', '0.01'],['EX_2dmmq8(e)', '0.01'],['EX_26dap_M(e)', '0.01'],['EX_cobalt2(e)', '0.01'],['EX_cu2(e)', '0.01'],['EX_fe3(e)', '0.01'],['EX_mn2(e)', '0.01'],['EX_zn2(e)', '0.01'],['EX_na1(e)', '49.13878039'],['EX_cl(e)', '35.42367951'],['EX_n2(e)', '221.9285714'],['EX_ca2(e)', '0.083087978'],['EX_fe2(e)', '0.009273883'],['EX_mg2(e)', '0.216827813'],['EX_k(e)', '10.45186108'],['EX_so4(e)', '0.913009078'],['EX_pi(e)', '4.844687796'],['EX_ala_L(e)', '16.70146138'],['EX_arg_L(e)', '5.384491745'],['EX_asn_L(e)', '0.908272088'],['EX_asp_L(e)', '9.083395943'],['EX_glu_L(e)', '17.32518669'],['EX_gly(e)', '25.44393675'],['EX_his_L(e)', '2.120432852'],['EX_ile_L(e)', '6.174957118'],['EX_leu_L(e)', '9.323422908'],['EX_lys_L(e)', '7.168752993'],['EX_met_L(e)', '1.702321591'],['EX_phe_L(e)', '5.193956124'],['EX_pro_L(e)', '11.3782441'],['EX_ser_L(e)', '3.882275699'],['EX_thr_L(e)', '2.837474815'],['EX_trp_L(e)', '0.56309339'],['EX_tyr_L(e)', '1.357683329'],['EX_val_L(e)', '7.89599481'], ['EX_Lcystin(e)', '0.108201688'], ['EX_gln_L(e)', '0.109479562'],['EX_glc_D(e)', '27.75372455'],['EX_cys_L(e)', '2.846894039'],['EX_M02144(e)', '2.846894039'],['EX_h2o(e)', '55509.29781'],['EX_h(e)', '0.000158489']]
rcm_add = np.array(rcm_add)

rcm_add = pd.DataFrame(rcm_add)

rcm_add.columns = ['reaction','fluxValue']
rcm_add['fluxValue'] =  np.double(rcm_add['fluxValue'])



pyg_test_all_add.shape



pyg_media_dict = dict(zip(pyg_test_all_add['reaction'], pyg_test_all_add['fluxValue']))
rcm_media_dict = dict(zip(rcm_add['reaction'], rcm_add['fluxValue']))
#rcm_media_dict



### load in hanks data 

cerillo_test_data = pd.read_csv('/home/hayden.gallo-umw/data/dfba_glv/averaged_cerillo.csv', index_col=0)

cerillo_test_data.shape

cerillo_test_data.head()


test_df = cerillo_test_data[(cerillo_test_data['Group_together'] == 'PCwEB') | (cerillo_test_data['Group_together'] == 'EBwPC')]
microbe_data = test_df.pivot(index='Time', columns= 'Group_together', values = 'OD').reset_index()

print(microbe_data.head())
'''
### should probably make a directory to store inference 
data_dir = str(wd + '/glv_dfba_testing_data/testing')

if os.path.exists(data_dir):
    print('already exists')
else:
    os.mkdir(data_dir)
'''

model_names = ['eb', 'p_copri']
models = [eb_model, p_copri_model]
#init_abun = [.0023, .002633]
init_abun = [.002, .002]

### so far must manually set initial values of parameters
### also might want to manually specify bounds too, could be something good 

# monoculture growth rate of EB
r_1 = 0.04198
# monoculture growth rate of P. copri
r_2 = .115
# co culture growth rate of EB
gamma_1 = 5
# co culture growth rate of P. copri # should be small b/c p. copri grows pretty much the same with or without EB
gamma_2 = 0
# intraspecies competition of EB
a_1 = -10
# intraspecies competition of P. copri
a_2 = -2.9


params = np.array([r_1, r_2, gamma_1, gamma_2, a_1, a_2])
#print(params)
total_sim_time = 460
num_t_steps = 460

#print(np.finfo(float))


### first perform original least squares fit of glv

glv_out, params_ls, time = ls_glv_fit(init_abun = init_abun, params = params, total_sim_time=total_sim_time, time_steps=num_t_steps, microbe_data=microbe_data)
### next perform bayesian glv fit using params_ls
#print(params_ls)
model = bayesian_glv_setup(params_init=params_ls, microbe_data=microbe_data, init_abun=init_abun)
(pm.model_to_graphviz(model=model))

trace = bayesian_glv_run(model=model, num_samples=400000, chains =10)
#trace_save_name = str(data_dir + '/trace.nc')
#trace.to_netcdf(trace_save_name)

### next sample from posterior of bayesian fit 
num_samples_post = 100
param_dict = posterior_param_samps(num_samples=num_samples_post, glv_trace=trace)

#print(param_dict)


### Submitting batch jobs to the cluster



# Job and file paths
job_name = "glv_dfba_testing"
base_output_dir = "/home/hayden.gallo-umw/glv_dfba_testing/test_8"
output_dir = "/home/hayden.gallo-umw/job_output/out_logs"
error_dir = "/home/hayden.gallo-umw/job_output/error_logs"
python_script_path = "/home/hayden.gallo-umw/glv_dfba_implement/glv_dfba/batch_sims_from_posterior.py"

os.makedirs(base_output_dir, exist_ok=True)

# Iterate over seeds and submit jobs
for i in range(0, num_samples_post):
    # build parameter np.array
    params = np.array([param_dict['r_1']['samples'][i], param_dict['r_2']['samples'][i], param_dict['gamma_1']['samples'][i], param_dict['gamma_2']['samples'][i], param_dict['a_1']['samples'][i], param_dict['a_2']['samples'][i]])    
    # convert np.array to comma separated list for later parsing
    params_str = ','.join(map(str, params))
    # take model_names list and make comma sep list
    model_names_str = ','.join(model_names)
    # take init_abun list and make comma sep list
    init_abun_str = ','.join(map(str, init_abun))
    # take time array and send to each job
    time_str = ','.join(map(str, time))


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
    #BSUB -q short
    #BSUB -W 4:00
    #BSUB -n 1
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=2GB]"

    # Execute the Python script with the parameters
    python {python_script_path} --params {params_str}  --model_names {model_names_str} --init_abun {init_abun_str} --job_save_dir {job_save_dir} --time {time_str}
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




'''
    # Create the batch script content
    batch_content = f"""#!/bin/bash
#BSUB -J {unique_job_name}
#BSUB -o {output_dir}/{unique_job_name}.%J
#BSUB -e {error_dir}/{unique_job_name}.%J
#BSUB -q short
#BSUB -W 4:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"

# Execute the Python script with the seed parameter
python {python_script_path} --params {params_str}  --model_names {model_names_str} --init_abun {init_abun_str} --job_save_dir {job_save_dir}
"""

    # Write the batch script to a file
    with open(batch_script, 'w') as file:
        file.write(batch_content)

    # Submit the job using 'bsub'
    try:
        subprocess.run(["bsub", "<", batch_script], check=True)
        print(f"Submitted job for params {params}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job for params {params}: {e}")
'''


