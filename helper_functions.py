### Hayden Gallo
### Bucci Lab
### 10/9/24

### Helper Function file for dynamic Flux Balance Analysis Constrained by gLV

### module import

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



### implementing function to create dict for initial microbial abundance paired with model
### becomes important later for loading models from just list of model names etc. 

### make dict of dicts to keep track of model names, models, and abundances of species

def init_model_abun(model_names, models, init_abun, glv_out):
    
    model_abun_dict = {}

    for i in range(0, len(model_names)):

        model_abun_dict[model_names[i]] = {}
        model_abun_dict[model_names[i]]['model'] = models[i]
        model_abun_dict[model_names[i]]['abun'] = init_abun[i]
        ### Need to figure out a better key system for incorporating glv out 
        model_abun_dict[model_names[i]]['glv_out'] = glv_out[:,i]
        model_abun_dict[model_names[i]]['fba_biomass'] = [init_abun[i]]
        model_abun_dict[model_names[i]]['flux_up'] = []
        model_abun_dict[model_names[i]]['flux_sec'] = []

    return model_abun_dict


def make_media(model, media):

    media_dict = {}

    for i in range(0, len(media)):

        #media_dict[media['reaction'].iloc[i].replace('[','(').replace(']',')')] = media['fluxValue'].iloc[i]
        if media['reaction'].iloc[i].replace('[','(').replace(']',')') in model.medium.keys():

            media_dict[media['reaction'].iloc[i].replace('[','(').replace(']',')')] = media['fluxValue'].iloc[i]

        else:
            continue

    return media_dict


def change_media(model_abun_dict, supplied_media):

    for key in model_abun_dict:
        
        temp_media = make_media(model_abun_dict[key]['model'], media=supplied_media)

        # set media conditions to be the temp_media

        model_abun_dict[key]['model'].medium = temp_media

        ### here we take our temp media and apply to model.medium while also checking to make sure lengths are correct, if len(temp_media) != len(model.medium.keys) 
        ### then we determine which metabolites are missing and manually change them in model.reactions.get_by_id('reaction').lower_bound = -flux

        if len(temp_media) == len(model_abun_dict[key]['model'].medium.keys()):
            #print('good to go')
            continue

        else:

            if 'EX_ribflv(e)' in temp_media.keys():
                print('yes')
                model_abun_dict[key]['model'].get_by_id(id='EX_ribflv(e)').lower_bound = temp_media['EX_ribflv(e)']
                
            else:
                print('no')
    return


### this is to be run after optimization step 

def model_opt_out(model_abun_dict, delta_t, pfba):

    total_sys_uptake = {}
    total_sys_secretion = {}

    for key in model_abun_dict:

        if pfba == True:

            # optimize model i via pfba 
            temp_pfba = cobra.flux_analysis.pfba(model_abun_dict[key]['model'])
            # put fluxes in df for manipulation
            temp_pfba_df = temp_pfba.to_frame().filter(regex='EX_', axis = 0)
            
            # filter out fluxes that are secreted
            # signs are flipped here compared to standard fba optimization in cobapy so must change them

            # secreted should have negative sign to align with normal FBA
            test_secrete = temp_pfba_df[temp_pfba_df['fluxes'] > 0]
            # filter out fluxes that are taken up
            # uptake should have positive sign to align with normal FBA
            test_uptake = temp_pfba_df[temp_pfba_df['fluxes'] < 0]

            temp_uptake = np.abs(test_uptake['fluxes']) * delta_t * model_abun_dict[key]['abun']
            temp_secrete = -1.0*(test_secrete['fluxes']) * delta_t * model_abun_dict[key]['abun']   
            #print(temp_secrete)
            pfba_obj_val = test_secrete.filter(regex='bio', axis=0)['fluxes'].iloc[0]

            #print('Obj val', pfba_obj_val)
            model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + pfba_obj_val
        
        else:

            test_secrete = model_abun_dict[key]['model'].summary().secretion_flux.loc[model_abun_dict[key]['model'].summary().secretion_flux['reaction'].filter(regex='EX_')]
            test_uptake = model_abun_dict[key]['model'].summary().uptake_flux.loc[model_abun_dict[key]['model'].summary().uptake_flux['reaction'].filter(regex='EX_')]

            test_secrete = test_secrete[test_secrete['flux'] !=0 ]
            test_uptake = test_uptake[test_uptake['flux'] !=0 ]

            temp_uptake = test_uptake['flux'] * delta_t * model_abun_dict[key]['abun']
            temp_secrete = test_secrete['flux'] * delta_t * model_abun_dict[key]['abun']

            ### now that we've converted metabolite fluxes using abundance of previous time step, now update bacterial abundance for time t + 1

            #model_abun_dict[key]['abun'] += model_abun_dict[key]['model'].summary()._objective_value * model_abun_dict[key]['abun'] * delta_t

            #print(model_abun_dict[key]['model'].summary()._objective_value * delta_t * model_abun_dict[key]['abun'], 'add this amount')
            #print(model_abun_dict[key]['model'].summary()._objective_value, 'model obj')
            #print(model_abun_dict[key]['model'].summary()._objective_value * delta_t * model_abun_dict[key]['abun'], 'amount add')


            #model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + model_abun_dict[key]['model'].summary()._objective_value * delta_t * model_abun_dict[key]['abun']
            model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + model_abun_dict[key]['model'].summary()._objective_value

            #print('Obj val', model_abun_dict[key]['model'].summary()._objective_value)

        ### this is wrong for some reason... 
        
        model_abun_dict[key]['fba_biomass'].append(model_abun_dict[key]['abun'])
        #print(model_abun_dict[model_names[i]]['abun'])

        # dict of uptake for model x in i to n 
        uptake_dict = dict(zip(temp_uptake.index, temp_uptake))
        # add uptake_dict at time t to model storage
        model_abun_dict[key]['flux_up'].append(uptake_dict.copy())
        # dict of sec for model x in i to n 
        secrete_dict = dict(zip(temp_secrete.index, temp_secrete))
        # add sec_dict at time t to model storage
        model_abun_dict[key]['flux_sec'].append(secrete_dict.copy())

        for key in uptake_dict:
            if key in total_sys_uptake.keys():
                #print(uptake_dict[key])
                #print(met_pool_dict[key])
                total_sys_uptake[key] += uptake_dict[key]
                #print(total_sys_uptake[key])
            else:
                total_sys_uptake[key] = np.abs(uptake_dict[key])
            #else:
                ### if there's a key in uptake dict that isn't found in met_pool something is wrong, kill sim ###
            #    RuntimeError()

        for key in secrete_dict:
            if key in total_sys_secretion.keys():
                total_sys_secretion[key] += np.abs(secrete_dict[key])
            else:
                if key == 'EX_biomass(e)':
                    #print('biomass')
                    continue
                else:
                    ### here add new metabolite to metabolite pool 
                    total_sys_secretion[key] = np.abs(secrete_dict[key])
                    #print(np.abs(secrete_dict[key]))
            

    return total_sys_uptake, total_sys_secretion


def change_biomass_bounds(model_abun_dict, glv_params, t_pt):

    temp_abun_list = []
    for key in model_abun_dict:
        temp_abun_list.append(model_abun_dict[key]['glv_out'][t_pt])
    
    gr_rt_t_pt = generalized_gLV(temp_abun_list, t_pt, params=glv_params)
    #print(gr_rt_t_pt)

    for count, key in enumerate(model_abun_dict):
        if gr_rt_t_pt[count] > 0:
            model_abun_dict[key]['model'].reactions.get_by_id(id = 'biomassPan').upper_bound = gr_rt_t_pt[count]

        else:
            model_abun_dict[key]['model'].reactions.get_by_id(id = 'biomassPan').upper_bound = 0



### shouldn't this be embedded in model_opt_out?
### maybe but i kinda want metabolite pool to stay untouched until end of time point 

def update_met_pool(uptake_dict, secrete_dict, met_pool_dict):

    for key in uptake_dict:
        if key in met_pool_dict.keys():
            #print(uptake_dict[key])
            #print(met_pool_dict[key])
            met_pool_dict[key] -= uptake_dict[key]
            #print(met_pool_dict[key])
        else:
            ### if there's a key in uptake dict that isn't found in met_pool something is wrong, kill sim ###
            RuntimeError()

    for key in secrete_dict:
        if key in met_pool_dict.keys():
            met_pool_dict[key] += secrete_dict[key]
        else:
            if key == 'EX_biomass(e)':
                #print('biomass')
                continue
            else:
                ### here add new metabolite to metabolite pool 
                met_pool_dict[key] = np.abs(secrete_dict[key])
                #print(np.abs(secrete_dict[key]))
                
    return met_pool_dict    


### abundance dict, loop through, opt 
### might be a good idea to vectorize this step at some point ###

def opt_model(model_abun_dict, pfba):

    if pfba == True:
        return
    else:
        for key in model_abun_dict:
            model_abun_dict[key]['model'].optimize()

    return    



### implementing static opt dfba approach

### this is the main function that wraps all other helper functions ### 

def static_dfba(list_model_names, list_models, initial_abundance, total_sim_time, num_t_steps, glv_out, glv_params, environ_cond, pfba):

    # implementing a static optimization approach dfba
    # this is a basic approach and will potentially implement more efficient approach later on 

    # to start must run an initial fba at initial conditions to obtain original growth rate etc. 

    # t: time 
    # x: is our steady state concentrations of biomass for both microbes and metabolites

    # we can integrate ODE from time t to t+1 and stop integration there, then we will take new variables and adjust concentrations of 
    # metabolites and microbes at time t+1 to then rerun fba again for each microbe 
    # I can make separate 

    # dxdt : taxa conc
    # dydt : metabolite concentration

    #dxdt = init_conc * growth_rate
    #dydt = init_conc*np.dot(init_conc*flux)

    # i think that the interaction coefficients should tell us something about if these bacteria will compete for the same resources or take from resources secreted by the other 


    ### Maybe start by transforming differential equations into linear approach b/c SOA method can be easily transformed 

    # steps of static dfba coupled with gLV

    # 1. fit gLV to time series data 
    # 2. obtain initial conditions of bacterial concentration and media conditions 
    # 3. discretize the time series into x = n timesteps
    # 4. estimate slope/rate from t=0 to t=1 for abundance of every taxa present in simulation
    # 5. simulate fba with up and lb constrained by species specific growth rate defined by gLV
    # 6. based on fluxes need, both secretory and uptake need to update pool of metabolites in environment before next time step
    # 7. repeat steps 4-6 until end of time series or we get infeasible FBA solution 
    # 8. once at end of time series, then report final metabolite concentrations etc. but use previous timesteps value (t-1) to update fluxes i.e. [x t-1]*flux

    
    # here i need to set the media conditions for each model 
    # really need to build genomic scale models and then gap fill based on media conditions 

    ### Come up with roadmap/diagram of ideas for Vanni 

    ## need to grab derivative at every time point, can i just do that by inserting concentration at time t?? need to do a bit of reading on this      


    #print(glv_out)
    ### Make initial metabolite pool dict
    # this is our initial metabolite pool

    # initialize list to hold dictionaries of metabolite pool over time
    met_pool_over_time = []

    met_pool_dict = dict(zip(environ_cond['reaction'], environ_cond['fluxValue']))

    ## add initial conditions to the list of dicts of metabolite pool

    met_pool_over_time.append(met_pool_dict.copy())

    # Step 0. Create model_abun_dict, main dictionary for everything

    model_abun_dict = init_model_abun(model_names=list_model_names,models = list_models, init_abun=initial_abundance, glv_out=glv_out)
    
    for i in range(0, num_t_steps):
        print('Time step: ', i)

        # Step 1. Change media conditions of models 

        change_media(model_abun_dict= model_abun_dict, supplied_media= environ_cond)

        # Step 2. Change upper bounds of biomass growth rate for each model

        change_biomass_bounds(model_abun_dict=model_abun_dict, glv_params=glv_params, t_pt=i)

        # Step 3. Run actual FBA/optimization

        opt_model(model_abun_dict=model_abun_dict, pfba=pfba)

        # Step 4. Adjust model optimization output fluxes based on abundance and time step size

        total_sys_uptake, total_sys_secretion = model_opt_out(model_abun_dict=model_abun_dict, delta_t= (total_sim_time/num_t_steps), pfba=pfba)

        # Step 5. Update total metabolite pool
        met_pool_dict  = update_met_pool(uptake_dict=total_sys_uptake, secrete_dict=total_sys_secretion, met_pool_dict=met_pool_dict)
        #print(met_pool_dict)
        met_pool_over_time.append(met_pool_dict.copy())
        #print(len(met_pool_over_time))



    
    return met_pool_over_time, model_abun_dict



# function that calculates residuals based on a given theta
def ode_model_resid(params):
    return (
        microbe_data.iloc[:,1:] - odeint(generalized_gLV, y0 = init_abun, t=microbe_data['Time'], args = (params,))
    ).values.flatten()


### implementing naive gLV

def generalized_gLV(X, t, params):
    x, y = X 
    r_1, r_2, gamma_1, gamma_2, a_1, a_2 =  params

    # EB concentration integrated overtime
    dxdt = x * (r_1 + a_1*x + gamma_1*y)
    
    # P.copri concentration integrated overtime
    dydt = y * (r_2 + gamma_2*x + a_2*y)

    return [dxdt, dydt]



def ls_glv_fit(init_abun, params, total_sim_time, time_steps, microbe_data):

    init_abun = np.array(init_abun)
    
    results = least_squares(ode_model_resid, x0=params, bounds=([0, 0, 0, -10, -10, -10], [10, 10, 10, 10, 0, 0]), xtol = 1e-10)

    params = results.x
    
    time = np.arange(0, int(total_sim_time+1), int(total_sim_time/time_steps))

    x_y = odeint(generalized_gLV, y0 = init_abun, t=time, args = (params,))


    return x_y, params, time
    




### implementing bayesian inference for gLV dynamics 
### based on tutorial found here:
### https://www.pymc.io/projects/examples/en/latest/ode_models/ODE_Lotka_Volterra_multiple_ways.html

# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(params_ls):
    return odeint(func=generalized_gLV, y0=init_abun, t=microbe_data['Time'], args=(params_ls,))

def bayesian_glv_setup(params_init, microbe_data):

    #params = results.x  # least squares solution used to inform the priors
    with pm.Model() as model:
        # Priors
        r_1 = pm.TruncatedNormal("r_1", mu=params_init[0], sigma=0.1, lower=0, initval=params_init[0])
        r_2 = pm.TruncatedNormal("r_2", mu=params_init[1], sigma=0.01, lower=0, initval=params_init[1])
        ### Should I make gamma normally distributed? or at least gamma_2 for PC due to 
        gamma_1 = pm.TruncatedNormal("gamma_1", mu=params_init[2], sigma=0.1, lower=0, initval=params_init[2])
        gamma_2 = pm.TruncatedNormal("gamma_2", mu=params_init[3], sigma=0.01, lower=0, initval=params_init[3])
        a_1 = pm.TruncatedNormal("a_1", mu=params_init[4], sigma=1, upper=0, initval=params_init[4])
        a_2 = pm.TruncatedNormal("a_2", mu=params_init[5], sigma=1, upper=0, initval=params_init[5])
        sigma = pm.HalfNormal("sigma", 10)

        # Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([r_1, r_2, gamma_1, gamma_2, a_1, a_2])
        )

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=microbe_data[["EBwPC", "PCwEB"]].values)

        return model




def bayesian_glv_run(model, num_samples, chains):
    # Variable list to give to the sample step parameter
    vars_list = list(model.values_to_rvs.keys())[:-1]

    sampler = "DEMetropolisZ"
    tune = draws = num_samples
    with model:
        trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], tune=tune, draws=draws, chains=chains)
    trace = trace_DEMZ
    #az.summary(trace)

    return trace



# gamma_1, r_1 lower
# a_1 upper 


def posterior_param_samps(num_samples, glv_trace):
    
    param_dict = {'a_1' : {'upper_lim' : 0, 'lower_lim': -np.inf}, 'a_2' : {'upper_lim' : 0, 'lower_lim': -np.inf}, 'r_1' : {'upper_lim' : np.inf, 'lower_lim': 0}, 'r_2' : {'upper_lim' : np.inf, 'lower_lim': 0}, 'gamma_1' : {'upper_lim' : np.inf, 'lower_lim': 0}, 'gamma_2' : {'upper_lim' : np.inf, 'lower_lim': 0}}

    for key in param_dict:

        #print(params_names[i])
        #print(pm.summary(trace)[['mean','sd']].loc[params_names[i]])
        mu = pm.summary(glv_trace)[['mean','sd']].loc[key].iloc[0]
        sd = pm.summary(glv_trace)[['mean','sd']].loc[key].iloc[1] 
        
        lower = param_dict[key]['lower_lim']
        upper = param_dict[key]['upper_lim']

        N = num_samples

        samples = truncnorm.rvs((lower-mu)/sd,(upper-mu)/sd,loc = mu, scale=sd, size = N)

        param_dict[key]['samples'] = samples
        #samples

        plt.hist(samples, bins=20, density=True)
        plt.show()
        plt.close()

    return param_dict


