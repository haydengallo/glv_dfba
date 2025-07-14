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
import re



### implementing function to create dict for initial microbial abundance paired with model
### becomes important later for loading models from just list of model names etc. 

### make dict of dicts to keep track of model names, models, and abundances of species

def init_model_abun(model_names, models, init_abun, glv_out):
    
    model_abun_dict = {}

    for i in range(0, len(model_names)):

        model_abun_dict[model_names[i]] = {}
        model_abun_dict[model_names[i]]['model'] = models[i]
        model_abun_dict[model_names[i]]['abun'] = init_abun[i]
        ## can just initialize current growth rate at zero for time 0
        model_abun_dict[model_names[i]]['curr_gr_rt'] = 0
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
        
        #temp_media = make_media(model_abun_dict[key]['model'], media=supplied_media)
        #print('This is temp_media', temp_media)
        # set media conditions to be the temp_media
        #print('this is temp_media', temp_media)
        #model_abun_dict[key]['model'].medium = temp_media

        ### Manual setting of conditions 
        # Find all exchange reactions ending with "_b"
        ex_b_reactions = [rxn for rxn in model_abun_dict[key]['model'].reactions if rxn.id.startswith('EX_') and rxn.id.endswith('_b')]

        #print(f"Found {len(ex_b_reactions)} EX_*_b reactions:")

        # Set all lower bounds to 0, with exceptions from diet data
        for rxn in ex_b_reactions:
            if rxn.id in supplied_media['reaction'].to_list():  # Compare reaction ID, not object
                #print('yes - found in diet')
                # Fix the loc indexing - need to find the row where reaction matches
                #flux_value = temp_media[rxn.id]
                #rxn.lower_bound = -100.0#-1.0*flux_value
                rxn.lower_bound = -1.0*supplied_media[supplied_media['reaction'] == rxn.id]['fluxValue'].iloc[0]
                #print(f"Setting {rxn.id} lower bound to {rxn.lower_bound}")
            else:
                #print(f"Setting {rxn.id} lower bound from {rxn.lower_bound} to 0")
                rxn.lower_bound = 0



        #print('This is new media conditions for model', model_abun_dict[key]['model'].medium)
        ### here we take our temp media and apply to model.medium while also checking to make sure lengths are correct, if len(temp_media) != len(model.medium.keys) 
        ### then we determine which metabolites are missing and manually change them in model.reactions.get_by_id('reaction').lower_bound = -flux
        # think this was unnecessary and causing problems
        #keys_only_in_dict1 = temp_media.keys() - model_abun_dict[key]['model'].medium.keys()
        #keys_only_in_dict2 = model_abun_dict[key]['model'].medium.keys() - temp_media.keys()
        #print('This model:', key)
        #print(f"Keys only in temp_media: {keys_only_in_dict1}")
        #print(f"Keys only in model medium: {keys_only_in_dict2}")
        '''
        if len(temp_media) == len(model_abun_dict[key]['model'].medium.keys()):
            #print('good to go')
            continue

        else:

            if 'EX_ribflv(e)' in temp_media.keys():
                print('yes')
                model_abun_dict[key]['model'].reactions.get_by_id(id='EX_ribflv(e)').lower_bound = temp_media['EX_ribflv(e)']
                
            else:
                print('no')
        '''        
    return


### this is to be run after optimization step 

def model_opt_out(model_abun_dict, delta_t, pfba, met_pool_dict, glv_params, t_pt, model_names):

    #total_sys_uptake = {}
    #total_sys_secretion = {}

    ### ok here decide random order of optimizing models 
    ### determine num of keys in model_abun_dict
    model_names

    order_of_models = np.random.choice(range(0, len(model_names)), size=len(model_names), replace=False)
    

    #for key in model_abun_dict:
    for model_i in order_of_models:
        key = list(model_abun_dict.keys())[model_i]
        model_key = key
        print('Bacteria is:', key)

        # initialize this dict for every model and update met_pool_dict after every model
        total_sys_uptake = {}
        total_sys_secretion = {}

        if pfba == True:

            # optimize model i via pfba 
            temp_pfba = cobra.flux_analysis.pfba(model_abun_dict[key]['model'])
            #test = model_abun_dict[key]['model'].slim_optimize()
            #print(test)
            #print('Status of pfba:', temp_pfba.status)
            #print('pfba', temp_pfba)
            biomass_pattern = re.compile(r'(bio)', re.IGNORECASE)
            biomass_reactions = [rxn for rxn in model_abun_dict[key]['model'].reactions if biomass_pattern.search(rxn.id)]
            biomass_reactions = str(biomass_reactions[-1]).split(':')[0]
            #print('Upper_bound:',model_abun_dict[key]['model'].reactions.get_by_id(id = biomass_reactions).upper_bound)
            #print('Print medium used for optimization:',model_abun_dict[key]['model'].medium)
            ## seem to have some infeasible solutions, unclear why 
            # put fluxes in df for manipulation

            ### switch out regex string with 'EX_' if using AGORA models###
            temp_pfba_df = temp_pfba.to_frame().filter(regex='EX_.*_b$|bio', axis = 0)
            
            # filter out fluxes that are secreted
            # signs are flipped here compared to standard fba optimization in cobrapy so must change them

            # secreted should have negative sign to align with normal FBA
            test_secrete = temp_pfba_df[temp_pfba_df['fluxes'] > 0]
            #print('Test_secrete',test_secrete)

            # filter out fluxes that are taken up
            # uptake should have positive sign to align with normal FBA
            test_uptake = temp_pfba_df[temp_pfba_df['fluxes'] < 0]

            temp_uptake = np.abs(test_uptake['fluxes']) * delta_t * model_abun_dict[key]['abun']
            temp_secrete = -1.0*(test_secrete['fluxes']) * delta_t * model_abun_dict[key]['abun']   
            #print(temp_secrete)
            ## basically need to account for case when objective value is zero so need to add if statement to get around if this is the case
            ### got lines 147-151 from chatgpt
            filtered = test_secrete.filter(regex='bio', axis=0)
            if filtered.empty:
                pfba_obj_val = 0
            else:
                pfba_obj_val = filtered['fluxes'].iloc[0]
            #if model_abun_dict[key]['model'].reactions.get_by_id(id = 'biomassPan').upper_bound < 1e-8:
            #    pfba_obj_val = 0
            #else:
            #    pfba_obj_val = test_secrete.filter(regex='bio', axis=0)['fluxes'].iloc[0]
            if pfba_obj_val != model_abun_dict[key]['model'].reactions.get_by_id(id = biomass_reactions).upper_bound:
                print('Upper bound not reached:')
                print('Upper_bound:',model_abun_dict[key]['model'].reactions.get_by_id(id = biomass_reactions).upper_bound)
                print('Obj val', pfba_obj_val)
            ## have to do this after confirming that no negative values were created
            #model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + (delta_t*pfba_obj_val)
        
        else:

            test_secrete = model_abun_dict[key]['model'].summary().secretion_flux.loc[model_abun_dict[key]['model'].summary().secretion_flux['reaction'].filter(regex='EX_.*_b$|bio')]
            test_uptake = model_abun_dict[key]['model'].summary().uptake_flux.loc[model_abun_dict[key]['model'].summary().uptake_flux['reaction'].filter(regex='EX_.*_b$|bio')]

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
            
        ### Ok now need to move all of the append statements after when i need to check that there are no negative values in the met_pool_dict
        
        #model_abun_dict[key]['fba_biomass'].append(model_abun_dict[key]['abun'])
        #print(model_abun_dict[model_names[i]]['abun'])

        # dict of uptake for model x in i to n 
        uptake_dict = dict(zip(temp_uptake.index, temp_uptake))
        # add uptake_dict at time t to model storage
        #model_abun_dict[key]['flux_up'].append(uptake_dict.copy())
        # dict of sec for model x in i to n 
        secrete_dict = dict(zip(temp_secrete.index, temp_secrete))
        # add sec_dict at time t to model storage
        #model_abun_dict[key]['flux_sec'].append(secrete_dict.copy())

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

        ####################################
        ### Everything after here is new ###
        ####################################

        # ok basically need update metabolite pool after going through every bug i'd say 

        ## hold on to the met_pool_dict from the previous time step in case we need to go back to it 
        met_pool_dict_pre_t_st = met_pool_dict.copy()

        met_pool_dict  = update_met_pool(uptake_dict=total_sys_uptake, secrete_dict=total_sys_secretion, met_pool_dict=met_pool_dict)
        #print('this is met pool dict directly after update 1', met_pool_dict)
        ### basically here go if there are any negative values for this met_pool_dict, need to utilize the met_pool_dict_pre_t_st
        ### and reoptimize however many number of GEMs haven't been through secretion and uptake procedures,
        if any(value < 0 for value in met_pool_dict.values()):

            ### have to reset model key for model name
            key = model_key
            ### need to remove everything that had been appended previously but was no good,  
            
            # remove any of the negative fluxes 
            # actually nevermind don't need to remove negative fluxes, but rather need to grab met_pool_dict from previous time step
            #met_pool_dict = dict((k, v) for k, v in met_pool_dict.items() if v >= 0)
             # Step 1. Change media conditions of models 
            
            ### update met_pool_df
            met_pool_df = pd.DataFrame.from_dict(met_pool_dict_pre_t_st, orient='index',
                        columns=['fluxValue'])
            met_pool_df['fluxValue'] =  np.double(met_pool_df['fluxValue'])
            met_pool_df = met_pool_df.reset_index()
            met_pool_df.columns = ['reaction','fluxValue']

            print('This is met pool from previous time step because went into negatives', met_pool_df)
            

            ### Need to replace this for Kbase models that aren't translated to vmh ###
            ### All i need to change is the media for the model, nothing else, biomass bounds do not change
            #temp_media = make_media(model_abun_dict[key]['model'], media=met_pool_df)

            # set media conditions to be the temp_media

            #model_abun_dict[key]['model'].medium = temp_media

            ex_b_reactions = [rxn for rxn in model_abun_dict[key]['model'].reactions if rxn.id.startswith('EX_') and rxn.id.endswith('_b')]

            #print(f"Found {len(ex_b_reactions)} EX_*_b reactions:")

            # Set all lower bounds to 0, with exceptions from diet data
            for rxn in ex_b_reactions:
                if rxn.id in met_pool_df['reaction'].to_list():  # Compare reaction ID, not object
                    #print('yes - found in diet')
                    # Fix the loc indexing - need to find the row where reaction matches
                    #flux_value = temp_media[rxn.id]
                    #rxn.lower_bound = -100.0#-1.0*flux_value
                    rxn.lower_bound = -1.0*met_pool_df[met_pool_df['reaction'] == rxn.id]['fluxValue'].iloc[0]
                    #print(f"Setting {rxn.id} lower bound to {rxn.lower_bound}")
                else:
                    #print(f"Setting {rxn.id} lower bound from {rxn.lower_bound} to 0")
                    rxn.lower_bound = 0

            ### end of additions ###

            if pfba == True:

                # optimize model i via pfba 
                temp_pfba = cobra.flux_analysis.pfba(model_abun_dict[key]['model'])
                #print('Status of pfba:', temp_pfba.status)
                print('pfba', temp_pfba)
                biomass_pattern = re.compile(r'(bio)', re.IGNORECASE)
                biomass_reactions = [rxn for rxn in model_abun_dict[key]['model'].reactions if biomass_pattern.search(rxn.id)]
                biomass_reactions = str(biomass_reactions[-1]).split(':')[0]
                #print('Upper_bound:',model_abun_dict[key]['model'].reactions.get_by_id(id = biomass_reactions).upper_bound)
                #print('Upper_bound:',model_abun_dict[key]['model'].reactions.get_by_id(id = 'biomassPan').upper_bound)
                #print('Print medium used for optimization:',model_abun_dict[key]['model'].medium)
                ## seem to have some infeasible solutions, unclear why 
                # put fluxes in df for manipulation
                temp_pfba_df = temp_pfba.to_frame().filter(regex='EX_.*_b$|bio', axis = 0)
                
                # filter out fluxes that are secreted
                # signs are flipped here compared to standard fba optimization in cobrapy so must change them

                # secreted should have negative sign to align with normal FBA
                test_secrete = temp_pfba_df[temp_pfba_df['fluxes'] > 0]
                #print('Test_secrete',test_secrete)

                # filter out fluxes that are taken up
                # uptake should have positive sign to align with normal FBA
                test_uptake = temp_pfba_df[temp_pfba_df['fluxes'] < 0]

                temp_uptake = np.abs(test_uptake['fluxes']) * delta_t * model_abun_dict[key]['abun']
                temp_secrete = -1.0*(test_secrete['fluxes']) * delta_t * model_abun_dict[key]['abun']   
                #print(temp_secrete)
                ## basically need to account for case when objective value is zero so need to add if statement to get around if this is the case
                ### got lines 147-151 from chatgpt
                filtered = test_secrete.filter(regex='bio', axis=0)
                if filtered.empty:
                    pfba_obj_val = 0
                else:
                    pfba_obj_val = filtered['fluxes'].iloc[0]
                #if model_abun_dict[key]['model'].reactions.get_by_id(id = 'biomassPan').upper_bound < 1e-8:
                #    pfba_obj_val = 0
                #else:
                #    pfba_obj_val = test_secrete.filter(regex='bio', axis=0)['fluxes'].iloc[0]

                if pfba_obj_val != model_abun_dict[key]['model'].reactions.get_by_id(id = biomass_reactions).upper_bound:
                    print('Upper bound not reached:')
                    print('Upper_bound:',model_abun_dict[key]['model'].reactions.get_by_id(id = biomass_reactions).upper_bound)
                    print('Obj val', pfba_obj_val)

                ### Adjust abundances such that if negative growth rate abundance does decrease according to negative growth rate of MDSINE, but if positive abun adjusts according to FBA output

                if model_abun_dict[key]['curr_gr_rt'] < 0:

                    print('Model abun before negativity:', model_abun_dict[key]['abun'])
                    model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + (model_abun_dict[key]['abun']*delta_t*model_abun_dict[key]['curr_gr_rt'])
                    #model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + (model_abun_dict[key]['abun']*model_abun_dict[key]['curr_gr_rt'])
                    print('Model abun after negativity:', model_abun_dict[key]['abun'])

                    ## Basically if ever gets to zero or negative reset to a very small number such that always available to grow 

                    if model_abun_dict[key]['abun'] <= 0:

                        model_abun_dict[key]['abun'] = 1e-15

                else:

                    model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + (model_abun_dict[key]['abun']*delta_t*pfba_obj_val)
            
            else:

                test_secrete = model_abun_dict[key]['model'].summary().secretion_flux.loc[model_abun_dict[key]['model'].summary().secretion_flux['reaction'].filter(regex='EX_.*_b$|bio')]
                test_uptake = model_abun_dict[key]['model'].summary().uptake_flux.loc[model_abun_dict[key]['model'].summary().uptake_flux['reaction'].filter(regex='EX_.*_b$|bio')]

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

            met_pool_dict  = update_met_pool(uptake_dict=total_sys_uptake, secrete_dict=total_sys_secretion, met_pool_dict=met_pool_dict)
            #print('this is met pool dict directly after update 2', met_pool_dict)
        else:

            ### have to reset model key for model name
            key = model_key
            
            ### OLD ###
            #model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + (delta_t*pfba_obj_val)
            ###########

            ### Adjust abundances such that if negative growth rate abundance does decrease according to negative growth rate of MDSINE, but if positive abun adjusts according to FBA output

            if model_abun_dict[key]['curr_gr_rt'] < 0:

                print('Model abun before negativity:', model_abun_dict[key]['abun'])
                model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + (model_abun_dict[key]['abun']*delta_t*model_abun_dict[key]['curr_gr_rt'])
                #model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + (model_abun_dict[key]['abun']*model_abun_dict[key]['curr_gr_rt'])
                print('Model abun after negativity:', model_abun_dict[key]['abun'])
                ## Basically if ever gets to zero or negative reset to a very small number such that always available to grow 

                if model_abun_dict[key]['abun'] <= 0:

                    model_abun_dict[key]['abun'] = 1e-15

            else:
                if pfba == True:
                
                    model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + (model_abun_dict[key]['abun']*delta_t*pfba_obj_val)
                
                else:

                    model_abun_dict[key]['abun'] = model_abun_dict[key]['abun'] + model_abun_dict[key]['model'].summary()._objective_value * delta_t * model_abun_dict[key]['abun']

            model_abun_dict[key]['fba_biomass'].append(model_abun_dict[key]['abun'])
            # add uptake_dict at time t to model storage
            model_abun_dict[key]['flux_up'].append(uptake_dict.copy())
            # add sec_dict at time t to model storage
            model_abun_dict[key]['flux_sec'].append(secrete_dict.copy())

        # short term fix
        # could just say if crap is smaller than 1e-10 just drop it and combine that with smaller enough time steps should be able to get around
        # possibility of having negative flux values... I think 
        #met_pool_dict = dict((k, v) for k, v in met_pool_dict.items() if v > 1e-10)
        
    return total_sys_uptake, total_sys_secretion, met_pool_dict

###############################
### Changing Biomass Bounds ###
###############################

### Function for changing with gLV params ###

def change_biomass_bounds(model_abun_dict, glv_params, t_pt):

    temp_abun_list = []
    for key in model_abun_dict:
        temp_abun_list.append(model_abun_dict[key]['glv_out'][t_pt])
    
    #gr_rt_t_pt = generalized_gLV(temp_abun_list, t_pt, params=glv_params)
    gr_rt_t_pt = multi_spec_gLV(temp_abun_list, t_pt, params=glv_params)
    #print(gr_rt_t_pt)
### might be something wrong here, not quite sure, but only happens for DO so leads me to believe its index related
    for count, key in enumerate(model_abun_dict):
        if gr_rt_t_pt[count] > 0:
            model_abun_dict[key]['model'].reactions.get_by_id(id = 'biomassPan').upper_bound = gr_rt_t_pt[count]

        else:
            model_abun_dict[key]['model'].reactions.get_by_id(id = 'biomassPan').upper_bound = 0


### Function for changing with MDSINE output ###

def change_biomass_bounds_MDSINE(model_abun_dict, t_pt, MDSINE_rates):
    biomass_pattern = re.compile(r'(bio)', re.IGNORECASE)
    #gr_rt_t_pt = generalized_gLV(temp_abun_list, t_pt, params=glv_params)
    gr_rt_t_pt = MDSINE_rates.iloc[:,t_pt].tolist()
    #print(gr_rt_t_pt)
### might be something wrong here, not quite sure, but only happens for DO so leads me to believe its index related
    for count, key in enumerate(model_abun_dict):

        ### This is the string to use for the biomass function 

        biomass_reactions = [rxn for rxn in model_abun_dict[key]['model'].reactions if biomass_pattern.search(rxn.id)]
        biomass_reactions = str(biomass_reactions[-1]).split(':')[0]

        ### Add current growth rate to dict for each model

        model_abun_dict[key]['curr_gr_rt'] = gr_rt_t_pt[count]

        ### Now test if given growth rate should be supplied to upper bound of biomass reaction

        if gr_rt_t_pt[count] > 0:
            model_abun_dict[key]['model'].reactions.get_by_id(id = biomass_reactions).upper_bound = gr_rt_t_pt[count]

        else:
            model_abun_dict[key]['model'].reactions.get_by_id(id = biomass_reactions).upper_bound = 0




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

def static_dfba(list_model_names, list_models, initial_abundance, total_sim_time, num_t_steps, glv_out, glv_params, environ_cond, pfba, MDSINE_rates, Diet, time_points_feed, time_scaler):

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
    init_cond_dict = met_pool_dict
    met_pool_df = environ_cond

    ## add initial conditions to the list of dicts of metabolite pool

    met_pool_over_time.append(met_pool_dict.copy())

    # Step 0. Create model_abun_dict, main dictionary for everything

    model_abun_dict = init_model_abun(model_names=list_model_names,models = list_models, init_abun=initial_abundance, glv_out=glv_out)

    diet_dict = dict(zip(Diet['reaction'], Diet['fluxValue']))
    
    for i in range(0, num_t_steps):
        print('Time step: ', i)
        print('met_pool at beginning of time step', met_pool_df)
        # Step 1a. If this is continously fed, ie dietary conditions then add those at the specified time points 
        '''
        if time_points_feed == None:# or Diet == None:
            print('not cont fed')

        elif i in time_points_feed:

            #met_pool_df = met_pool_df.add(Diet, fill_value=0)

            # Stack the two dataframes
            met_pool_df = pd.concat([met_pool_df, Diet])

            # Group by reaction and sum
            met_pool_df = met_pool_df.groupby('reaction', as_index=False).sum()
        '''



        # Step 1. Change media conditions of models 
        change_media(model_abun_dict= model_abun_dict, supplied_media= met_pool_df)

        #change_media(model_abun_dict= model_abun_dict, supplied_media= environ_cond)

        # Step 2. Change upper bounds of biomass growth rate for each model
        # Step 2a. This is for when utilizing output from MDSINE and not utilizing glv_params
        if glv_params == None:

            #print('Using MDSINE rates')
            change_biomass_bounds_MDSINE(model_abun_dict=model_abun_dict, t_pt=i, MDSINE_rates=MDSINE_rates)
        else:
        # Step 2b: This is for when utilizing output from gLV directly such that params are needed    
            change_biomass_bounds(model_abun_dict=model_abun_dict, glv_params=glv_params, t_pt=i)





        # Step 3. Run actual FBA/optimization

        opt_model(model_abun_dict=model_abun_dict, pfba=pfba)

        # Step 4. Adjust model optimization output fluxes based on abundance and time step size

        total_sys_uptake, total_sys_secretion, met_pool_dict = model_opt_out(model_abun_dict=model_abun_dict, delta_t= (total_sim_time/num_t_steps), pfba=pfba, glv_params=glv_params, t_pt=i, model_names = list_model_names, met_pool_dict=met_pool_dict)

        #print('this is met pool dict returned after model opt out step', met_pool_dict)
        ### should actually move this update_met_pool function within model_opt_out so as to be done after every optimization of species 
        # Step 5. Update total metabolite pool
        #met_pool_dict  = update_met_pool(uptake_dict=total_sys_uptake, secrete_dict=total_sys_secretion, met_pool_dict=met_pool_dict)

        ### Here should include a function just in case to go through met_pool_dict and remove any keys with negative values
        # https://www.geeksforgeeks.org/python-filter-the-negative-values-from-given-dictionary/
        ## got from link above
        # could just say if crap is smaller than 1e-10 just drop it and combine that with smaller enough time steps should be able to get around
        # possibility of having negative flux values... I think 
        ### hmm but what if it gets rid of things that are secreted in small amounts at beginning????
        met_pool_dict = dict((k, v) for k, v in met_pool_dict.items() if v > 1e-30)

        #print('This is updated met_pool_dict:\n',met_pool_dict)
        
        #original
        #met_pool_over_time.append(met_pool_dict.copy())
        #print(len(met_pool_over_time))

        ### update met_pool_df
        met_pool_df = pd.DataFrame.from_dict(met_pool_dict, orient='index',
                       columns=['fluxValue'])
        met_pool_df['fluxValue'] =  np.double(met_pool_df['fluxValue'])
        met_pool_df = met_pool_df.reset_index()
        met_pool_df.columns = ['reaction','fluxValue']
        '''
        if i > time_scaler/12:
            met_pool_lag = int(time_scaler/12)
            #Diet_dict = Diet.set_index('reaction').to_dict()
            #print('This is met pool at time ', i-met_pool_lag, ':', met_pool_over_time[i-met_pool_lag])
            #ratios = {k: (1-((diet_dict[k] / met_pool_over_time[i-met_pool_lag][k])/(1))) for k in diet_dict if k in met_pool_over_time[i-met_pool_lag]}
            # Saturating limiter: small when concentration >> diet, near 1 when diet ~ concentration
            ratios = {
                k: max(0.98, min(0.999, diet_dict[k] / (met_pool_over_time[i - met_pool_lag][k] + diet_dict[k] + 1e-8)))
                for k in diet_dict
                if k in met_pool_over_time[i - met_pool_lag]
            }


            extra_keys = [k for k in met_pool_over_time[i-met_pool_lag] if k not in diet_dict]

            #for key in ratios:
            #    old = met_pool_df.loc[met_pool_df['reaction'] == key, 'fluxValue'].iloc[0]
            for key, ratio in ratios.items():
                met_pool_df.loc[met_pool_df['reaction'] == key, 'fluxValue'] *= ratio

                #print('Old met conc', old)
                if ratios[key] < 0.999:
                    met_pool_df.loc[met_pool_df['reaction'] == key, 'fluxValue'] *= ratios[key]
                    #new = met_pool_df.loc[met_pool_df['reaction'] == key, 'fluxValue'].iloc[0]
                    #print('New met conc', new)
                else:
                    met_pool_df.loc[met_pool_df['reaction'] == key, 'fluxValue'] *= 0.999
                    #new = met_pool_df.loc[met_pool_df['reaction'] == key, 'fluxValue'].iloc[0]     
                    #print('New met conc', new)

            for met in extra_keys:
                #old = met_pool_df.loc[met_pool_df['reaction'] == met, 'fluxValue'].iloc[0]
                #print('Old met conc', old)
                met_pool_df.loc[met_pool_df['reaction'] == met, 'fluxValue'] *= 0.999
                #new = met_pool_df.loc[met_pool_df['reaction'] == met, 'fluxValue'].iloc[0]
                #print('New met conc', new)

            #for key in ratios:
            #    print('Old met conc', met_pool_df[met_pool_df['reaction'] == key]['fluxValue'].iloc[0])
            #    met_pool_df[met_pool_df['reaction'] == key]['fluxValue'].iloc[0]=ratios[key]*met_pool_df[met_pool_df['reaction'] == key]['fluxValue'].iloc[0]
            #    print('New met conc', met_pool_df[met_pool_df['reaction'] == key]['fluxValue'].iloc[0])

            #for met in extra_keys:
            #    print('Old met conc', met_pool_df[met_pool_df['reaction'] == met]['fluxValue'].iloc[0])
            #    met_pool_df[met_pool_df['reaction'] == met]['fluxValue'].iloc[0]=.999*met_pool_df[met_pool_df['reaction'] == met]['fluxValue'].iloc[0]
            #    print('New met conc', met_pool_df[met_pool_df['reaction'] == met]['fluxValue'].iloc[0])

        # Putting cont feeding at the end of every timepoint, i.e. for next time point and to get around adding at zeroth time
        # Stack the two dataframes
        ### Decay rate
        #print('Metabolite pool before decay term', met_pool_df)
        else:
            met_pool_df['fluxValue'] = .999*met_pool_df['fluxValue']
        '''
        met_pool_df['fluxValue'] = (1-(1/384))*met_pool_df['fluxValue']
        #print('Metabolite pool after decay term', met_pool_df)
        met_pool_df = pd.concat([met_pool_df, Diet])

        # Group by reaction and sum
        met_pool_df = met_pool_df.groupby('reaction', as_index=False).sum()

        ## naive threshold, if anything is greater than 100 set it back to 100
        #met_pool_df['fluxValue'] = met_pool_df['fluxValue'].clip(upper=10) 
        met_pool_dict = dict(zip(met_pool_df['reaction'], met_pool_df['fluxValue']))
        met_pool_over_time.append(met_pool_dict.copy())
        #met_pool_dict['EX_cpd00076_b'] = 2.0


        #print('Updated met_pool', met_pool_df)

        ### Think i needed to overwrite environ_cond met_pool_dict

    
    return met_pool_over_time, model_abun_dict

######################
### Loss Functions ###
######################

# function that calculates residuals based on a given theta
def ode_model_resid(params, microbe_data, init_abun):
    return (
        microbe_data.iloc[:,1:] - odeint(generalized_gLV, y0 = init_abun, t=microbe_data['Time'], args = (params,))
    ).values.flatten()

### this function for computing SSE when we are trying to fit gLV to multiple datasets, to be used in conjunction with a call of scipy.optimize.minimize to determine weakly informative priors for MCMC fit of gLV
def total_loss(params, microbe_data_list, abun_list):
    total_loss_sq = 0
    for i in range(0, len(abun_list)):
        loss = ode_model_resid(params=params, microbe_data=microbe_data_list[i], init_abun=abun_list[i])
        total_loss_sq += np.sum(loss**2)

    return total_loss_sq


def ode_model_resid_multi(params, microbe_data, init_abun):
    return (
        microbe_data.iloc[:,1:] - odeint(multi_spec_gLV, y0 = init_abun, t=microbe_data['Time'], args = (params,))
    ).values.flatten()

def total_loss_multi(params, microbe_data_list, abun_list):
    total_loss_sq = 0
    for i in range(0, len(abun_list)):
        loss = ode_model_resid_multi(params=params, microbe_data=microbe_data_list[i], init_abun=abun_list[i])
        total_loss_sq += np.sum(loss**2)

    return total_loss_sq


###########################
### gLV implementations ###
###########################



### this function for computing SSE when we are trying to fit gLV to multiple datasets, to be used in conjunction with a call of scipy.optimize.minimize to determine weakly informative priors for MCMC fit of gLV
def total_loss(params, microbe_data_list, abun_list):
    total_loss_sq = 0
    for i in range(0, len(abun_list)):
        loss = ode_model_resid(params=params, microbe_data=microbe_data_list[i], init_abun=abun_list[i])
        total_loss_sq += np.sum(loss**2)

    return total_loss_sq


### implementing naive gLV

def generalized_gLV(X, t, params):
    x, y = X 
    r_1, r_2, gamma_1, gamma_2, a_1, a_2 =  params

    # EB concentration integrated overtime
    dxdt = x * (r_1 + a_1*x + gamma_1*y)
    
    # P.copri concentration integrated overtime
    dydt = y * (r_2 + gamma_2*x + a_2*y)

    return [dxdt, dydt]

def multi_spec_gLV(X, t, params):

    ### hmm need to expand this to account for N species, i guess for now I can just expand such that I account for 3 species 
    x, y, z = X 
    r_1, r_2, r_3, gamma_EP, gamma_ED, gamma_PE, gamma_PD, gamma_DE, gamma_DP, a_1, a_2, a_3 =  params

    # EB concentration integrated overtime
    dxdt = x * (r_1 + a_1*x + gamma_EP*y + gamma_ED*z)

    # P. copri concentration integrated overtime 
    dydt = y * (r_2 + a_2*y + gamma_PE*x + gamma_PD*z)

    # Dorea concentration integrated overtime 
    dzdt = z * (r_3 + a_3*z + gamma_DE*x + gamma_DP*y)


    return [dxdt, dydt, dzdt]


def multi_spec_gLV(X, t, params):

    ### hmm need to expand this to account for N species, i guess for now I can just expand such that I account for 3 species 
    x, y, z = X 
    r_1, r_2, r_3, gamma_EP, gamma_ED, gamma_PE, gamma_PD, gamma_DE, gamma_DP, a_1, a_2, a_3 =  params

    # EB concentration integrated overtime
    dxdt = x * (r_1 + a_1*x + gamma_EP*y + gamma_ED*z)

    # P. copri concentration integrated overtime 
    dydt = y * (r_2 + a_2*y + gamma_PE*x + gamma_PD*z)

    # Dorea concentration integrated overtime 
    dzdt = z * (r_3 + a_3*z + gamma_DE*x + gamma_DP*y)


    return [dxdt, dydt, dzdt]



def ls_glv_fit(init_abun, params, total_sim_time, time_steps, microbe_data):

    init_abun = np.array(init_abun)
    
    #results = least_squares(ode_model_resid, x0=params, bounds=([0, 0, 0, -10, -10, -10], [10, 10, 10, 10, 0, 0]), xtol = 1e-15, args = (microbe_data, init_abun))

    # for some reason need to change tolerances and differntiate step sizes for the least_squares solver on hpc as compared to my local machine
    # nevermind, seems fine now, there's definitely some weird stuff going on with this least squares function 
    # basically least squares returns different answers on mac and linux and because of bounds set on parameters both in lsq and for priors of bayesian inference, then there isn't congruency and error thrown
    results = least_squares(ode_model_resid, x0=params, bounds=([0, 0, 0, 0, -10, -10], [10, 10, 10, 10, 0, 0]), diff_step=1e-12, xtol = 1e-15, args = (microbe_data, init_abun))

    #results = least_squares(ode_model_resid, x0=params, bounds=([0, 0, 0, -10, -10, -10], [10, 10, 10, 10, 0, 0]), diff_step=1e-6, ftol=1e-7, xtol=1e-7, gtol=1e-7, args = (microbe_data, init_abun))

    #results = least_squares(ode_model_resid, x0=params, bounds=([0, 0, 0, -10, -10, -10], [10, 10, 10, 10, 0, 0]), args = (microbe_data, init_abun), verbose = 1)
    print(results.x)

    params = results.x
    
    time = np.arange(0, int(total_sim_time+1), int(total_sim_time/time_steps))

    x_y = odeint(generalized_gLV, y0 = init_abun, t=time, args = (params,))

    return x_y, params, time
    




### implementing bayesian inference for gLV dynamics 
### based on tutorial found here:
### https://www.pymc.io/projects/examples/en/latest/ode_models/ODE_Lotka_Volterra_multiple_ways.html

# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector, pt.dvector, pt.dmatrix], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(params_ls, init_abun, microbe_data):

    time_values = microbe_data[:,0]
    #print(time_values)

    return odeint(func=generalized_gLV, y0=init_abun, t=time_values, args=(params_ls,))

def bayesian_glv_setup(params_init, microbe_data, init_abun):

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


        #Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([r_1, r_2, gamma_1, gamma_2, a_1, a_2]), pm.math.stack(init_abun), pt.as_tensor(microbe_data.values)
        )

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=microbe_data[["EBwPC", "PCwEB"]].values)
        # can i add second pm.norm liek Y_obs_2 where its the monoculture data

        return model




def bayesian_glv_run(model, num_samples, chains):
    # Variable list to give to the sample step parameter
    vars_list = list(model.values_to_rvs.keys())[:-1]

    sampler = "DEMetropolisZ"
    tune = draws = num_samples
    with model:
        trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], tune=tune, draws=draws, chains=chains, cores = 10)
    trace = trace_DEMZ
    #az.summary(trace)

    return trace


def bayesian_glv_setup_multi(params_init, microbe_data_list, abun_list):

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

        #for i in range(0, len(microbe_data_list)):
        #    # Likelihood for each dataset, with a unique sigma per dataset
        #    sigma = pm.HalfNormal(f"sigma_{i}", 10)

        # Loop over each dataset
        for i, microbe_data in enumerate(microbe_data_list):
            # ODE solution for each dataset
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([r_1, r_2, gamma_1, gamma_2, a_1, a_2]), pm.math.stack(abun_list[i]), pt.as_tensor(microbe_data.values)
            )
            sigma = pm.HalfNormal(f"sigma_{i}", 10)

            pm.Normal(f"Y_obs_{i}", mu=ode_solution, sigma=sigma, observed=microbe_data.iloc[:,1:].values)

        return model




def bayesian_glv_run_multi(model, num_samples, chains):
    # Variable list to give to the sample step parameter
    #vars_list = list(model.values_to_rvs.keys())[:11]+[list(model.values_to_rvs.keys())[12]]+[list(model.values_to_rvs.keys())[14]]+[list(model.values_to_rvs.keys())[16]]+[list(model.values_to_rvs.keys())[18]]

    # this seems to be a better way to variable select, basically the variables we always want ends with '__' and if we just filter for names that have this we can easily grab correct variable lists
    # https://stackoverflow.com/questions/15403021/regular-expression-to-filter-list-of-strings-matching-a-pattern

    vars_list = [e for e in list(model.values_to_rvs.keys()) if e.name.endswith('__')]

    sampler = "DEMetropolisZ"
    tune = draws = num_samples
    with model:
        trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], tune=tune, draws=draws, chains=chains, cores = 10)
    trace = trace_DEMZ
    #az.summary(trace)

    return trace




def bayesian_glv_setup_multi(params_init, microbe_data_list, abun_list):

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

        #for i in range(0, len(microbe_data_list)):
        #    # Likelihood for each dataset, with a unique sigma per dataset
        #    sigma = pm.HalfNormal(f"sigma_{i}", 10)

        # Loop over each dataset
        for i, microbe_data in enumerate(microbe_data_list):
            # ODE solution for each dataset
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([r_1, r_2, gamma_1, gamma_2, a_1, a_2]), pm.math.stack(abun_list[i]), pt.as_tensor(microbe_data.values)
            )
            sigma = pm.HalfNormal(f"sigma_{i}", 10)

            pm.Normal(f"Y_obs_{i}", mu=ode_solution, sigma=sigma, observed=microbe_data.iloc[:,1:].values)

        return model




def bayesian_glv_run_multi(model, num_samples, chains):
    # Variable list to give to the sample step parameter
    vars_list = list(model.values_to_rvs.keys())[:7]+[list(model.values_to_rvs.keys())[8]]+[list(model.values_to_rvs.keys())[10]]

    sampler = "DEMetropolisZ"
    tune = draws = num_samples
    with model:
        trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], tune=tune, draws=draws, chains=chains, cores = 10)
    trace = trace_DEMZ
    #az.summary(trace)

    return trace


@as_op(itypes=[pt.dvector, pt.dvector, pt.dmatrix], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix_three_spec(params_ls, init_abun, microbe_data):

    time_values = microbe_data[:,0]
    #print(time_values)

    return odeint(func=multi_spec_gLV, y0=init_abun, t=time_values, args=(params_ls,))

def bayesian_glv_setup_three_spec(params_init, microbe_data_list, abun_list):

    #params = results.x  # least squares solution used to inform the priors
    with pm.Model() as model:
        # Priors
        r_1 = pm.TruncatedNormal("r_1", mu=params_init[0], sigma=0.1, lower=0, initval=params_init[0])
        r_2 = pm.TruncatedNormal("r_2", mu=params_init[1], sigma=0.1, lower=0, initval=params_init[1])
        r_3 = pm.TruncatedNormal("r_3", mu=params_init[2], sigma=0.1, lower=0, initval=params_init[2])

        gamma_EP = pm.TruncatedNormal("gamma_EP", mu=params_init[3], sigma=0.1, lower=0, initval=params_init[3])
        #gamma_ED = pm.TruncatedNormal("gamma_ED", mu=params_init[4], sigma=0.1, lower=0, upper=0, initval=params_init[4])
        gamma_PE = pm.TruncatedNormal("gamma_PE", mu=params_init[5], sigma=0.1, lower=-.1, initval=params_init[5])
        gamma_PD = pm.TruncatedNormal("gamma_PD", mu=params_init[6], sigma=0.1, lower=0, initval=params_init[6])
        #gamma_DE = pm.TruncatedNormal("gamma_DE", mu=params_init[7], sigma=0.1, lower=0, upper=0, initval=params_init[7])
        gamma_DP = pm.TruncatedNormal("gamma_DP", mu=params_init[8], sigma=0.1, upper=0, initval=params_init[8])

        a_1 = pm.TruncatedNormal("a_1", mu=params_init[9], sigma=.1, upper=0, initval=params_init[9])
        a_2 = pm.TruncatedNormal("a_2", mu=params_init[10], sigma=.1, upper=0, initval=params_init[10])
        a_3 = pm.TruncatedNormal("a_3", mu=params_init[11], sigma=.1, upper=0, initval=params_init[11])

        #for i in range(0, len(microbe_data_list)):
        #    # Likelihood for each dataset, with a unique sigma per dataset
        #    sigma = pm.HalfNormal(f"sigma_{i}", 10)

        # Loop over each dataset
        for i, microbe_data in enumerate(microbe_data_list):
            # ODE solution for each dataset
            ode_solution = pytensor_forward_model_matrix_three_spec(
                pm.math.stack([r_1, r_2, r_3, gamma_EP, 0, gamma_PE, gamma_PD, 0, gamma_DP, a_1, a_2, a_3]), pm.math.stack(abun_list[i]), pt.as_tensor(microbe_data.values)
            )
            sigma = pm.HalfNormal(f"sigma_{i}", 10)

            pm.Normal(f"Y_obs_{i}", mu=ode_solution, sigma=sigma, observed=microbe_data.iloc[:,1:].values)

        return model




def bayesian_glv_run_three_spec(model, num_samples, chains):
    # Variable list to give to the sample step parameter
    #vars_list = list(model.values_to_rvs.keys())[:11]+[list(model.values_to_rvs.keys())[12]]+[list(model.values_to_rvs.keys())[14]]+[list(model.values_to_rvs.keys())[16]]+[list(model.values_to_rvs.keys())[18]]

    # this seems to be a better way to variable select, basically the variables we always want ends with '__' and if we just filter for names that have this we can easily grab correct variable lists
    # https://stackoverflow.com/questions/15403021/regular-expression-to-filter-list-of-strings-matching-a-pattern

    vars_list = [e for e in list(model.values_to_rvs.keys()) if e.name.endswith('__')]
    
    sampler = "DEMetropolisZ"
    tune = draws = num_samples
    with model:
        trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], tune=tune, draws=draws, chains=chains, cores = chains)
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



def posterior_param_samps_multi(num_samples, glv_trace):
    
    param_dict = {'a_1' : {'upper_lim' : 0, 'lower_lim': -np.inf}, 'a_2' : {'upper_lim' : 0, 'lower_lim': -np.inf}, 'a_3' : {'upper_lim' : 0, 'lower_lim': -np.inf}, 'r_1' : {'upper_lim' : np.inf, 'lower_lim': 0}, 'r_2' : {'upper_lim' : np.inf, 'lower_lim': 0}, 'r_3' : {'upper_lim' : np.inf, 'lower_lim': 0},
                   'gamma_DP' : {'upper_lim' : 0, 'lower_lim': -np.inf}, 'gamma_EP' : {'upper_lim' : np.inf, 'lower_lim': 0}, 'gamma_PD' : {'upper_lim' : np.inf, 'lower_lim': -np.inf}, 'gamma_PE' : {'upper_lim' : np.inf, 'lower_lim': -np.inf}}

    for key in param_dict:
        print(key)
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

        #plt.hist(samples, bins=20, density=True)
        #plt.show()
        #plt.close()

    return param_dict


