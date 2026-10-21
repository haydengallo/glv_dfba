### Hayden Gallo
### Bucci Lab
### Running gLV-FBA on Venturelli data, simulating all co-culture combos -- KBase GEMs, HPC (LSF) version
###
### Combines:
###   - the KBase-model version of gLV_FBA_Venturelli_sims (ModelSEED media IDs, SCFA recording and
###     plotting through SCFA_EXCHANGES), and
###   - the LSF batching from gLV_FBA_Venturelli_sims_panGenus_parallel_hpc.py
###
### Usage:
###   python this_script.py --sim-num N          # submission mode: one LSF job per experiment per draw (N draws)
###   python this_script.py --sim-idx I --sim-num J   # worker mode (what each LSF job runs): experiment I, draw J
###   python this_script.py --aggregate --sim-num N   # collect all result pickles into final CSVs
###
### With pfba (FLUX_SAMPLING = False) every draw is identical, so use --sim-num 1.


import numpy as np
import cobra
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")   # no display on compute nodes
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

import joblib
import multiprocessing
from scipy.stats import truncnorm
import copy
from pathlib import Path
import subprocess

import argparse
import re

from helper_functions import *

from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import json
import logging
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

import pickle

import warnings
warnings.filterwarnings("ignore")

### since we are on hpc need to explicitly set the gurobi lic environment variable
os.environ["GRB_LICENSE_FILE"] = "/share/pkg/gurobi/11.0.2/lib/gurobi.lic"

logging.getLogger("cobra").setLevel(logging.ERROR)

#############################################
### Parameters to set prior to simulation ###
#############################################

data_dir = '/home/hayden.gallo-umw/DySCoMeMo_flux_sampling_sims/Venturelli_in_vitro/'
data_input_dir = data_dir + 'data_input'

### Downloaded KBase models. Files are matched to strains by the text before the first "_"
### in the file name (e.g. BT_monoculture_gf.xml -> BT). Supported: .json, .xml, .sbml, .mat
cobra_models_dir_path = data_input_dir + '/kbase_recon_10_20_2026_no_succ_import_BT_BL_fix'
cobra_models_dir = Path(cobra_models_dir_path)
BIOMASS_ID = 'bio1'

### Exchange reaction naming in the downloaded models
EX_FMT = "EX_{cpd}_b"

MONOCULTURES_ONLY = False 
DFBA_TRADITIONAL = False

scal_fac = 5
t_steps = 48

test_num = 43
multi_t_pt_dataset = 'no'

### Simulation mode
FLUX_SAMPLING = True            # False = pfba (as in the desktop KBase script); True = flux sampling
USE_INTERPOLATED_TRAJ = False     # single-timepoint data: use interpolated MDSINE2 trajectories (desktop behavior)
AGORA_MODELS_FLAG = 'no'        # passed to static_dfba as in your desktop script; set to what helper_functions expects for KBase models
MAKE_PLOTS = False
SAVE_NPY = False                  # save met_pool.npy / abun_save.npy per experiment

if multi_t_pt_dataset == 'yes':
    test_name = 'KBase_model_sims_multi_t_pt'
else:
    test_name = 'KBase_model_sims'

notes = ('with scaling factor of 5, testing with all experiments and testing to see if i can quickly convert dysco over to using traditional dfba growth maximization had to fix BT and BL, increasing thiamin etc, running monocultures, using interpolated trajectories and 10/20 models, add non limiting amount of co2 testing monoculture of 10/13/26 models, adding coprococcus comes from earlier using 10/8/26 recons all 5 sims experiments KBase monoculture-fitted GEMs (growth-rate + exometabolite gapfilling), ModelSEED base media, monoculture only '
         f'{"flux sampling" if FLUX_SAMPLING else "pfba"}, HPC batch')

plot_dir_path_base = data_dir + test_name + '/test_' + str(test_num)
os.makedirs(Path(plot_dir_path_base), exist_ok=True)


##########################################################################
### SCFAs: experimental column name -> ModelSEED exchange reaction ID ###
##########################################################################
SCFA_CPDS = {
    'Butyrate':  'cpd00211',
    'Acetate':   'cpd00029',
    'Lactate':   'cpd00159',   # L-lactate
    'Succinate': 'cpd00036',
}
SCFA_EXCHANGES = {name: EX_FMT.format(cpd=cpd) for name, cpd in SCFA_CPDS.items()}
SCFA_PALETTE = {'Butyrate': 'tab:orange', 'Acetate': 'tab:green',
                'Lactate': 'tab:blue', 'Succinate': 'tab:red'}

def scfa_table(met_pool_over_time):
    """Time x SCFA DataFrame with experimental names as columns; 0 where never produced."""
    df = pd.DataFrame(met_pool_over_time).fillna(0)
    out = pd.DataFrame(index=df.index)
    for name, ex in SCFA_EXCHANGES.items():
        out[name] = df[ex] if ex in df.columns else 0.0
    return out


### define generalized Lotka-Volterra function
def gLV(t, init_abun, paired_growth_matrix, basal_grow):
    return init_abun * (basal_grow + np.dot(paired_growth_matrix, init_abun))


### Metadata from metadata_2019_06_17.py (Clark et al. 2021)

numspecies = 25
allspecies = ['ER','FP','AC','CC','RI','EL','CH','DP','BH','CA','PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC','BY','DF','BL','BP','BA']

namedict = {
   'BA': 'Bifidobacterium_adolescentis_ATCC_15703_NC_008618',
   'CA': 'Collinsella_aerofaciens_ATCC_25986',
   'BT': 'Bacteroides_thetaiotaomicron_VPI-5482_NC_004663',
   'BU': 'Bacteroides_uniformis_ATCC_8492',
   'PC': 'Prevotella_copri_DSM_18205',
   'AC': 'Anaerostipes_caccae_DSM_14662_4',
   'BH': 'Blautia_hydrogenotrophica_DSM_10507',
   'CC': 'Coprococcus_comes_1.0.1_Cont2276_NZ_ABVR01000038',
   'CG': 'Clostridium_asparagiforme_DSM_15981_C_asparagiforme_1.0_Cont7.2_NZ_ACCJ01000522',
   'ER': 'Eubacterium_rectale_ATCC_33656_NC_012781',
   'DP': 'Desulfovibrio_piger_ATCC_29098',
   'EL': 'Eggerthella_lenta_DSM_2243_NC_013204',
   'BY': 'Bacteroides_cellulosilyticus_DSM_14838_1.0_Cont4.1_NZ_ACCH01000108',
   'BF': 'Bacteroides_fragilis_NCTC_9343',
   'CD': 'Clostridioides_difficile',
   'RI': 'Roseburia_intestinalis_L1_82',
   'BP': 'Bifidobacterium_pseudocatenulatum_DSM20438',
   'BV': 'Bacteroides_vulgatus_ATCC_8482_NC_009614',
   'CH': 'Clostridium_hiranonis_DSM_13275',
   'DF': 'Dorea_formicigenerans_ATCC_27755',
   'CS': 'Clostridium_scindens_ATCC_35704',
   'PJ': 'Parabacteroides_johnsonii_DSM_18315_NZ_ABYH01000014',
   'FP': 'Faecalibacterium_prausnitzii_A2_165_NZ',
   'EH': 'Eubacterium_hallii_DSM_3353_1.0_Cont383.1_NZ_ACEP01000116',
   'EC': 'Escherichia_coli',
   'BC': 'Bacteroides_caccae_ATCC_43185',
   'HB': 'Holdemanella_biformis_DSM_3989',
   'BO': 'Bacteroides_ovatus_ATCC_8483',
   'DL': 'Dorea_longicatena_DSM_13814',
   'BL': 'Bifidobacterium_longum_subsp_infantis',
   'B.cereus': 'Bacillus_cereus'
   }


## load in the data needed for running simulations

int_matrix = np.load(data_input_dir + '/int_matrix_reindexed_12_11_25.npy', allow_pickle=True)
growth_rates = np.load(data_input_dir + '/growth_rates_12_11_25.npy', allow_pickle=True)

m2_init_abun = pd.read_csv(data_input_dir + '/m2_time_series_init_no_mets.csv')
m2_init_abun = m2_init_abun.set_index('Experiments')

### must use the m2 final with no mets b/c species are ordered correctly here and not in m2 final with mets
m2_final_abun_bac = pd.read_csv(data_input_dir + '/m2_time_series_final_no_mets.csv', index_col=0)

m2_final_abun_met = pd.read_csv(data_input_dir + '/m2_time_series_final_w_mets.csv', index_col=0)
m2_final_abun_met = m2_final_abun_met.set_index('Experiments')

### multifunctional dataset
multi_func_data = pd.read_csv(data_input_dir + '/2021_02_19_MultifunctionalDynamicData.csv', index_col=0)

multi_func_data_filt = multi_func_data.filter(regex='_OD|Community|Time|Butyrate|Succinate|Acetate|Lactate', axis=1)
multi_func_data_filt.columns = list(multi_func_data_filt.columns.str.split('_').str[0])
multi_func_data_filt = multi_func_data_filt.rename(columns={'Community': 'Experiments'})
multi_func_data_filt_reorder = multi_func_data_filt.set_index('Experiments')
multi_func_data_filt_reorder = multi_func_data_filt_reorder.drop(['Time','Butyrate','Succinate','Acetate','Lactate'], axis=1)[m2_init_abun.columns.to_list()]
multi_func_data_filt_reorder['Time'] = multi_func_data_filt['Time'].to_list()
multi_func_data_filt_reorder['Butyrate'] = multi_func_data_filt['Butyrate'].to_list()
multi_func_data_filt_reorder['Succinate'] = multi_func_data_filt['Succinate'].to_list()
multi_func_data_filt_reorder['Acetate'] = multi_func_data_filt['Acetate'].to_list()
multi_func_data_filt_reorder['Lactate'] = multi_func_data_filt['Lactate'].to_list()
multi_func_data_filt_reorder_initial = multi_func_data_filt_reorder[multi_func_data_filt_reorder['Time'] == 0.0].drop(['Time','Butyrate','Succinate','Acetate','Lactate'], axis=1)
multi_func_data_filt_reorder_final = multi_func_data_filt_reorder[multi_func_data_filt_reorder['Time'] != 0.0]

m2_final_abun_met = m2_final_abun_met.drop(columns=['Time'])

interpolated_data = None
Venturelli_long_MDSINE2_interpolated_data = None

if multi_t_pt_dataset == 'yes':
    m2_init_abun = pd.read_csv(data_input_dir + '/multi_t_pt_for_glv_fba_init_vals_MDSINE2.csv', index_col=0)
    ### VERY IMPORTANT: reindex MDSINE2 data to align with the experimental data, and reorder columns
    m2_init_abun = m2_init_abun.reindex(multi_func_data_filt_reorder_initial.index.to_list())
    m2_init_abun = m2_init_abun[allspecies]

    m2_final_abun_met = multi_func_data_filt_reorder_final[['Time', 'Butyrate', 'Acetate', 'Lactate', 'Succinate']]
    m2_final_abun_bac = multi_func_data_filt_reorder.drop(columns=['Butyrate', 'Acetate', 'Lactate', 'Succinate'])

    Venturelli_long_MDSINE2_interpolated_data = np.load(data_input_dir + '/Venturelli_long_MDSINE2_interpolated_data.npy', allow_pickle=True).item()
else:
    m2_final_abun_met = m2_final_abun_met[['Butyrate', 'Acetate', 'Lactate', 'Succinate']]
    if USE_INTERPOLATED_TRAJ:
        with open(data_input_dir + '/interpolated_values.pkl', 'rb') as file:
            interpolated_data = pickle.load(file)

time = np.linspace(0, t_steps, t_steps + 1)
args = (int_matrix, growth_rates)


### Output templates (NaN = no result for that experiment)
bac_abun_predict_final_df = m2_final_abun_bac.copy()
met_abun_predict_final_df = m2_final_abun_met.copy()
if multi_t_pt_dataset == 'yes':
    met_abun_predict_final_df = met_abun_predict_final_df.drop(columns=['Time'])
    bac_abun_predict_final_df = bac_abun_predict_final_df.drop(columns=['Time'])
bac_abun_predict_final_df = bac_abun_predict_final_df.astype(float)
met_abun_predict_final_df = met_abun_predict_final_df.astype(float)
bac_abun_predict_final_df[:] = np.nan
met_abun_predict_final_df[:] = np.nan


################################
### Media (ModelSEED IDs)    ###
################################

defined_media_cpd = {
    "cpd00063": -1.290526021,    # Ca2+
    "cpd00971": -47.34272139,    # Na+
    "cpd00058": -0.021265311,    # Cu2+
    "cpd00048": -10.22721245,    # Sulfate
    "cpd00215": -0.009822218,    # Pyridoxal
    "cpd00184": -0.021,          # Thymidine
    "cpd00309": -0.024981921,    # Xanthine
    "cpd00393": -0.002402832,    # Folate
    "cpd00247": -0.064061499,    # Orotate
    "cpd00205": -6.617592674,    # K+
    "cpd00149": -0.055,          # Co2+
    "cpd00209": -0.109323669,    # Nitrate
    "cpd10516": -0.066,          # Fe3+
    "cpd10515": -0.066,          # Fe2+
    "cpd00254": -4.380495746,    # Mg2+
    "cpd00030": -0.33,           # Mn2+
    "cpd11574": -0.004856255,    # Molybdate
    "cpd03396": -0.000578243,    # Selenite
    "cpd15574": -0.003403444,    # Tungstate
    "cpd00099": -14.45186396,    # Cl-
    "cpd00244": -0.01543217,     # Ni2+
    "cpd00034": -0.061931009,    # Zn2+
    "cpd00644": -100,#-0.000419695,    # Pantothenate
    "cpd00635": -1.47561e-06,    # Cobalamin
    "cpd00263": -0.000972583,    # Pyridoxine
    "cpd00067": -0.001565646,    # H+
    "cpd00220": -1.000531406,    # Riboflavin
    "cpd00087": -2.80628e-06,    # Tetrahydrofolate
    "cpd00305": -100,#-0.000593064,    # Thiamin
    "cpd00443": -0.073072918,    # 4-Aminobenzoate
    "cpd00218": -0.021,          # Pyridoxamine
    "cpd00013": -9.347366847,    # NH3/NH4+
    "cpd00009": -6.61327063,     # Phosphate
    "cpd00133": -0.034392401,    # Nicotinamide
    "cpd00028": -0.015338835,    # Heme
    "cpd00307": -5.94059e-05,    # Cytosine
    "cpd00207": -5.95514e-05,    # Guanine
    "cpd00128": -5.99423e-05,    # Adenine
    "cpd00092": -5.88829e-05,    # Uracil
    "cpd00121": -6.272202487,    # myo-Inositol
    "cpd00104": -0.040952069,    # Biotin
    "cpd00035": -5.3,            # L-Alanine
    "cpd00051": -21.81400689,    # L-Arginine
    "cpd00132": -2.6,            # L-Asparagine
    "cpd00041": -0.4,            # L-Aspartate
    "cpd00084": -8.4,            # L-Cysteine
    "cpd00023": -0.662721893,    # L-Glutamate
    "cpd00053": -2.7,            # L-Glutamine
    "cpd00119": -1,              # L-Histidine
    "cpd00322": -1.6,            # L-Isoleucine
    "cpd00107": -3.6,            # L-Leucine
    "cpd00039": -2.4,            # L-Lysine
    "cpd00060": -0.84,           # L-Methionine
    "cpd00066": -4.5,            # L-Phenylalanine
    "cpd00129": -5.9,            # L-Proline
    "cpd00054": -6.4,            # L-Serine
    "cpd00161": -1.9,            # L-Threonine
    "cpd00065": -0.73,           # L-Tryptophan
    "cpd00156": -2.8,            # L-Valine
    "cpd00069": -3.201059661,    # L-Tyrosine
    "cpd11575": -71.68003181,    # MOPS
    "cpd00242": -47.6150797,     # Bicarbonate
    "cpd00224": -21.31486045,    # L-Arabinose
    "cpd00027": -24.97835209,    # D-Glucose
    "cpd00159": -28.30817052,    # L-Lactate
    "cpd00179": -4.382120947,    # Maltose
    "cpd00001": -55.50645091,    # H2O
    "cpd00011": -100.0            # CO2
}

defined_media = {EX_FMT.format(cpd=c): v for c, v in defined_media_cpd.items()}

defined_media_df = pd.DataFrame.from_dict(defined_media, orient='index')
defined_media_df = defined_media_df.reset_index()
defined_media_df.columns = ['reaction', 'fluxValue']
defined_media_df['fluxValue'] = -1.0 * defined_media_df['fluxValue']


################################
### Load KBase models        ###
################################

cobra_models = sorted(cobra_models_dir.glob('*.sbml'))
cobra_models = {f.stem : f for f in cobra_models}

### Just loading the models needed in 

loaded_models = {}

#count = 0

for key in cobra_models:
    #if count == 1:
    #    break
    #print(key.split('_'))
    model_name = key.split('_')[0] + '_' + key.split('_')[1]
    #model_name = key.split('.')[0]
    #model_name = model_name[3:]
    #print(model_name)
    model = cobra.io.read_sbml_model(cobra_models[key])
    loaded_models[model_name] = model
    #count+=1


adjusted_names = []

for key in namedict:
    temp = namedict[key].split('_')
    temp_name = temp[0] + '_' + temp[1]
    adjusted_names.append(temp_name)

adjusted_names_dict = dict(zip(namedict.keys(), adjusted_names))
reverse_order_name_dict = dict(zip(adjusted_names, namedict.keys()))

correct_model_dict_order = {}
correct_model_name_order = []


for i in allspecies:
    model_to_grab = adjusted_names_dict[i]
    #model_to_grab_genus = model_to_grab.split('_')[0]
    print(model_to_grab)
    correct_model_dict_order[model_to_grab] = loaded_models[model_to_grab]
    correct_model_name_order.append(model_to_grab)

print(correct_model_name_order)


### Experiments that can run (every present species has a model)
#exp_names = m2_init_abun.index.to_list()
#n_total = len(m2_init_abun)
#runnable_sims, skipped_sims = [], {}
#for i in range(n_total):
    #init = np.array(m2_init_abun.iloc[i, :].to_list())
    #present = [m2_init_abun.columns[k] for k in np.where(init != 0)[0]]
    #xlacking = [s for s in present if s not in kbase_models]

#    runnable_sims.append(i)
#print(f'{len(runnable_sims)}/{n_total} experiments runnable')

exp_names = m2_init_abun.index.to_list()
n_total = len(m2_init_abun)
runnable_sims, skipped_sims = [], {}
for i in range(n_total):
    row = pd.to_numeric(m2_init_abun.iloc[i, :], errors='coerce').fillna(0)
    present = row.index[row != 0].tolist()
    if MONOCULTURES_ONLY and len(present) != 1:
        continue
    runnable_sims.append(i)
print(f'{len(runnable_sims)}/{n_total} experiments runnable')






param_save = {'Sim_num': test_num, 'Scaling_factor': scal_fac, 'Total_time_steps': t_steps,
              'notes': notes, 'flux_sampling': FLUX_SAMPLING, 'ex_fmt': EX_FMT,
              'models_dir': cobra_models_dir_path}
with open(plot_dir_path_base + '/params.txt', 'w') as file:
    file.write(json.dumps(param_save))


def create_plots(FBA_biomass_df, glv_biomass_df, met_pool_over_time,
                 m2_final_abun_bac_filt, m2_final_abun_met_filt,
                 bac_to_keep_for_inf, scal_fac, t_steps,
                 plot_dir_path_spec_exp, spec_exp_name, reverse_order_name_dict):
    """Create all plots for a simulation"""

    FBA_biomass_df_plot = FBA_biomass_df.copy()
    FBA_biomass_df_plot = FBA_biomass_df_plot.melt(ignore_index=False)
    FBA_biomass_df_plot = FBA_biomass_df_plot.reset_index()
    FBA_biomass_df_plot.columns = ['FeatureID', 'time', 'count']
    FBA_biomass_df_plot['count'] = FBA_biomass_df_plot['count'] * scal_fac

    glv_biomass_df_plot = glv_biomass_df.copy()
    glv_biomass_df_plot = glv_biomass_df_plot.melt(ignore_index=False)
    glv_biomass_df_plot = glv_biomass_df_plot.reset_index()
    glv_biomass_df_plot.columns = ['FeatureID', 'time', 'count']
    glv_biomass_df_plot['count'] = glv_biomass_df_plot['count'] * scal_fac

    # Plot 1: Combined biomass trajectories
    fig, axs = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=FBA_biomass_df_plot, x='time', y='count', hue='FeatureID')
    plt.savefig(plot_dir_path_spec_exp + '/biomass_single_plot_' + spec_exp_name + '.pdf', bbox_inches="tight")
    plt.close()

    # Plot 2: Individual species plots
    if multi_t_pt_dataset == 'yes':
        temp_filtered_final_abun = m2_final_abun_bac_filt.drop(columns=['Time'])
        temp_filtered_final_abun = temp_filtered_final_abun.T.iloc[bac_to_keep_for_inf, :].T
        temp_filtered_final_abun['Time'] = m2_final_abun_bac_filt['Time'].to_list()
        temp_filtered_final_abun_melt = temp_filtered_final_abun.melt(id_vars=['Time'])
        temp_filtered_final_abun_melt.columns = ['Time', 'Species', 'Abun']
    else:
        m2_final_abun_bac_filt_list = list(m2_final_abun_bac_filt[bac_to_keep_for_inf])
    num_plots = len(FBA_biomass_df.index)
    cols = 5
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=False, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, key in enumerate(FBA_biomass_df.index):
        temp_glv = glv_biomass_df_plot[glv_biomass_df_plot['FeatureID'] == key]
        temp_FBA = FBA_biomass_df_plot[FBA_biomass_df_plot['FeatureID'] == key]

        sns.lineplot(ax=axes[i], data=temp_glv, x='time', y='count', color='blue', lw=3)
        sns.lineplot(ax=axes[i], data=temp_FBA, x='time', y='count', color='red', lw=3)

        if multi_t_pt_dataset == 'yes':
            species_name = reverse_order_name_dict[key]
            temp_for_plot = temp_filtered_final_abun_melt[temp_filtered_final_abun_melt['Species'] == species_name]
            sns.scatterplot(ax=axes[i], data=temp_for_plot, x='Time', y='Abun', color='green', s=100)
        else:
            sns.scatterplot(ax=axes[i], x=[t_steps], y=[m2_final_abun_bac_filt_list[i]], color='green', s=100)

        axes[i].set_title(f"{key}")
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Conc')

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.savefig(plot_dir_path_spec_exp + '/biomass_exp_vs_sim_single_species_' + spec_exp_name + '.pdf', bbox_inches="tight")
    plt.close()

    # Plot 3: All metabolite concentrations (exchange IDs labeled with compound names)
    met_pool_over_time_df = pd.DataFrame(met_pool_over_time).fillna(0)
    met_pool_named = met_pool_over_time_df.rename(columns=lambda c: f"{EX_NAMES.get(c, c)} ({c})")
    met_pool_named_melt = met_pool_named.melt(ignore_index=False).reset_index()
    met_pool_named_melt.columns = ['Time', 'Metabolite', 'Concentration']

    fig, axs = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=met_pool_named_melt, x='Time', y='Concentration', hue='Metabolite')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, ncol=2)
    plt.savefig(plot_dir_path_spec_exp + '/met_concentration' + spec_exp_name + '.pdf', bbox_inches="tight")
    plt.close()

    # Plot 4: SCFAs, simulated vs experimental
    scfa = scfa_table(met_pool_over_time)
    scfa_melt = scfa.melt(ignore_index=False).reset_index()
    scfa_melt.columns = ['Time', 'Metabolite', 'Concentration']

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=scfa_melt, x='Time', y='Concentration', hue='Metabolite',
                 palette=SCFA_PALETTE, ax=ax)

    for name, color in SCFA_PALETTE.items():
        if multi_t_pt_dataset == 'yes':
            if name in m2_final_abun_met_filt.columns:
                exp_data = m2_final_abun_met_filt[['Time', name]].dropna()
                if len(exp_data) > 0:
                    sns.scatterplot(x=exp_data['Time'].values, y=exp_data[name].values, color=color,
                                    s=100, edgecolor='black', label=f"{name} (exp)", ax=ax)
        else:
            y_val = m2_final_abun_met_filt.get(name, None)
            if y_val is not None and not pd.isna(y_val):
                sns.scatterplot(x=[t_steps], y=[y_val], color=color, s=100,
                                edgecolor='black', label=f"{name} (exp)", ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Metabolite", fontsize=12)
    ax.set_title("Simulated vs Experimental Metabolite Concentrations", fontsize=14)
    plt.savefig(plot_dir_path_spec_exp + '/mets_of_int_exp_vs_sim' + spec_exp_name + '.pdf', bbox_inches="tight")
    plt.close()

    # Plot 5: Species relative abundance over time (multi-timepoint data), exp vs gLV vs gLV-FBA
    if multi_t_pt_dataset == 'yes':

        temp_wide = temp_filtered_final_abun_melt.pivot(index='Species', columns='Time', values='Abun')
        temp_wide_RA = temp_wide / temp_wide.sum(axis=0)

        glv_biomass_df_RA = glv_biomass_df / glv_biomass_df.sum(axis=0)
        FBA_biomass_df_RA = FBA_biomass_df / FBA_biomass_df.sum(axis=0)

        temp_wide_RA_plot = temp_wide_RA.melt(ignore_index=False).reset_index()
        temp_wide_RA_plot.columns = ['FeatureID', 'time', 'count']
        temp_wide_RA_plot['count'] = temp_wide_RA_plot['count'] * scal_fac

        FBA_biomass_df_RA_plot = FBA_biomass_df_RA.melt(ignore_index=False).reset_index()
        FBA_biomass_df_RA_plot.columns = ['FeatureID', 'time', 'count']
        FBA_biomass_df_RA_plot['count'] = FBA_biomass_df_RA_plot['count'] * scal_fac

        glv_biomass_df_RA_plot = glv_biomass_df_RA.melt(ignore_index=False).reset_index()
        glv_biomass_df_RA_plot.columns = ['FeatureID', 'time', 'count']
        glv_biomass_df_RA_plot['count'] = glv_biomass_df_RA_plot['count'] * scal_fac

        glv_biomass_df_RA_plot['time'] = pd.to_numeric(glv_biomass_df_RA_plot['time'])
        temp_wide_RA_plot['time'] = pd.to_numeric(temp_wide_RA_plot['time'])
        FBA_biomass_df_RA_plot['time'] = pd.to_numeric(FBA_biomass_df_RA_plot['time'])

        feature_order = glv_biomass_df_RA_plot['FeatureID'].value_counts().index.tolist()
        reversed_order = feature_order[::-1]

        twentysiz = [
            "#690f19", "#b80000", "#d63220", "#d64e20", "#c1693a",
            "#d38838", "#AE7219", "#C09038", "#d1ad57", "#d8c33a", "#f1e149",
            "#d2e626", "#a0c618", "#7dc119", "#40a903", "#0c9515",
            "#1da05a", "#1da18d", "#1d91a1", "#1d6ea1", "#1d31a1",
            "#1a1a87", "#5b5bc9", "#8d8df1", "#8670cc", "#9870cc",
            "#000000"
        ]
        sns.set_palette(twentysiz)
        color_map = dict(zip(feature_order, twentysiz))

        continuous_pivot_MDSINE = (
            glv_biomass_df_RA_plot.pivot(index='time', columns='FeatureID', values='count').fillna(0)
        )[reversed_order]
        continuous_pivot_FBA = (
            FBA_biomass_df_RA_plot.pivot(index='time', columns='FeatureID', values='count').fillna(0)
        )[reversed_order]

        temp_wide_RA_plot['FeatureID'] = [adjusted_names_dict[f] for f in temp_wide_RA_plot['FeatureID']]

        fig, (ax_top, ax_middle, ax_bottom) = plt.subplots(
            3, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
        temp_wide_RA_plot['FeatureID'] = pd.Categorical(temp_wide_RA_plot['FeatureID'], categories=feature_order, ordered=True)

        sns.histplot(data=temp_wide_RA_plot, x='time', weights='count', hue='FeatureID',
                     multiple='stack', ax=ax_top, alpha=0.8, binwidth=4, palette=color_map)
        ax_top.legend_.remove()
        ax_top.set_ylabel('Relative Abundance')
        ax_top.set_title('Experimental Abundance')

        ax_middle.stackplot(continuous_pivot_MDSINE.index,
                            *[continuous_pivot_MDSINE[col] for col in continuous_pivot_MDSINE.columns],
                            alpha=0.8, colors=[color_map[feat] for feat in reversed_order], labels=reversed_order)
        ax_middle.set_ylabel('Relative Abundance')
        ax_middle.set_title('Trajectories from MDSINE')

        ax_bottom.stackplot(continuous_pivot_FBA.index,
                            *[continuous_pivot_FBA[col] for col in continuous_pivot_FBA.columns],
                            alpha=0.8, colors=[color_map[feat] for feat in reversed_order], labels=reversed_order)
        ax_bottom.set_ylabel('Relative Abundance')
        ax_bottom.set_xlabel('Time')
        ax_bottom.set_title('Trajectories from FBA')

        legend_elements = [Patch(facecolor=color_map[feat], label=feat) for feat in feature_order]
        fig.legend(handles=legend_elements, title='Species', loc='center left',
                   bbox_to_anchor=(.8, 0.5), borderaxespad=0, frameon=False, ncol=1)
        plt.subplots_adjust(hspace=0.1, right=1)
        plt.suptitle('Relative Species Abun \n' + str(spec_exp_name), x=0.6, fontsize=20)
        plt.subplots_adjust(right=0.75)
        plt.savefig(plot_dir_path_spec_exp + '/species_RA_' + spec_exp_name + '.pdf', bbox_inches="tight")
        plt.close()


def run_simulations(i, draw=0):
    """Run a single simulation (experiment i, draw `draw`). Outputs go to <exp>/draw_<draw>/."""

    models_list = [copy.deepcopy(model) for model in list(correct_model_dict_order.values())]

    sim_to_grab = i
    spec_exp_name = exp_names[sim_to_grab]

    plot_dir_path_spec_exp = os.path.join(plot_dir_path_base, spec_exp_name, f'draw_{draw}')
    os.makedirs(plot_dir_path_spec_exp, exist_ok=True)

    ### MDSINE2 scaling
    init_abun = np.array(m2_init_abun.iloc[sim_to_grab, :].to_list()) * 10e8
    sol = odeint(gLV, init_abun, time, args=args, tfirst=True)
    sol = sol / 10e8
    init_abun = init_abun / 10e8

    if multi_t_pt_dataset == 'yes':
        m2_final_abun_bac_filt = m2_final_abun_bac.loc[spec_exp_name, :]
        m2_final_abun_met_filt = m2_final_abun_met.loc[spec_exp_name, :]
        sol = np.array(Venturelli_long_MDSINE2_interpolated_data[spec_exp_name])
    else:
        m2_final_abun_bac_filt = m2_final_abun_bac.iloc[sim_to_grab, :]
        m2_final_abun_met_filt = m2_final_abun_met.iloc[sim_to_grab, :]
        if USE_INTERPOLATED_TRAJ:
            sol = np.array(interpolated_data[spec_exp_name])

    bac_to_keep_for_inf = list(np.where(init_abun != 0)[0])

    glv_abun_df = pd.DataFrame(sol)

    rate_array = np.zeros((25, glv_abun_df.shape[0] - 1))
    for j in range(0, len(glv_abun_df.T.columns) - 1):
        rate_array[:, j] = (glv_abun_df.T.iloc[:, j + 1] / glv_abun_df.T.iloc[:, j]) - 1
    rate_df = pd.DataFrame(rate_array).fillna(0)

    glv_abun_df = glv_abun_df.iloc[:, bac_to_keep_for_inf]
    rate_df = rate_df.iloc[bac_to_keep_for_inf, :]
    init_abun_scaled = list((init_abun / scal_fac)[bac_to_keep_for_inf])
    correct_model_name_order_current_sim = [correct_model_name_order[k] for k in bac_to_keep_for_inf]
    models_list_current_sim = list(np.array(models_list)[bac_to_keep_for_inf])
    glv_abun_df = glv_abun_df / scal_fac

     # Set media conditions for each model
    for j in range(0, len(models_list_current_sim)):
        test_media = make_media(models_list_current_sim[j], defined_media_df)
        models_list_current_sim[j].medium = test_media
        print(models_list_current_sim[j].slim_optimize())
        print(models_list_current_sim[j].medium)
        print(correct_model_name_order_current_sim[j])
    
    # Verify model order
    filt_columns_list_names = m2_init_abun.iloc[:,bac_to_keep_for_inf].columns.to_list()
    checking_order_list = [adjusted_names_dict[name] for name in filt_columns_list_names]
    
    if checking_order_list == correct_model_name_order_current_sim:
        print('yes, the order and identity of GEMs selected for inference is correct')
    else:
        print('no good')
        print('This is correct order of models: ',correct_model_name_order_current_sim)
        print('This is the order given: ', checking_order_list)
        raise ValueError(f"Model order mismatch for simulation {i}")
    

    met_pool_over_time, model_abun_dict, mets_used_for_constraint = static_dfba(
        list_model_names=correct_model_name_order_current_sim,
        list_models=models_list_current_sim,
        initial_abundance=init_abun_scaled,
        total_sim_time=t_steps,
        num_t_steps=t_steps,
        glv_out=np.array(glv_abun_df),
        glv_params=None,
        environ_cond=defined_media_df,
        pfba=not FLUX_SAMPLING,
        MDSINE_rates=rate_df,
        Diet=pd.DataFrame(),
        output_file_path=plot_dir_path_spec_exp,
        flux_sampling=FLUX_SAMPLING,
        host=None,
        random_constraints='No',
        AGORA_models=AGORA_MODELS_FLAG,
        calc_neg_consumption='no',
        stable_t_points=[],
        dfba_traditional = DFBA_TRADITIONAL
    )

    if SAVE_NPY:
        np.save(plot_dir_path_spec_exp + '/met_pool.npy', met_pool_over_time)
        np.save(plot_dir_path_spec_exp + '/abun_save.npy', model_abun_dict)

    FBA_biomass = np.zeros([len(model_abun_dict.keys()), t_steps + 1])
    glv_biomass = np.zeros([len(model_abun_dict.keys()), t_steps + 1])
    for count, key in enumerate(model_abun_dict):
        FBA_biomass[count, :] = model_abun_dict[key]['fba_biomass']
        glv_biomass[count, :] = model_abun_dict[key]['glv_out'][:t_steps + 1]

    FBA_biomass_df = pd.DataFrame(FBA_biomass, index=list(model_abun_dict.keys()))
    glv_biomass_df = pd.DataFrame(glv_biomass, index=list(model_abun_dict.keys()))

    if MAKE_PLOTS:
        create_plots(FBA_biomass_df, glv_biomass_df, met_pool_over_time,
                     m2_final_abun_bac_filt, m2_final_abun_met_filt,
                     bac_to_keep_for_inf, scal_fac, t_steps,
                     plot_dir_path_spec_exp, spec_exp_name, reverse_order_name_dict)

    FBA_biomass_df_plot = FBA_biomass_df * scal_fac
    scfa = scfa_table(met_pool_over_time)
    scfa.to_csv(plot_dir_path_spec_exp + '/scfa_over_time.csv')

    if multi_t_pt_dataset == 'yes':
        timepoints_to_keep = [16, 32, 48]
        timepoints_to_keep_bac_abun = [0, 16, 32, 48]
        met_final = scfa.loc[timepoints_to_keep].copy()

        bac_subset = FBA_biomass_df_plot.iloc[:, timepoints_to_keep_bac_abun].T
        bac_final_predictions = pd.DataFrame(0.0, index=bac_subset.index,
                                             columns=bac_abun_predict_final_df.columns)
        inferred_species_cols = bac_abun_predict_final_df.columns[bac_to_keep_for_inf]
        bac_final_predictions.loc[:, inferred_species_cols] = bac_subset.values
    else:
        bac_final_predictions = np.zeros(25)
        bac_final_predictions[bac_to_keep_for_inf] = FBA_biomass_df_plot.iloc[:, -1].values
        met_final = scfa.loc[scfa.index == t_steps].iloc[-1]   # Series: Butyrate/Acetate/Lactate/Succinate

    return {
        'sim_idx': i,
        'draw': draw,
        'bac_predictions': bac_final_predictions,
        'met_predictions': met_final,
        'spec_exp_name': spec_exp_name
    }


def result_path_for(i, draw):
    return os.path.join(plot_dir_path_base, exp_names[i], f'sim_{draw}_result.pkl')


def fill_result(bac_df, met_df, result):
    """Write one experiment's result into the output frames."""
    sim_idx = result['sim_idx']
    exp_name = result['spec_exp_name']
    bac_pred = result['bac_predictions']
    met_pred = result['met_predictions']

    if multi_t_pt_dataset == 'yes':
        bac_rows = np.where(bac_df.index == exp_name)[0]
        assert len(bac_rows) == len(bac_pred), f"Bacteria row count mismatch for {exp_name}"
        for pos, (_, pred_values) in zip(bac_rows, bac_pred.iterrows()):
            bac_df.iloc[pos, :] = pred_values.values

        met_rows = np.where(met_df.index == exp_name)[0]
        assert len(met_rows) == len(met_pred), f"Metabolite row count mismatch for {exp_name}"
        for pos, (_, pred_values) in zip(met_rows, met_pred.iterrows()):
            for name in SCFA_EXCHANGES:
                met_df.iloc[pos, met_df.columns.get_loc(name)] = pred_values[name]
    else:
        bac_df.iloc[sim_idx, :] = bac_pred
        for name in SCFA_EXCHANGES:
            met_df.loc[exp_name, name] = met_pred[name]


def aggregate_results(n_draws):
    """Collect result pickles into per-draw CSVs plus final (mean over draws) CSVs."""
    bac_draws, met_draws = [], []
    for draw in range(n_draws):
        bac_df = bac_abun_predict_final_df.copy()
        met_df = met_abun_predict_final_df.copy()
        found, missing = 0, []
        for i in runnable_sims:
            path = result_path_for(i, draw)
            if not os.path.exists(path):
                missing.append(exp_names[i])
                continue
            with open(path, 'rb') as f:
                fill_result(bac_df, met_df, pickle.load(f))
            found += 1
        print(f'draw {draw}: {found} results, {len(missing)} missing')
        if missing:
            print('  missing:', missing[:20], '...' if len(missing) > 20 else '')
        if n_draws > 1:
            bac_df.to_csv(os.path.join(plot_dir_path_base, f'final_abun_predict_df_sim_{draw}.csv'))
            met_df.to_csv(os.path.join(plot_dir_path_base, f'final_met_predict_df_sim_{draw}.csv'))
        bac_draws.append(bac_df)
        met_draws.append(met_df)

    def mean_over(draws, template):
        stacked = np.stack([d.values.astype(float) for d in draws])
        return pd.DataFrame(np.nanmean(stacked, axis=0), index=template.index, columns=template.columns)

    bac_final = mean_over(bac_draws, bac_abun_predict_final_df)
    met_final = mean_over(met_draws, met_abun_predict_final_df)
    abun_path = os.path.join(plot_dir_path_base, 'final_abun_predict_df.csv')
    met_path = os.path.join(plot_dir_path_base, 'final_met_predict_df.csv')
    bac_final.to_csv(abun_path)
    met_final.to_csv(met_path)
    print(f'Saved:\n  {abun_path}\n  {met_path}' + ('  (mean over draws)' if n_draws > 1 else ''))
    ### final sim vs exp figures -- only made here, not in worker jobs
    make_sim_vs_exp_plots(bac_final, met_final, met_draws=met_draws)

def _corr_stats(exp_vals, sim_vals):
    """Pearson + Spearman on finite pairs. NaN-safe, returns dict + cleaned arrays."""
    x = np.asarray(exp_vals, dtype=float)
    y = np.asarray(sim_vals, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {'n': len(x),
           'n_sim_zero': int(np.sum(y == 0)),
           'n_exp_zero': int(np.sum(x == 0))}
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        out.update(pearson_r=np.nan, pearson_p=np.nan, spearman_r=np.nan, spearman_p=np.nan)
        return out, x, y
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    out.update(pearson_r=pr[0], pearson_p=pr[1], spearman_r=sr[0], spearman_p=sr[1])
    return out, x, y


def _identity_line(ax, x, y):
    hi = max(np.max(x) if len(x) else 0, np.max(y) if len(y) else 0)
    ax.plot([0, hi], [0, hi], color='black', lw=1, zorder=0)


def make_sim_vs_exp_plots(bac_final, met_final, met_draws=None):
    """
    Final simulated-vs-experimental figures. Called once from aggregate_results,
    so worker jobs never produce these.

    bac_final / met_final are copies of the experimental templates, so their
    index and columns already align with m2_final_abun_bac / m2_final_abun_met.
    Experiments that were never run are all-NaN and get dropped.
    """
    ### ---------------- metabolites ----------------
    met_sim = met_final.dropna(how='all')
    met_exp = m2_final_abun_met.loc[met_sim.index, met_sim.columns]
        ### long form: one row per (experiment, metabolite, draw), for seaborn's CI
    met_long = None
    if met_draws is not None and len(met_draws) > 1:
        frames = []
        for k, d in enumerate(met_draws):
            lf = d.loc[met_sim.index].melt(ignore_index=False).reset_index()
            lf.columns = ['Experiments', 'Metabolite', 'sim']
            lf['draw'] = k
            frames.append(lf)
        met_long = pd.concat(frames, ignore_index=True).dropna(subset=['sim'])
        ex = met_exp.melt(ignore_index=False).reset_index()
        ex.columns = ['Experiments', 'Metabolite', 'exp']
        met_long = met_long.merge(ex, on=['Experiments', 'Metabolite'])

    n_met = len(met_sim.columns)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axes = axes.flatten()
    stat_rows = []

    for i, name in enumerate(met_sim.columns):
        ax = axes[i]
        stats, x, y = _corr_stats(met_exp[name], met_sim[name])
        stats['variable'] = name
        stat_rows.append(stats)

        #sns.scatterplot(ax=ax, x=x, y=y, s=28, alpha=0.75, edgecolor='none')
        #if stats['n'] >= 3 and np.std(x) > 0:
        #    sns.regplot(ax=ax, x=x, y=y, scatter=False, line_kws={'lw': 1.5})
        #_identity_line(ax, x, y)
                ### points + bootstrapped 95% CI straight from the per-draw table
        drew_ci = False
        if met_long is not None:
            sub = met_long[met_long['Metabolite'] == name]
            if len(sub):
                sns.lineplot(data=sub, x='exp', y='sim', ax=ax,
                             errorbar=('ci', 95), err_style='bars',
                             marker='o', markersize=5, linestyle='none',
                             err_kws={'elinewidth': 0.8, 'capsize': 0},
                             estimator='mean', alpha=0.8, zorder=2)
                drew_ci = True
        if not drew_ci:
            sns.scatterplot(ax=ax, x=x, y=y, s=28, alpha=0.75, edgecolor='none')
        if stats['n'] >= 3 and np.std(x) > 0:
            sns.regplot(ax=ax, x=x, y=y, scatter=False, line_kws={'lw': 1.5})
        _identity_line(ax, x, y)

        ax.set_title(f"{name}\n"
                     f"Pearson {stats['pearson_r']:.3f} (p={stats['pearson_p']:.3g})   "
                     f"Spearman {stats['spearman_r']:.3f}\n"
                     f"n={stats['n']}, sim zeros={stats['n_sim_zero']}",
                     fontsize=10)
        ax.set_xlabel('Experimental Conc (mM)')
        ax.set_ylabel('Simulated Conc (mM)')

    for j in range(n_met, len(axes)):
        fig.delaxes(axes[j])

    subtitle = 'monocultures' if MONOCULTURES_ONLY else 'all experiments'
    fig.suptitle(f'Experimental vs Simulated Concentrations ({subtitle})')
    met_fig_path = os.path.join(plot_dir_path_base, 'final_sim_vs_exp_metabolites.pdf')
    fig.savefig(met_fig_path, bbox_inches='tight')
    plt.close(fig)

    ### ---------------- abundance ----------------
    abun_sim = bac_final.dropna(how='all')
    abun_exp = m2_final_abun_bac.loc[abun_sim.index, abun_sim.columns]

    sim_melt = abun_sim.melt(ignore_index=False).reset_index()
    sim_melt.columns = ['Experiments', 'Species', 'OD_sim']
    exp_melt = abun_exp.melt(ignore_index=False).reset_index()
    exp_melt.columns = ['Experiments', 'Species', 'OD_exp']
    merged = exp_melt.merge(sim_melt, on=['Experiments', 'Species'])

    ### species absent from an experiment carry no information about fit
    n_before = len(merged)
    merged = merged[merged['OD_exp'] > 0]
    n_dropped = n_before - len(merged)

    abun_stats, x, y = _corr_stats(merged['OD_exp'], merged['OD_sim'])
    abun_stats['variable'] = 'Abundance'
    stat_rows.append(abun_stats)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    if merged['Species'].nunique() <= 30:
        sns.scatterplot(data=merged, x='OD_exp', y='OD_sim', hue='Species', ax=ax,
                        s=30, alpha=0.8, edgecolor='none')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=2)
    else:
        sns.scatterplot(data=merged, x='OD_exp', y='OD_sim', ax=ax, s=30, alpha=0.7,
                        edgecolor='none')
    if abun_stats['n'] >= 3:
        sns.regplot(data=merged, x='OD_exp', y='OD_sim', scatter=False, ax=ax,
                    line_kws={'lw': 1.5})
    _identity_line(ax, merged['OD_exp'].to_numpy(), merged['OD_sim'].to_numpy())

    ax.set_xlabel('Experimental OD')
    ax.set_ylabel('Simulated OD')
    ax.set_title(f"Experimental vs Simulated Abundance ({subtitle})\n"
                 f"Pearson {abun_stats['pearson_r']:.3f} (p={abun_stats['pearson_p']:.3g})   "
                 f"Spearman {abun_stats['spearman_r']:.3f}   n={abun_stats['n']}")
    abun_fig_path = os.path.join(plot_dir_path_base, 'final_sim_vs_exp_abundance.pdf')
    fig.savefig(abun_fig_path, bbox_inches='tight')
    plt.close(fig)

    print(f'Plots:\n  {met_fig_path}\n  {abun_fig_path}')
    print(f'  ({len(met_sim)} experiments plotted; '
          f'{n_dropped} abundance rows dropped where OD_exp == 0)')
    print(pd.DataFrame(stat_rows).set_index('variable').to_string())

if __name__ == "__main__":
    import textwrap
    import time as time_library

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim-idx", type=int, default=None,
        help="0-based experiment index. If given, runs that one simulation directly (worker mode). "
             "If omitted, submits one LSF job per experiment (submission mode).",
    )
    parser.add_argument(
        "--sim-num", type=int, default=1,
        help="Worker mode: draw index. Submission/aggregate mode: number of draws.",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Collect finished result pickles into the final CSVs (no jobs submitted).",
    )
    cli_args = parser.parse_args()

    num_sims = cli_args.sim_num

    if cli_args.aggregate:
        # ---- Aggregate mode ----
        aggregate_results(num_sims)

    elif cli_args.sim_idx is not None:
        # ---- Worker mode: run a single simulation ----
        sim_idx = cli_args.sim_idx
        if not (0 <= sim_idx < n_total):
            raise IndexError(f"sim_idx {sim_idx} out of range [0, {n_total})")
        #if sim_idx not in runnable_sims:
        #    raise ValueError(f"{exp_names[sim_idx]} needs models not available: {skipped_sims[exp_names[sim_idx]]}")

        spec_exp_name = exp_names[sim_idx]
        result_path = result_path_for(sim_idx, num_sims)

        print(f"Running experiment {spec_exp_name} (idx {sim_idx}), (sim batch {num_sims})")
        results = run_simulations(sim_idx, draw=num_sims)

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[{spec_exp_name}] saved -> {result_path}")

    else:
        # ---- Submission mode: one LSF job per runnable experiment per draw ----
        this_script_path = os.path.abspath(__file__)
        job_log_dir = os.path.join(plot_dir_path_base, "job_output", "out_logs")
        error_log_dir = os.path.join(plot_dir_path_base, "job_output", "error_logs")
        lsf_script_dir = os.path.join(plot_dir_path_base, "lsf_scripts")
        os.makedirs(job_log_dir, exist_ok=True)
        os.makedirs(error_log_dir, exist_ok=True)
        os.makedirs(lsf_script_dir, exist_ok=True)

        #if skipped_sims:
            #print(f"Not submitting {len(skipped_sims)} experiments with missing models (see params.txt)")

        n_submitted, n_existing = 0, 0
        for j in range(0, num_sims):
            for i in runnable_sims:
                spec_exp_name = exp_names[i]
                if os.path.exists(result_path_for(i, j)):
                    n_existing += 1
                    continue

                unique_job_name = f"{spec_exp_name}_test_{test_num}_sim_{j}"
                batch_script = os.path.join(lsf_script_dir, f"{unique_job_name}.lsf")

                batch_content = textwrap.dedent(f"""\
    #!/bin/bash
    #BSUB -J {unique_job_name}
    #BSUB -o {job_log_dir}/{unique_job_name}.%J.out
    #BSUB -e {error_log_dir}/{unique_job_name}.%J.err
    #BSUB -q long
    #BSUB -W 72:00
    #BSUB -n 1
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=4GB]"

    cd {os.path.dirname(this_script_path)}
    python {this_script_path} --sim-idx {i} --sim-num {j}
    """)

                with open(batch_script, 'w') as f:
                    f.write(batch_content)

                try:
                    with open(batch_script) as f:
                        subprocess.run(["bsub"], stdin=f, check=True)
                    print(f"Submitted draw {j} for sim ({spec_exp_name})")
                    n_submitted += 1
                except subprocess.CalledProcessError as e:
                    print(f"Failed to submit job for sim {spec_exp_name}: {e}")
                    continue

                if n_submitted % 1000 == 0 and n_submitted != 0:
                    time_library.sleep(300)   # cool down for 5 minutes

        print(f"\nSubmitted {n_submitted} jobs (skipped {n_total*num_sims - n_submitted} already-complete).")

        print(f"When jobs finish: python {this_script_path} --aggregate --sim-num {num_sims}")
