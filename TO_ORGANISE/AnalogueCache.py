# var = 'SST'
var = 'SAT'

# For SST the regions are these:
regions = ['north_atlantic', 'subpolar_gyre', 'intergyre', 'tropical_atlantic_north',
           'tropical_atlantic_south', 'global', 'global60', 'nh', 'spg_rahmstorf', 'spg_menary18']

# For SAT:
# regions = ['europe1']

import numpy as np
import glob
import pickle
import os
from scipy import stats
import time
import sys
import mfilter
import random

# ==============
testing = False
# ==============

# ==============
# Constants we're reading in (NONE!)
# ==============
# target_region = sys.argv[1]
# window = np.long(sys.argv[2])
# region_set = sys.argv[3]
# experiment_set = sys.argv[4]
# analogue_skill_score = sys.argv[5]

# ==============
# Constants currently fixed
# ==============
# basedir = '/home/mmenary/python/notebooks/cmip6_2/output/'

## THIS WILL NEED CHANGING TO SOMETHING LIKE
## /work/scratch-nopw/lfbor/CMIP_SAT
user = "lfbor"
datadir = '/work/scratch-nopw/{:s}/CMIP_{:s}/'.format(user, var)

# processed_output_dir = '/modfs/ipslfs/dods/mmenary/AnalogueCache/'
# max_lead = 20
# forecast_metric = 'MaxCorr'

# if region_set == 'north_atlantic':
#     regions_processed = ['north_atlantic']
# elif region_set == 'subpolar_gyre':
#     regions_processed = ['subpolar_gyre']
# elif region_set == 'labsea':
#     regions_processed = ['labsea']
# elif region_set == 'tropical_atlantic_north':
#     regions_processed = ['tropical_atlantic_north']
# elif region_set == 'tropical_atlantic_south':
#     regions_processed = ['tropical_atlantic_south']
# else:
#     raise ValueError("Not programmed other sets yet")

# if experiment_set == 'piControl':
#     experiments_processed = ['piControl']
# elif experiment_set == 'NotHistorical':
#     experiments_processed = ['hist-aer', 'hist-stratO3', 'rcp85', 'hist-GHG',
#                              'ssp126', 'rcp45', 'ssp585', 'piControl', 'hist-nat']
# elif experiment_set == 'Everything':
#     experiments_processed = ['hist-aer', 'hist-stratO3', 'rcp85', 'hist-GHG',
#                              'ssp126', 'rcp45', 'ssp585', 'piControl', 'hist-nat',
#                              'historical']
# else:
#     raise ValueError("Not programmed other sets yet")

if analogue_skill_score  == 'Corr':
    smoothing = False
elif analogue_skill_score[:12] == 'CorrSmoothed':
    smoothing = True
    smo_len = np.long(analogue_skill_score[12:])
else:
    raise ValueError("Not programmed other analogue skill-scores yet")

if testing:
    testing_string = '_TEST'
    print "\n+|+ TESTING +|+\n"
else:
    testing_string = ''

# ==============
# Make a class to store the data. This might save me time in the future...
# ==============
class CMIPTimeSeries:
    def __init__(self, project, var, sub_var, model, experiment, time, time_series, ens_mem=1):
        # Project: CMIP5/6
        # Var: e.g. SST
        # Sub_Var: e.g. a particular region, like North Atlantic
        # Model: model
        # Experiment: experiment
        # Time: Model year or something
        # Time_series: The actual data (1D)
        self.project = project
        self.var = var
        self.sub_var = sub_var
        self.model = model
        self.experiment = experiment
        self.time = time
        self.time_series = time_series
        self.ens_mem = ens_mem

# ==============
# Make a class to store the output data. This might save me time in the future...
# ==============
class OutputMetrics:
    def __init__(self):
        self.source_index = []          # The index into the stored_data list
        self.rmse = []                  # The RMSE between truth and model in the window
        self.rmse_of_anom = []          # The RMSE between truth anomaly and model anomaly in the window
        self.corrs = []                 # The maximum correlation (any window - noted below)
        self.grads = []                 # The grad at maximal corr (for constructing the forecast data)
        self.intercepts = []            # The intercept at maximal corr (for constructing the forecast data)
        self.info = []                  # Currently the model
#         self.source_subvar_index = [] # Could add the region
        self.source_time_index = []     # The temporal index of the maximal corr

# ==============
# Define some functions to use later
# ==============
def calc_rmse(arr1, arr2):
    return np.sqrt(np.sum((arr1 - arr2)**2.))

def skip_this_one(probability_of_skipping):
    return random.random() < probability_of_skipping

# This has all the raw data in
input_save_file = '/data/mmenary/python_saves/HistoricalAnalogues_Inputs_{:s}.pkl'.format(var)

# # This is where we will store the processed correlations
# experiments_processed_string = '-'.join(experiments_processed)
# regions_processed_string = '-'.join(regions_processed)
# processed_save_file = '{:s}HistoricalAnalogues_Processed_Target-{:s}_{:s}_{:s}_{:s}_Window{:d}.pkl'
# processed_save_file = processed_save_file.format(processed_output_dir, target_region,
#                                                  experiments_processed_string, regions_processed_string,
#                                                  analogue_skill_score, window)
# if testing:
#     processed_save_file += '.TEST.pkl'

# ==============
print "Save data files to read will be:"
print input_save_file
print processed_save_file
print " "
# ==============

# ==============
# Put the input data into a (useful?) structure
# ==============
print "Reading source (model) data"
if os.path.isfile(input_save_file):
    with open(input_save_file, 'rb') as handle:
        print "Loading save file: {:s}".format(input_save_file)
        stored_data = pickle.load(handle)
        print "Loading finished..."
else:
    files = glob.glob(datadir + 'CMIP?_{:s}_*_*_Annual.pkl'.format(var))
    # files = glob.glob(datadir + 'CMIP?_SST_*_*_Annual_TimeSeries.pkl')
    nfiles = len(files)

    stored_data = []
    for ifile, this_file in enumerate(files):
        base_file = os.path.basename(this_file)
        split_file = base_file.split('_')
        print '{:d}/{:d}: {:s}'.format(ifile+1, nfiles, base_file)
        project = split_file[0]
        var = split_file[1]
        model = split_file[2]
        experiment_and_ens = split_file[3].split('-')

        if len(experiment_and_ens) == 1:
            # piControl
            experiment = experiment_and_ens[0]
            ens_mem_part = 1
        if len(experiment_and_ens) == 2:
            # Historical, scenario, etc
            experiment = experiment_and_ens[0]
            ens_mem_part = experiment_and_ens[1]
        elif len(experiment_and_ens) == 3:
            # Must be hist-aer or another DAMIP run
            experiment = experiment_and_ens[0] + '-' + experiment_and_ens[1]
            ens_mem_part = experiment_and_ens[2]

        ens_mem = long(ens_mem_part)
        # if experiment[:7] == 'decadal':
        #     continue

        with open(this_file, 'rb') as handle:
            ## THIS WILL NEED UPDATING
            _, sst_ts_ann, _, _, _, year_ann = pickle.load(handle)
            # sst_ts_ann, year_ann = pickle.load(handle)
            for iregion, region in enumerate(regions):
                if region not in sst_ts_ann.keys():
                    continue
                new_time_series = CMIPTimeSeries(project, var, region, model, experiment, year_ann,
                                                 sst_ts_ann[region], ens_mem=ens_mem)
                stored_data.append(new_time_series)

    with open(input_save_file, 'wb') as handle:
        print "Writing save file: {:s}".format(input_save_file)
        pickle.dump(stored_data, handle,  protocol=pickle.HIGHEST_PROTOCOL)
        print "Writing finished..."
