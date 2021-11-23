#!/usr/bin/env python3
# The above is required for sbatch submission

# ==============
# This script constructs the (area-averaged) annual-mean forecasts based on the chosen analogue selection criteria.
# It no longer actually calculates the skill as there are still different ways that you could combine these forecasts
# that would result in different levels of skill. That part is done in some python notebooks that allow for quicker
# visualisation of the effects of different choices.
# NOTE: For the case of the maps (STEP3b), the skill _is_ calculated in these Python scripts, as the amount of data
# makes it inpractical to do that in notebooks.
# ==============

glob_only = False
residual = False
look_for_globbed_savefile = False

import numpy as np
import glob
import pickle
import os
from scipy import stats
import time
import mfilter
import random
from scipy import interpolate
import sys
import hashlib
import xarray as xr

# ==============
# Constants we're reading in
# ==============
analogue_var = sys.argv[1]              # The variable used when creating the analogues
forecast_var = sys.argv[2]              # The variable we are forecasting
target_region = sys.argv[3]             # The region where we measure the skill
num_mems_to_take = np.int(sys.argv[4])  # The number of ensemble members to use
window = np.int(sys.argv[5])            # The window over which the analogue goodness was computed
target_domain_string = sys.argv[6]      # The region we used to create the analogues
smoothing = np.int(sys.argv[7])         # Whether the analogue data was pre-smoothed (and by how much)
testing = sys.argv[8]                   # Testing mode
pass_number = np.int(sys.argv[9])       # When picking the analogues, multiple passes might be required
method = sys.argv[10]                   # How were the analogues created? Currently "Corr" or "RMSE" methods
subset = sys.argv[11]                   # Whether to only pick from a subset of experients (investigating forcing)
clim_string = sys.argv[12]              # The climatology period used in analogues and observations
concatenate_string = sys.argv[13]       # Whether to analogues where we previously concatenated the historical+scenario runs

picontrols_only = False
skip_local_hist = False
strong_forcing_only = False
if subset == 'picontrols_only':
    picontrols_only = True  # Only use piControl runs to look at skill that has no forcing component
elif subset == 'skip_local_hist':
    skip_local_hist = True  # Similar to above but remove forcing influence by not letting NEARBY historicals be used
    nearby_hist = 75  # Hist (or hist-aer etc) within this window will not be used
elif subset == 'strong_forcing_only':
    strong_forcing_only = True  # Similar to above but remove forcing influence by not letting NEARBY historicals be used
    earliest_hist = 1990  # The earliest year in the hist (or hist-aer etc) experiments to allow for "strong" forcing
elif subset == 'None':
    pass
else:
    raise ValueError("Unknown subset used")

clim_start = int(clim_string[:4])
clim_end = int(clim_string[5:])

if concatenate_string == 'True':
    concatenate_hist_with_fut = True
    concat_string = '_CONCAT'
else:
    concatenate_hist_with_fut = False
    concat_string = ''

myhost = os.uname()[1]
print("myhost = {:s}".format(myhost))

usr = os.environ["USER"]

if 'ciclad' in myhost:
    raise ValueError("deprecated")
    # Ciclad options
    # forecast_datadir = '/data/mmenary/python_saves/CMIP_{:s}/'.format(forecast_var)
    # analogue_datadir = '/data/mmenary/python_saves/CMIP_{:s}/'.format(analogue_var)
    # processed_output_dir = '/prodigfs/ipslfs/dods/mmenary/AnalogueCache'
    # scripts_dir = '/home/mmenary/python/scripts/'
    # hadisst_save_file = '/data/mmenary/python_saves/HadISST_time_series_regions.pkl'
    # en4_save_file = '/data/mmenary/python_saves/EN4_0-500m_time_series_regions.pkl'
    # hadcrut4_save_file = '/data/mmenary/python_saves/HadCRUT4_time_series_regions.pkl'
else:
    # Jasmin options
    forecast_datadir = '/work/scratch-nopw/{}/CMIP_{:s}/'.format(usr, forecast_var)
    analogue_datadir = '/work/scratch-nopw/{}/CMIP_{:s}/'.format(usr, analogue_var)
    processed_output_dir = '/work/scratch-nopw/{}/AnalogueCache'.format(usr)
    scripts_dir = os.environ['ANALOGUE_SCRIPTS_DIR']  #'/home/users/{}/python/scripts/'.format(usr)
    hadisst_save_file = '/home/users/{}/data/HadISST_time_series_regions.pkl'.format(usr)
    en4_save_file = '/home/users/{}/data/EN4_0-500m_time_series_regions.pkl'.format(usr)
    hadcrut4_save_file = '/home/users/{}/data/HadCRUT4_time_series_regions.pkl'.format(usr)

if (clim_start != 1960) or (clim_end != 1990):
    processed_output_dir += '_{:d}-{:d}'.format(clim_start, clim_end)
skill_output_dir = processed_output_dir

print("Numpy version", np.__version__)

def read_target_domain(in_string):
    out_list = []
    for ii in range(4):
        out_list.append(np.int(in_string[ii*3:(ii+1)*3]))
    return out_list

target_domain = read_target_domain(target_domain_string)

# ==============
# Constants currently fixed
# ==============
nlead = 11
lead_times = np.arange(nlead)
start_lead = [1, 2, 3, 4, 5, 6,  1,  2]  # For the multiannual skill
end_lead =   [5, 6, 7, 8, 9, 10, 10, 10]
# Backwards windows to find mean/sd of Forecast var: (not necessarily analogue creation window)
mean_windows = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 45]

if testing == 'True':
    testing_string = '_TEST'
    print("\n+|+ TESTING +|+\n")
    step = 5
else:
    testing_string = ''
    step = 1

if smoothing > 1:
    smoothing_string = '_Smo{:d}'.format(smoothing)
else:
    smoothing_string = ''

if pass_number > 1:
    pass_string = '_PASS{:d}'.format(pass_number)
else:
    pass_string = ''

if method == 'RMSE':
    rmse_string = '_RMSEmethod'
else:
    rmse_string = ''

# ==================
# Just to get the associated projects for later
# ==================
cmip5_list_file = os.path.join(scripts_dir, 'model_lists/cmip5_list.txt')
cmip6_list_file = os.path.join(scripts_dir, 'model_lists/cmip6_list.txt')

cmip5_models = []
with open(cmip5_list_file, 'r') as f:
    for line in f.readlines():
        cmip5_models.append(line.strip())

cmip6_models = []
with open(cmip6_list_file, 'r') as f:
    for line in f.readlines():
        cmip6_models.append(line.strip())

# ==================
# Note the final target save file
# ==================
skill_template_info = 'ANALOGUE{:s}_FORECAST{:s}_DOMAIN{:s}_TARGET{:s}_WINDOW{:d}_MEMS{:d}{:s}_SpatialSkill{:s}{:s}{:s}{:s}_info'
skill_base_info = skill_template_info.format(analogue_var, forecast_var, target_domain_string, target_region, window, num_mems_to_take,
                                   smoothing_string, pass_string, testing_string, rmse_string, concat_string)
skill_template_forecast = 'ANALOGUE{:s}_FORECAST{:s}_DOMAIN{:s}_TARGET{:s}_WINDOW{:d}_MEMS{:d}{:s}_SpatialSkill{:s}{:s}{:s}{:s}_forecast'
skill_base_forecast = skill_template_forecast.format(analogue_var, forecast_var, target_domain_string, target_region, window, num_mems_to_take,
                                   smoothing_string, pass_string, testing_string, rmse_string, concat_string)
skill_template_means = 'ANALOGUE{:s}_FORECAST{:s}_DOMAIN{:s}_TARGET{:s}_WINDOW{:d}_MEMS{:d}{:s}_SpatialSkill{:s}{:s}{:s}{:s}_means'
skill_base_means = skill_template_means.format(analogue_var, forecast_var, target_domain_string, target_region, window, num_mems_to_take,
                                   smoothing_string, pass_string, testing_string, rmse_string, concat_string)

if picontrols_only:
    skill_base_info += '_piControlsOnly'
    skill_base_forecast += '_piControlsOnly'
    skill_base_means += '_piControlsOnly'
elif skip_local_hist:
    skill_base_info += '_SkipLocalHist{:d}'.format(nearby_hist)
    skill_base_forecast += '_SkipLocalHist{:d}'.format(nearby_hist)
    skill_base_means += '_SkipLocalHist{:d}'.format(nearby_hist)
elif strong_forcing_only:
    skill_base_info += '_StrongForcing{:d}'.format(earliest_hist)
    skill_base_forecast += '_StrongForcing{:d}'.format(earliest_hist)
    skill_base_means += '_StrongForcing{:d}'.format(earliest_hist)
skill_base_info += '.nc'
skill_base_forecast += '.nc'
skill_base_means += '.nc'
skill_file_info = os.path.join(skill_output_dir, skill_base_info)
skill_file_forecast = os.path.join(skill_output_dir, skill_base_forecast)
skill_file_means = os.path.join(skill_output_dir, skill_base_means)
print("Will write to:\n   {:s} and {:s} and {:s}\n".format(skill_file_info,skill_file_forecast,skill_file_means))

if pass_number > 1:
    if pass_number > 2:
        previous_pass_string = '_PASS{:d}'.format(pass_number - 1)
    else:
        previous_pass_string = ''
    previous_skill_base = skill_template_info.format(analogue_var, forecast_var, target_domain_string, target_region, window, num_mems_to_take,
                                                smoothing_string, previous_pass_string, testing_string, rmse_string, concat_string)
    previous_skill_file = os.path.join(skill_output_dir, previous_skill_base)
    print("Will read from previous skill data:\n   {:s}\n".format(previous_skill_file))

# ==================
# Get all the processed files that match
# ==================
processed_base = '{:s}_{:s}_*_*-*_Window{:d}{:s}_SpatialProcessed{:s}{:s}.nc'
processed_base = processed_base.format(analogue_var, target_domain_string, window, smoothing_string, testing_string, rmse_string)

id = hashlib.md5(processed_base.encode()).hexdigest()
globbed_file_save = os.path.join(processed_output_dir, 'PreCalc_GlobbedFiles_{:s}.pkl'.format(id))
if os.path.isfile(globbed_file_save) and look_for_globbed_savefile and not concatenate_hist_with_fut:
    print("Reading pre-globbed files: {:s}".format(globbed_file_save))
    if glob_only:
        raise ValueError('Only here to make the pre-calc glob files. Quitting...')
    with open(globbed_file_save, 'rb') as handle:
        processed_files = pickle.load(handle)
else:
    print("Globbing: {:s}".format(processed_base))
    processed_files = glob.glob(os.path.join(processed_output_dir, processed_base))
    # Sorting is important otherwise will get different behaviour on Jasmin/Ciclad:
    # BUT it means competing processes might try and access the same files at the same
    # time, potentially slowing everything down a lot. I don't think from a results point
    # of view this should be important, although it will affect repeatability
    processed_files.sort()

    if concatenate_hist_with_fut:
        processed_base2 = '{:s}_{:s}_*_*-*_Window{:d}{:s}_SpatialProcessed{:s}{:s}_CONCAT.pkl'
        processed_base2 = processed_base2.format(analogue_var, target_domain_string, window, smoothing_string, testing_string, rmse_string)
        processed_files2 = glob.glob(os.path.join(processed_output_dir, processed_base2))
        processed_files += processed_files2
        processed_files.sort()

        files2remove = []
        for ifile, this_file in enumerate(processed_files):
            if 'CONCAT' in this_file:
                # Need to remove associated future and non-CONCAT historical files
                parts = this_file.split('historical')
                for duplicate_expt in ['historical', 'rcp85', 'ssp585']:
                    dupe_file = parts[0] + duplicate_expt + parts[1][:-11] + '.pkl'  # Removes the CONCAT part
                    if dupe_file in processed_files:
                        # Should only match one future, and maybe a historical
                        files2remove.append(dupe_file)
        for this_file in files2remove:
            processed_files.remove(this_file)

    if not concatenate_hist_with_fut:
        with open(globbed_file_save, 'wb') as handle:
            print("Writing globbed files: {:s}".format(globbed_file_save))
            pickle.dump(processed_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if glob_only:
        raise ValueError('Only here to make the pre-calc glob files. Quitting...')

nfiles = len(processed_files)
print("Number of processed files found: {:d}".format(nfiles))
print(processed_files)

# ==================
# Read the obs data for validation
# This is literally just to get the "year" info
# ==================
if residual:
    raise ValueError('This is not defined yet')
else:
    if forecast_var == 'SST':
        forecast_save_file = hadisst_save_file
    elif forecast_var == 'DepthAverageT':
        forecast_save_file = en4_save_file
    elif forecast_var == 'SAT':
        forecast_save_file = hadcrut4_save_file
    with open(forecast_save_file, 'rb') as handle:
        print("Loading save file: {:s}".format(forecast_save_file))
        _, _, _, year_forecast_obs = pickle.load(handle, encoding='latin')

    if analogue_var == 'SST':
        analogue_save_file = hadisst_save_file
    elif analogue_var == 'DepthAverageT':
        analogue_save_file = en4_save_file
    elif analogue_var == 'SAT':
        analogue_save_file = hadcrut4_save_file
    with open(analogue_save_file, 'rb') as handle:
        print("Loading save file: {:s}".format(analogue_save_file))
        _, _, _, year_analogue_obs = pickle.load(handle, encoding='latin')

nyrs_analogue = len(year_analogue_obs)
nyrs_forecast = len(year_forecast_obs)
print("nyrs in year_forecast_obs = {:d}".format(nyrs_forecast))
print("nyrs in year_analogue_obs = {:d}".format(nyrs_analogue))

# ==================
# Define all the functions we will use
# ==================
def check_duplicate(model_in, expt_in, ens_in, index_in, stored_infos_in):
    is_duplicate = False
    ind_m = np.argwhere(stored_infos_in[:, 1] == model_in)
    ind_ex = np.argwhere(stored_infos_in[:, 2] == expt_in)
    ind_ens = np.argwhere(stored_infos_in[:, 3] == ens_in)
    common = np.intersect1d(ind_m, np.intersect1d(ind_ex, ind_ens))
    if len(common) > 0:
        for index in stored_infos_in[common, 4]:
            if not isinstance(index, str):
                if (index-index_in) < window:
                    is_duplicate = True
    return is_duplicate

# Function used below to search and store the best corrs and their info
def keep_best_corrs(input_corr, min_corr, corr_info, model, experiment, ens_mem,
                    num_mems_to_take=10, nlead_min=11, pass_number=1):
    # "input_corr" has analogue times, the rest have forecast times
    # Loop through the TARGET years
    for tt, year in enumerate(year_forecast_obs):
        if year in year_analogue_obs:
            tt_analogue = np.argwhere(year_analogue_obs == year)[0][0]
        else:
            # This year exists in the forecast data, but not the analogue data
            continue
        this_corr_info = corr_info[tt, :, :]

        # # Mask out STORED corrs that are duplicates
        # if pass_number > 1:
        #     this_corr_info = remove_close(this_corr_info)
        #     this_corr_info = sort_corrs(this_corr_info)
        #     min_corr[tt] = np.min([np.array(this_corr_info[:, 0]).min(), min_corr[tt]])

        # Mask out corrs that are lower than the best we've already found
        these_corrs = np.ma.masked_less(input_corr[tt_analogue, :], min_corr[tt])

        # Mask out corrs near the end of the time series (as these won't be useful in building the analogue)
        if len(these_corrs) < nlead_min:
            these_corrs[:].mask = True
        else:
            these_corrs[-nlead_min:].mask = True

        # Otherwise start the storing operations
        if these_corrs.count() > 0:
            # Flip so as to start with the best/unmasked. fill_value is important
            # as it means the masked values go at the start  (which becomes the end)
            these_corrs_sorted = np.flip(these_corrs.argsort(fill_value=-1), 0)
            for count, index in enumerate(these_corrs_sorted):
                if check_duplicate(model, experiment, ens_mem, index, this_corr_info):  # Not sure if this will work on pass=1
                    continue
                if skip_local_hist and (experiment[:4] == 'hist'):
                    if np.abs(year - year_model[index]) < nearby_hist:
                        continue
                elif strong_forcing_only and (experiment[:4] == 'hist'):  # piControl already removed. Keep all scenarios anyway
                    if year_model[index] < earliest_hist:
                        continue
                # "IF" below because we might be looking at a masked element
                # We overwrite the bottom value as the array is sorted by size (increasing)
                if these_corrs[index] > min_corr[tt]:
                    this_corr_info[0, 0] = these_corrs[index]
                    this_corr_info[0, 1] = model
                    this_corr_info[0, 2] = experiment
                    this_corr_info[0, 3] = ens_mem
                    this_corr_info[0, 4] = index  # This is lead=0
                else:
                    break  # Exit if presumably masked as all remainder should be too
                this_corr_info = sort_corrs(this_corr_info)
                if np.isfinite(this_corr_info[0, 0]):
                    min_corr[tt] = this_corr_info[0, 0]
        # elif pass_number > 1:
        #     print "I may have removed corr's and not added any..."
        #     print tt, model, experiment, ens_mem

        # Put the corr info into the yearly array
        corr_info[tt, :, :] = this_corr_info
    return min_corr, corr_info

# Function to pad the missing data inside the time series'
def check_and_pad1d(sst_in, year_in):
    diff = year_in[1:] - year_in[:-1]
    if len(np.unique(diff)) != 1:
        year_model = np.arange(year_in[0], year_in[-1] + 1) # Make new time axis
        nyrs_in = len(year_model)

        sst_model = np.ma.masked_all(shape=(nyrs_in))
        for iyr, year in enumerate(year_model):
            if year in year_in:
                iyr_in = np.argwhere(year_in == year)[0][0]
                sst_model[iyr] = sst_in[iyr_in]
            else:
                print(iyr, year)
                if ((year-1) in year_in) and ((year+1) in year_in):
                    # If data either side exists then interpolate
                    print(" ++ INTERPOLATING MISSING DATA")
                    iyr_in_m1 = np.argwhere(year_in == (year-1))[0][0]
                    iyr_in_p1 = np.argwhere(year_in == (year+1))[0][0]
                    sst_model[iyr] = (sst_in[iyr_in_m1] + sst_in[iyr_in_p1]) / 2.
    else:
        sst_model = sst_in
        year_model = year_in
    return sst_model, year_model

# Function to read in all the relevant forecast data associated with the best
# correlations found by "keep_best_corrs()"
def read_forecast_data(corr_info, nlead):
    forecast = np.ma.masked_all(shape=(nyrs_forecast, num_mems_to_take, nlead))
    analogue_means = np.ma.masked_all(shape=(nyrs_forecast, num_mems_to_take, len(mean_windows)))
    analogue_sds = np.ma.masked_all(shape=(nyrs_forecast, num_mems_to_take, len(mean_windows)))
    for tt, year in enumerate(year_forecast_obs):
        loop_t0 = time.time()
        print('{:d}/{:d}'.format(tt+1, nyrs_forecast))
        #if tt != 33: continue ###########################################
        for imem in range(num_mems_to_take):
            model = corr_info[tt, imem, 1]
            experiment = corr_info[tt, imem, 2]
            ens_mem = corr_info[tt, imem, 3]
            index = corr_info[tt, imem, 4]  # At lead=0

            if not isinstance(model, str):
                continue

            if model in cmip5_models:
                project = 'CMIP5'
            elif model in cmip6_models:
                project = 'CMIP6'
            else:
                # Probably an empty string due to removing last correlations just before finishing keep_best_corrs() loop
                continue

            if experiment == 'piControl':
                base_file_timeser = '{:s}_{:s}timeser_{:s}_{:s}_Annual.nc'.format(project, analogue_var, model, experiment)
            else:
                base_file_timeser = '{:s}_{:s}timeser_{:s}_{:s}-{:s}_Annual.nc'.format(project, analogue_var, model, experiment, ens_mem)
            this_file = os.path.join(forecast_datadir, base_file_timeser)

            if not os.path.isfile(this_file):
                print(tt, imem, corr_info[tt, imem, :])
                raise ValueError("This file should exist: {:s}".format(this_file))

            print(this_file)
            print(target_region)

            if os.path.isfile(this_file):
                ds_timeser = xr.open_dataset(this_file_time).to_array()
                print('Time series loaded')
                print(ds_timeser)
                sst_ts_in = np.ma.masked_array(ds_timeser.values)
                year_model_in = ds_timeser['time'].values

    #with open(this_file, 'rb') as handle:
        #sst_in, sst_ts_in, area_in, lon_in, lat_in, year_model_in = pickle.load(handle, encoding='latin')
        #nyrs = len(year_model_in)
        #_, nj_hist, ni_hist = sst_in.shape
            else:
                raise ValueError("{:s} does not exist".format(this_file_timeser))

            #with open(this_file, 'rb') as handle:
                #_, sst_ts_in, _, _, _, year_in = pickle.load(handle, encoding='latin')
                #sst_ts_in, year_in, _
                #_, sst_ts_in, _, _, _, year_in
                #print(sst_ts_in)
                #sst_ts_in  =  sst_ts_in[target_region]

            if len(year_in) == 0:
                print("File exists but seems to be empty...")
                print(this_file)
                continue

            # These need to be padded just like they were in AnalogueCache_Spatial.py
            sst_ts_in, year_in = check_and_pad1d(sst_ts_in, year_in)  # Forecast data

            # "index" might not be correct if we're using different analogue/forecast variables
            if analogue_var != forecast_var:
                if experiment == 'piControl':
                    base_file = '{:s}_{:s}_{:s}_{:s}_Annual_Regridded.pkl'.format(project, analogue_var, model, experiment)
                else:
                    base_file = '{:s}_{:s}_{:s}_{:s}-{:s}_Annual_Regridded.pkl'.format(project, analogue_var, model, experiment, ens_mem)
                this_file2 = os.path.join(analogue_datadir, base_file)

                if not os.path.isfile(this_file2):
                    print(tt, imem, corr_info[tt, imem, :])
                    raise ValueError("This file should exist: {:s}".format(this_file2))

                with open(this_file2, 'rb') as handle:
                    _, analogue_ts, _, _, _, year_in_analogue = pickle.load(handle, encoding='latin')
                    analogue_ts  =  analogue_ts[analogue_ts.keys()[0]]

                _, year_in_analogue = check_and_pad1d(analogue_ts, year_in_analogue)  #  Analogue data
                chosen_year = year_in_analogue[index]
                if chosen_year not in year_in:
                    print("This REALLY should exist. Forecast file is probably missing beginning/end", chosen_year, index, this_file, this_file2)
                    continue
                index2 = np.argwhere(year_in == chosen_year)[0][0]
                if index2 != index:
                    print("Changing index from {:d} to {:d}, Y{:d}. {:s} {:s} {:s}".format(index, index2, chosen_year, model, experiment, ens_mem))
                    print(this_file)
                    print(this_file2)
                index = index2

            # ntimes is to account for when the source data has less than nlead data points remaining
            ntimes = np.min([nlead, len(sst_ts_in[index:])])
            forecast[tt, imem, :ntimes] = sst_ts_in[index:index+ntimes]  # From index (lead=0) onwards

            # And store the mean/sd during the analogue period
            # (really should be made elsewhere, but would require more of a rewrite)
            # NOTE: These are the FORECAST var during the analogue creation window BUT not necessarily
            # for that LENGTH of window. I've now modified to use multiple different lengths
            for imean_window, mean_window in enumerate(mean_windows):
                if (index - mean_window) < 0:
                    continue
                analogue_means[tt, imem, imean_window] = np.ma.mean(sst_ts_in[index-mean_window:index])
                analogue_sds[tt, imem, imean_window] = np.ma.std(sst_ts_in[index-mean_window:index])
        loop_t1 = time.time()
        print("read_forecast_data forecast_year_obs loop {:.2f}".format((loop_t1 - loop_t0) / 60.))
    return forecast, analogue_means, analogue_sds

# # Calculate the skill
# def calculate_skill(forecast_in, nlead, multi=False, start_lead=[1], end_lead=[5], since1960=False):
#     # Calculate the skill, either for each validity time (annual mean)
#     # or for multi-annual means, using the following pattern:
#     # 1-5, 2-6, 3-7, 4-8, 5-9, 6-10, 1-10, 2-10
#     if since1960:
#         offset1960 = np.argwhere(year_ann == 1960)[0][0]
#     else:
#         offset1960 = 0
#     if not multi:
#         forecast_skill = np.ma.masked_all(nlead)
#         for ilead in lead_times:
#             this_target_ts = target_time_series[offset1960+ilead:]
#             this_source_ts = forecast_in[offset1960:nyrs-ilead,  ilead]
#             # print 'this_target_ts', this_target_ts
#             # print 'this_source_ts', this_source_ts
#             # print nyrs, ilead, offset1960
#             real = np.nonzero(this_target_ts *  this_source_ts)
#             _, _, corr, _, _ = stats.linregress(this_source_ts[real], this_target_ts[real])
#             forecast_skill[ilead] = corr
#     elif multi:
#         forecast_skill = np.ma.masked_all(len(start_lead))
#         for iforecast, (ss, ee) in enumerate(zip(start_lead, end_lead)):
#             nleads = (ee + 1 - ss)
#             this_target_ts = np.zeros(shape=(nyrs - (offset1960 + ee)))
#             for ilead in range(ss, ee+1, 1):  # +1 to include the end time
#                 this_target_ts += target_time_series[offset1960+ilead:nyrs-(ee-ilead)]
#             this_target_ts /= nleads
#
#             this_source_ts = np.ma.mean(forecast_in[offset1960:nyrs-ee, ss:ee+1], axis=1)
#
#             real = np.nonzero(this_target_ts *  this_source_ts)
#             _, _, corr, _, _ = stats.linregress(this_source_ts[real], this_target_ts[real])
#             forecast_skill[iforecast] = corr
#
#     return forecast_skill

# To remove elements that are close to each other
def remove_close(in_arr):  # 2D (nmems by 5)
    if in_arr.ndim == 1:
        return in_arr

    nmems, _ = in_arr.shape

    out_arr = in_arr.copy()

    # Go backwards as highest corr is last (note range starts from A but finishes one before B)
    for imem1 in range(nmems-1, 0, -1):
        better_model = in_arr[imem1, 1]
        if better_model == '':
            continue
        better_expt = in_arr[imem1, 2]
        better_ens = in_arr[imem1, 3]
        better_index = in_arr[imem1, 4]
        for imem2 in range(imem1-1, -1, -1):
            this_model = in_arr[imem2, 1]
            this_expt = in_arr[imem2, 2]
            this_ens = in_arr[imem2, 3]
            this_index = in_arr[imem2, 4]

            if (this_model == better_model) and (this_expt == better_expt) and (this_ens == better_ens) and np.abs(this_index - better_index) < window:
                out_arr[imem2, 0] = -1.0
                out_arr[imem2, 1:4] = ''
                out_arr[imem2, 4] = -window
    return out_arr

def sort_corrs(in_arr):
    out_arr = in_arr[in_arr[:, 0].argsort(fill_value=-1)]
    return out_arr

if analogue_var != forecast_var:
    with open(os.path.join(scripts_dir, '{:s}_MissAgainst_{:s}.txt'.format(forecast_var, analogue_var))) as handle:
        missing_model_expt_ens = handle.readlines()
        missing_model_expt_ens = [x.strip() for x in missing_model_expt_ens]
        missing_model_expt_ens = [x.split('_')[2] + '_' + x.split('_')[3] for x in missing_model_expt_ens]
    print("Missiang model+expt_ens combinations:")
    print(missing_model_expt_ens)

def check_allowed_file(filename, analogue_var, forecast_var):
    if analogue_var == forecast_var:
        return True
    this_model_expt_ens = os.path.basename(filename) # e.g. SST_+65+10+45-60_CanESM5_hist-nat-9_Window3_SpatialProcessed.pkl
    this_model_expt_ens = this_model_expt_ens.split('_')[2] + '_' + this_model_expt_ens.split('_')[3]
    this_model_expt_ens_piconversion = this_model_expt_ens[:-2]
    if (this_model_expt_ens in missing_model_expt_ens) or (this_model_expt_ens_piconversion in missing_model_expt_ens):
        print("Skipping ANALOGUE file because no associated FORECAST file")
        return False
    else:
        return True

# ==================
# Now finally process the data
# ==================
print('Finding best analogues')

t0 = time.time()

if pass_number == 1:
    # Must initialse as a float not integer!!
    min_ann_corr = np.repeat(-1., nyrs_forecast)

    # TO store the corr, model, experiment, ens_mem, index (year)
    ann_corr_info = np.ma.masked_all(shape=(nyrs_forecast, num_mems_to_take, 5), dtype=object)

    min_trend_corr = np.repeat(-1., nyrs_forecast)
    trend_corr_info = np.ma.masked_all(shape=(nyrs_forecast, num_mems_to_take, 5), dtype=object)
else:
    raise ValueError('Deprecated')
    with open(previous_skill_file, 'rb') as handle:
        print("Reading previous skill file: {:s}".format(previous_skill_file))
        data = pickle.load(handle, encoding='latin')
        if len(data) == 14:
            ann_corr_info, _, _, _, _, _, _, trend_corr_info, _, _, _, _, _, _ = data
        elif len(data) == 18:
            ann_corr_info, _, _, _, _, _, _, _, _, trend_corr_info, _, _, _, _, _, _, _, _ = data
        min_ann_corr = np.ma.min(ann_corr_info[:, :, 0], axis=1)
        min_trend_corr = np.ma.min(trend_corr_info[:, :, 0], axis=1)

for ifile, pf in enumerate(processed_files):
    print('{:d}/{:d} {:s}'.format(ifile+1, nfiles, pf))
    if not check_allowed_file(pf, analogue_var, forecast_var):
        print(" ++ Skipping this file")
        continue
    # if ifile > 10: continue###############################################################################
    split = os.path.basename(pf).split('_')
    experiment_and_ens = split[3].split('-')
    model = split[2]
    if len(experiment_and_ens) == 2:
        experiment = experiment_and_ens[0]
        ens_mem = experiment_and_ens[1]
    elif len(experiment_and_ens) == 3:
        experiment = experiment_and_ens[0] + '-' + experiment_and_ens[1]
        ens_mem = experiment_and_ens[2]

    if picontrols_only:
        if experiment != 'piControl':
            continue
    elif strong_forcing_only:
        if experiment == 'piControl':
            continue

    #with open(pf, 'rb') as handle:
    ds = xr.open_dataset(pf).to_array()
    ds_data = np.ma.masked_array(ds.values)
    print(ds_data[0])
    corr_annual = ds_data[0]
    corr_trend = ds_data[1]
    year_model = ds['time'].values
    #corr_annual, corr_trend, _, year_model = pickle.load(handle, encoding='latin')
    min_ann_corr, ann_corr_info = keep_best_corrs(corr_annual, min_ann_corr, ann_corr_info, model,
                                                  experiment, ens_mem, num_mems_to_take=num_mems_to_take,
                                                  nlead_min=nlead, pass_number=pass_number)
    min_trend_corr, trend_corr_info = keep_best_corrs(corr_trend, min_trend_corr, trend_corr_info, model,
                                                      experiment, ens_mem, num_mems_to_take=num_mems_to_take,
                                                      nlead_min=nlead, pass_number=pass_number)


t1 = time.time()
print("Time taken do keep_best_corrs = {:.2f} minutes".format((t1 - t0) /60.))

print('Finding models and data associated with best analogues')
trend_forecast, trend_forecast_means, trend_forecast_sds = read_forecast_data(trend_corr_info, nlead)
t2 = time.time()
print("Time taken do read_forecast_data (trends) = {:.2f} minutes".format((t2 - t1) /60.))

ann_forecast, ann_forecast_means, ann_forecast_sds = read_forecast_data(ann_corr_info, nlead)
t3 = time.time()
print("Time taken do read_forecast_data (annual) = {:.2f} minutes".format((t3 - t2) /60.))
# print('Creating actual analogue forecasts')
# trend_forecast_anomt0 = trend_forecast - np.repeat(trend_forecast[:, :, 0][:, :, None], nlead, axis=2)
# trend_forecast_anomt0_mmm = np.ma.mean(trend_forecast_anomt0, axis=1)
# trend_forecast_anomt0_mmm_recentred = trend_forecast_anomt0_mmm + np.repeat(target_time_series[:, None],
#                                                                             nlead, axis=1)
#
# ann_forecast_anomt0 = ann_forecast - np.repeat(ann_forecast[:, :, 0][:, :, None], nlead, axis=2)
# ann_forecast_anomt0_mmm = np.ma.mean(ann_forecast_anomt0, axis=1)
# ann_forecast_anomt0_mmm_recentred = ann_forecast_anomt0_mmm + np.repeat(target_time_series[:, None], nlead, axis=1)
#
# print('Creating skill of analogue forecasts')
# trend_forecast_skill = calculate_skill(trend_forecast_anomt0_mmm_recentred, nlead)
# ann_forecast_skill = calculate_skill(ann_forecast_anomt0_mmm_recentred, nlead)
# trend_forecast_skill1960 = calculate_skill(trend_forecast_anomt0_mmm_recentred, nlead, since1960=True)
# ann_forecast_skill1960 = calculate_skill(ann_forecast_anomt0_mmm_recentred, nlead, since1960=True)
#
# print('Creating multiannual mean skill of analogue forecasts')
# trend_forecast_multiskill = calculate_skill(trend_forecast_anomt0_mmm_recentred, nlead, multi=True,
#                                             start_lead=start_lead, end_lead=end_lead)
# ann_forecast_multiskill = calculate_skill(ann_forecast_anomt0_mmm_recentred, nlead, multi=True,
#                                           start_lead=start_lead, end_lead=end_lead)
# trend_forecast_multiskill1960 = calculate_skill(trend_forecast_anomt0_mmm_recentred, nlead, multi=True,
#                                                 start_lead=start_lead, end_lead=end_lead, since1960=True)
# ann_forecast_multiskill1960 = calculate_skill(ann_forecast_anomt0_mmm_recentred, nlead, multi=True,
#                                               start_lead=start_lead, end_lead=end_lead, since1960=True)
#
# with open(skill_file, 'wb') as handle:
#     print "Writing save file: {:s}".format(skill_file)
#     skill_data = [ann_corr_info, ann_forecast, ann_forecast_means, ann_forecast_sds,
#                   ann_forecast_anomt0_mmm_recentred, ann_forecast_skill,
#                   ann_forecast_skill1960, ann_forecast_multiskill, ann_forecast_multiskill1960,
#                   trend_corr_info, trend_forecast, trend_forecast_means, trend_forecast_sds,
#                   trend_forecast_anomt0_mmm_recentred, trend_forecast_skill,
#                   trend_forecast_skill1960, trend_forecast_multiskill, trend_forecast_multiskill1960]
#     pickle.dump(skill_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Write output to xarray for export to .nc
# This has to happen in 3 files, because there are 3 different dimensionalities in the output data: 150x100x5, 150x100x11, 150x100x16
# (the dimensions are, as far as I can tell: time (150), member (100), infos (5), ntimes (11), time window (16))
info_array = ['these_corrs','model','experiment','ens_mem','index']
mem_array = range(1,101,1)
window_array = range(1,17,1)
ntimes_array = range(1,12,1)

print(year_model.shape, year_analogue_obs.shape, year_forecast_obs.shape, ann_corr_info.shape, ann_forecast.shape, ann_forecast_means.shape, ann_forecast_sds.shape, trend_corr_info.shape, trend_forecast.shape, trend_forecast_means.shape, trend_forecast_sds.shape)

info_xr = xr.DataArray(ann_corr_info, name='ann_corr_info', dims = ['time','member','info'], coords = {'time': (['time'],year_analogue_obs), 'member': (['member'],mem_array), 'info': (['info'],info_array)}).to_dataset(name='ann_corr_info')
info_xr['trend_corr_info'] = xr.DataArray(trend_corr_info, name='trend_corr_info', dims = ['time','member','info'], coords = {'time': (['time'],year_analogue_obs), 'member': (['member'],mem_array), 'info': (['info'],info_array)})

forecast_xr = xr.DataArray(ann_forecast, name='ann_forecast', dims = ['time','member','ntimes'], coords = {'time': (['time'],year_analogue_obs), 'member': (['member'],mem_array), 'ntimes': (['ntimes'],ntimes_array)}).to_dataset(name='ann_forecast')
forecast_xr['trend_forecast'] = xr.DataArray(trend_forecast, name='trend_forecast', dims = ['time','member','ntimes'], coords = {'time': (['time'],year_analogue_obs), 'member': (['member'],mem_array), 'ntimes': (['ntimes'],ntimes_array)})

means_xr = xr.DataArray(ann_forecast_means, name='ann_forecast_means', dims = ['time','member','window'], coords = {'time': (['time'],year_analogue_obs), 'member': (['member'],mem_array), 'window': (['window'],window_array)}).to_dataset(name='ann_forecast_means')
means_xr['ann_forecast_sds'] = xr.DataArray(ann_forecast_sds, name='ann_forecast_sds', dims = ['time','member','window'], coords = {'time': (['time'],year_analogue_obs), 'member': (['member'],mem_array), 'window': (['window'],window_array)})
means_xr['trend_forecast_means'] = xr.DataArray(trend_forecast_means, name='trend_forecast_means', dims = ['time','member','window'], coords = {'time': (['time'],year_analogue_obs), 'member': (['member'],mem_array), 'window': (['window'],window_array)})
means_xr['trend_forecast_sds'] = xr.DataArray(trend_forecast_sds, name='trend_forecast_sds', dims = ['time','member','window'], coords = {'time': (['time'],year_analogue_obs), 'member': (['member'],mem_array), 'window': (['window'],window_array)})

#with open(skill_file, 'wb') as handle:
#    print("Writing save file: {:s}".format(skill_file))
#    skill_data = [ann_corr_info, ann_forecast, ann_forecast_means, ann_forecast_sds,
#                  trend_corr_info, trend_forecast, trend_forecast_means, trend_forecast_sds]
#    pickle.dump(skill_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

info_xr.to_netcdf(path=skill_file_info,format="NETCDF4")
forecast_xr.to_netcdf(path=skill_file_forecast,format="NETCDF4")
means_xr.to_netcdf(path=skill_file_means,format="NETCDF4")

t4 = time.time()
print("Time taken to write save-file = {:.2f}".format((t4 - t3) / 60.))

print("COMPLETE!")
