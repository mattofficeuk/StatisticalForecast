#!/usr/bin/env python3
# The above is required for sbatch submission

residual = False
method='RMSE'  # Overwrite the "spatial correlation" with RMSE instead
# analogue_var = 'DepthAverageT'
analogue_var = 'SST'

# Whether to save the trends data. This will look into a separate file (which must exist)
# that tells the script whether to save the trends or not. If these trends are not required
# then this script will exit. Note that the "target_region" is hardcoded as europe1
save_trends = False

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

myhost = os.uname()[1]
print("myhost = {:s}".format(myhost))
usr = os.environ["USER"]


if 'ciclad' in myhost:
    # Ciclad options
    datadir = '/data/mmenary/python_saves/CMIP_{:s}/'.format(analogue_var)
    processed_output_dir = '/prodigfs/ipslfs/dods/mmenary/AnalogueCache'
    scripts_dir = '/home/mmenary/python/scripts/'
    hadisst_save_file = '/data/mmenary/python_saves/HadISST_AnnualMapCMIPStyleRegridded.pkl'
    hadisst_save_file_residual = '/data/mmenary/python_saves/HadISST_AnnualMapCMIPStyleRegridded_Residual.pkl'
    en4_save_file = '/data/mmenary/python_saves/EN4_0-500m_AnnualMapCMIPStyleRegridded.pkl'
    en4_save_file_residual = '/data/mmenary/python_saves/EN4_0-500m_AnnualMapCMIPStyleRegridded_Residual.pkl'
else:
    # Jasmin options
    #datadir = '/gws/nopw/j04/acsis/mmenary/python_saves/CMIP_{:s}/'.format(analogue_var)
    datadir = '/work/scratch-nopw/{}/CMIP_{:s}/'.format(usr, analogue_var)
    #processed_output_dir = '/gws/nopw/j04/acsis/mmenary/python_saves/AnalogueCache/'
    processed_output_dir = '/work/scratch-nopw/{}/AnalogueCache'.format(usr)
    scripts_dir = '/home/users/{}/python/scripts/'.format(usr)
    hadisst_save_file = '/home/users/{}/data/HadISST_AnnualMapCMIPStyleRegridded.pkl'.format(usr)
    hadisst_save_file_residual = '/home/users/{}/data/HadISST_AnnualMapCMIPStyleRegridded_Residual.pkl'.format(usr)
    en4_save_file = '/home/users/{}/data/EN4_0-500m_AnnualMapCMIPStyleRegridded.pkl'.format(usr)
    en4_save_file_residual = '/home/users/{}/data/EN4_0-500m_AnnualMapCMIPStyleRegridded_Residual.pkl'.format(usr)  # Not made yet

def read_target_domain(in_string):
    out_list = []
    for ii in range(4):
        out_list.append(np.int(in_string[ii*3:(ii+1)*3]))
    return out_list

# ==============
# Constants we're reading in
# ==============
model = sys.argv[1]
experiment = sys.argv[2]
ens_mem = sys.argv[3]
window = np.long(sys.argv[4])
target_domain_string = sys.argv[5]
smoothing = np.long(sys.argv[6])  # This is the x/y (regridded) spatial smoothing
testing = sys.argv[7]
clim_string = sys.argv[8]
concatenate_string = sys.argv[9] # Whether to try and merge available future experiments into the historical ones (but not the other
                                 # way around) in order to avoid an artificial cutoff around 2005/2015

clim_start = int(clim_string[:4])
clim_end = int(clim_string[5:])

if (clim_start != 1960) or (clim_end != 1990):
    processed_output_dir += '_{:d}-{:d}'.format(clim_start, clim_end)

if concatenate_string == 'True':
    concatenate_hist_with_fut = True
else:
    concatenate_hist_with_fut = False

if (smoothing % 2) != 1:
    raise ValueError("Smoothing must be odd")

target_domain = read_target_domain(target_domain_string)

# ==============
# Constants currently fixed
# ==============
# basedir = '/home/mmenary/python/notebooks/cmip6_2/output/'
nj = 180
ni = 360
lon_re = np.repeat((np.arange(-180, 180) + 0.5)[np.newaxis, :], nj, axis=0)
lat_re = np.repeat((np.arange(-90, 90) + 0.5)[:, None], ni, axis=1)

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

if residual:
    residual_string = 'Residual'
else:
    residual_string = ''

if method == 'RMSE':
    rmse_string = '_RMSEmethod'
else:
    rmse_string = ''

# ==================
# Note the final target save file, does it already exist?
# ==================
processed_base = '{:s}_{:s}_{:s}_{:s}-{:s}_Window{:d}{:s}_Spatial{:s}Processed{:s}{:s}'
processed_filled = processed_base.format(analogue_var, target_domain_string, model, experiment, ens_mem,
                                         window, smoothing_string, residual_string, testing_string, rmse_string)
processed_file = os.path.join(processed_output_dir, processed_filled) + '.pkl'
print(processed_file)

target_saved_base = '{:s}ObsSaved_{:s}_Window{:d}{:s}_Spatial{:s}Processed{:s}{:s}.pkl'
target_saved_filled = target_saved_base.format(analogue_var, target_domain_string, window, smoothing_string,
                                               residual_string, testing_string, rmse_string)
target_saved_file = os.path.join(processed_output_dir, target_saved_filled)
print(target_saved_file)
if os.path.isfile(target_saved_file):
    print("Loading time-saver target data:  {:s}".format(target_saved_file))
    with open(target_saved_file, 'rb') as handle:
        target_saved = pickle.load(handle)
else:
    target_saved = {}
target_keys = target_saved.keys()

if save_trends:
    target_region = 'europe1'
    target_region = 'subpolar_gyre'
    chosen_num_mems = 100
    trends_or_annual = '_ANNUAL'
    requested_files_file = 'InputFilesList{:s}_ANALOGUE{:s}_DOMAIN{:s}_TARGET{:s}_WINDOW{:d}_MEMS{:d}_SpatialSkill{:s}{:s}.txt'
    requested_files_file = requested_files_file.format(trends_or_annual, analogue_var, target_domain_string, target_region, window, chosen_num_mems, smoothing_string, rmse_string)
    requested_files_file = os.path.join(scripts_dir, requested_files_file)
    requested_files = []
    with open(requested_files_file, 'r') as f:
        for line in f.readlines():
            requested_files.append(line.strip())

    if processed_filled not in requested_files:
        raise ValueError('Will not continue. Only saving trends files and this one is not requested: {:s}'.format(processed_filled))

    trends_base = '{:s}_SavedTrends_{:s}_{:s}_{:s}-{:s}_Window{:d}{:s}_Spatial{:s}Processed{:s}{:s}.pkl'
    trends_filled = trends_base.format(analogue_var, target_domain_string, model, experiment, ens_mem, window,
                                       smoothing_string, residual_string, testing_string, rmse_string)
    trends_file = os.path.join(processed_output_dir, trends_filled)
    print(trends_file)

    if os.path.isfile(trends_file):
        raise ValueError("Already created this trends file - will not re-create")

# ==================
# Read the model data
# ==================
cmip5_list_file = os.path.join(scripts_dir, 'cmip5_list.txt')
cmip6_list_file = os.path.join(scripts_dir, 'cmip6_list.txt')

cmip5_models = []
with open(cmip5_list_file, 'r') as f:
    for line in f.readlines():
        cmip5_models.append(line.strip())

cmip6_models = []
with open(cmip6_list_file, 'r') as f:
    for line in f.readlines():
        cmip6_models.append(line.strip())

if model in cmip5_models:
    project = 'CMIP5'
elif model in cmip6_models:
    project = 'CMIP6'
else:
    raise ValueError("Unknown model")

if experiment == 'piControl':
    base_file = '{:s}_{:s}_{:s}_{:s}_Annual.pkl'.format(project, analogue_var, model, experiment)
else:
    base_file = '{:s}_{:s}_{:s}_{:s}-{:s}_Annual.pkl'.format(project, analogue_var, model, experiment, ens_mem)
this_file = os.path.join(datadir, base_file)
print("Attempting to read: {:s}".format(this_file))

if os.path.isfile(this_file):
    with open(this_file, 'rb') as handle:
        sst_in, sst_ts_in, area_in, lon_in, lat_in, year_model_in = pickle.load(handle, encoding='latin')
        nyrs = len(year_model_in)
        _, nj_hist, ni_hist = sst_in.shape
else:
    raise ValueError("{:s} does not exist".format(this_file))

if concatenate_hist_with_fut:
    print("Considering concatenating")
    # Attempt to concatenate a future file if this experiment is historical. It will be difficult to stop similar years
    # in the hist+fut and fut being chosen by Skill, except note that no future experients _were_ chosen by Skill
    # for W=35, so they shouldn't be now either. Not robust to changes of Window, method, etc though...
    if experiment == 'historical':
        if project == 'CMIP5':
            experiment_fut = 'rcp85'  # Just stick to RCP85/ssp585 for now
        elif project == 'CMIP6':
            experiment_fut = 'ssp585'
        base_file_fut = '{:s}_{:s}_{:s}_{:s}-{:s}_Annual.pkl'.format(project, analogue_var, model, experiment_fut, ens_mem)
        this_file_fut = os.path.join(datadir, base_file_fut)

        if os.path.isfile(this_file_fut):
            with open(this_file_fut, 'rb') as handle:
                sst_fut_in, _, _, _, _, year_model_fut_in = pickle.load(handle,encoding='latin')
                _, nj_fut, ni_fut = sst_fut_in.shape

            if (nj_hist == nj_fut) and (ni_hist == ni_fut):
                print(" --||-- Concatenating historical and future simulations")
                unique_years = np.unique(np.concatenate((year_model_in, year_model_fut_in)))
                concat_sst = np.ma.masked_all(shape=(len(unique_years), nj_hist, ni_hist))
                print(concat_sst.shape)
                for tt, year in enumerate(unique_years):
                    if year in year_model_fut_in:
                        # Choose future first so if there are duplicates we choose the future experiment
                        iyr = np.argwhere(year_model_fut_in == year)[0][0]
                        concat_sst[tt, :, :] = sst_fut_in[iyr, :, :]
                    elif year in year_model_in:
                        iyr = np.argwhere(year_model_in == year)[0][0]
                        concat_sst[tt, :, :] = sst_in[iyr, :, :]

                sst_in = concat_sst
                year_model_in = unique_years
                processed_file = os.path.join(processed_output_dir, processed_filled) + '_CONCAT.pkl'
            else:
                print("Will not concatenate - future file is different shape")
        else:
            print("Will not concatenate - no future file")
    else:
        print("Will not concatenate - not historical")

# ==================
# Read the historical data for validation
# ==================
if residual:
    if analogue_var == 'SST':
        with open(hadisst_save_file_residual, 'rb') as handle:
            target_sst_regridded, _, _, _, _, year_ann = pickle.load(handle,encoding='latin')
    elif analogue_var == 'DepthAverageT':
        with open(en4_save_file_residual, 'rb') as handle:
            target_sst_regridded, _, _, _, _, year_ann = pickle.load(handle,encoding='latin')
else:
    if analogue_var == 'SST':
        with open(hadisst_save_file, 'rb') as handle:
            target_sst_regridded, _, _, _, _, year_ann = pickle.load(handle,encoding='latin')
    elif analogue_var == 'DepthAverageT':
        with open(en4_save_file, 'rb') as handle:
            target_sst_regridded, _, _, _, _, year_ann = pickle.load(handle,encoding='latin')

# Make climatology for later
t0 = np.argwhere(year_ann == clim_start)[0][0]
t1 = np.argwhere(year_ann == clim_end)[0][0]
target_sst_regrided_clim = np.ma.mean(target_sst_regridded[t0:t1, :, :], axis=0)

# ==================
# Define all the functions we will use
# ==================
def get_area():
    radius_earth = 6.4e6
    dlon = dlat = np.radians(1.)
    area = np.ones(shape=(nj, ni))
    for jj in range(nj):
        area[jj, :] *= radius_earth**2 * np.cos(np.radians(lat_re[jj])) * np.sin(dlon) * np.sin(dlat)
    return area

def regrid_sst(sst_in, year_in):
    nyrs = len(year_in)
    area = get_area()
    sst_regridded = np.ma.masked_all(shape=(nyrs, nj, ni))

    for tt in range(nyrs):
#         if tt not in [97, 98, 99, 100]: continue
        print('{:d}/{:d}'.format(tt, nyrs))
        sst_regridded[tt, :, :] = interpolate.griddata(np.array([lat_in.ravel(), lon_in.ravel()]).T,
                                                       sst_in[tt, :, :].ravel(), (lat_re, lon_re),
                                                       method='linear')
    mask_regridded = interpolate.griddata(np.array([lat_in.ravel(), lon_in.ravel()]).T,
                                          sst_in[0, :, :].mask.ravel(), (lat_re, lon_re),
                                          method='linear')
    sst_regridded = np.ma.array(sst_regridded, mask=np.repeat(mask_regridded[np.newaxis, :, :], nyrs, axis=0))
    return sst_regridded

def mask_by_domain(sst_in, year_in):
    # TODO: If smoothing then increase this mask size appropriately so it won't be too small later
    mask = ((lon_re > (target_domain[1]+smoothing//2)) | (lon_re < (target_domain[3]-smoothing//2)) | (lat_re > (target_domain[0]+smoothing//2)) | (lat_re < (target_domain[2]-smoothing//2)))
    mask = np.repeat(mask[None, :, :], len(year_in), axis=0)
    sst_masked = np.ma.array(sst_in, mask=mask)
    return sst_masked

def compute_trend_in_windows(sst_masked, year_in, window_in, tol=0.9):
    sst_trend = np.ma.masked_all(shape=sst_masked.shape)
    if window_in > sst_masked.shape[0]:
        raise ValueError('Window greater than model year length. Quitting (this is fine).')
    nyrs_in = len(year_in)
    min_len = window_in * tol

    for tt, year in enumerate(year_in[:(nyrs_in+1)-window]):
#         if tt != 90: continue
        print('{:d}/{:d} ({:d}->{:d})'.format(tt+1, len(year_in[:nyrs_in-window]), year, year+window))
        start_time = time.time()

        this_map = sst_masked[tt:tt+window, :, :]
        xx = np.arange(window)
        for jj in range(0, nj, step):
            for ii in range(0, ni, step):
                local_ts = this_map[:, jj, ii]
                real = np.nonzero(local_ts * local_ts)[0]
#                 if (jj == 120) and (ii == 150): print len(real), min_len
                if len(real) < min_len:
                    continue
                grad, _, _, _, _ = stats.linregress(xx, local_ts)
                sst_trend[tt+window-1, jj, ii] = grad

        end_time = time.time()
        elapsed = end_time - start_time
        if tt == 0:
            print(" ++||++ Trends: Time taken (1 loop): {:.1f} seconds".format(elapsed))
            print(" ++||++ Trends: Estimated total time: {:.1f} minutes".format((elapsed * len(year_in[:nyrs_in-window])) / 60.))
            print(" ++||++ Trends: Time now: ", time.ctime())

    sst_trend = np.ma.array(sst_trend, mask=sst_masked.mask)
    return sst_trend

def compute_mean_in_windows(sst_masked, year_in, window_in, tol=0.9):
    sst_mean = np.ma.masked_all(shape=sst_masked.shape)
    if window_in > sst_masked.shape[0]:
        raise ValueError('Window greater than model year length. Quitting (this is fine).')
    nyrs_in = len(year_in)
    min_len = window_in * tol

    for tt, year in enumerate(year_in[:(nyrs_in+1)-window]):
        print('{:d}/{:d} ({:d}->{:d})'.format(tt, len(year_in[:nyrs_in-window]), year, year+window))

        sst_mean[tt+window-1, :, :] = np.ma.mean(sst_masked[tt:tt+window, :, :], axis=0)

    sst_mean = np.ma.array(sst_mean, mask=sst_masked.mask)
    return sst_mean

def make_anomaly(sst_masked, year_in, target=False, model=False):
    if ((not target) and (not model)) or (target and model):
        raise ValueError('Must set one and one only of /target or /model')
    if target:
        sst_masked = sst_masked - target_sst_regrided_clim[np.newaxis, :, :]
    elif model:
        sst_masked = sst_masked - sst_clim[np.newaxis, :, :]
    return sst_masked

def smooth(in_arr):
    # Apply some spatial smoothing (weighted) to the trend maps
    # TODO
    min_frac = 0.4
    smoothing_sq = smoothing**2.
    out_arr = np.ma.masked_all_like(in_arr)
    print(in_arr.shape)
    nt, _, _ = in_arr.shape
    area = get_area()
    for jj in range(0, nj-smoothing, step):
        print('{:d}/{:d}'.format(jj, nj-smoothing))
        jj_0 = jj
        jj_1 = jj + smoothing
        for ii in range(0, ni-smoothing, step):
            ii_0 = ii
            ii_1 = ii + smoothing
            area_flat = area[jj_0:jj_1, ii_0:ii_1].flatten()
            for tt in range(nt):
                if np.ma.is_masked(in_arr[tt, jj+smoothing//2, ii+smoothing//2]):
                    continue
                ts = in_arr[tt, jj_0:jj_1, ii_0:ii_1].flatten()
                real = np.nonzero(ts * area_flat)[0]
                if len(real)/smoothing_sq < min_frac:
                    continue
                area_mean = np.ma.sum(ts[real] * area_flat[real]) / np.ma.sum(area_flat[real])
                out_arr[tt, jj+smoothing//2, ii+smoothing//2] = area_mean
    return out_arr

def mask_coasts(in_arr, n_cells=2):
    # n_cells: Number of grid cells from coast to mask
    # This might not be necessary if we smooth
    pass

def calc_rmse(aa, bb):
    return np.sqrt(np.ma.mean((aa - bb)**2))

def compute_spatial_correlation(sst_obs, year_obs, sst_model, year_model):
    nyrs_obs = len(year_obs)
    nyrs_model = len(year_model)
    corr = np.ma.masked_all(shape=(nyrs_obs, nyrs_model))
    for tt_t, year_t in enumerate(year_obs):
        print('{:d}/{:d}'.format(tt_t, nyrs_obs))
        for tt_m, year_m in enumerate(year_model):
#            if tt_m not in [50, 99]: continue
            obs_data = sst_obs[tt_t, :, :].flatten()
            model_data = sst_model[tt_m, :, :].flatten()
            real = np.nonzero(obs_data  * model_data)
            if len(real[0]) < 5:
                continue
            if method == 'RMSE':
                # Not actually RMSE, but we need high values to be better
                this_corr = 1. / calc_rmse(obs_data[real], model_data[real])
            else:
                this_corr = np.corrcoef(obs_data[real], model_data[real])[0][1]
            corr[tt_t, tt_m] = this_corr

    return corr

def check_and_pad(sst_in, year_in):
    diff = year_in[1:] - year_in[:-1]
    if len(np.unique(diff)) != 1:
        year_model = np.arange(year_in[0], year_in[-1] + 1) # Make new time axis
        nyrs = len(year_model)

        _, nj, ni = sst_in.shape
        sst_model = np.ma.masked_all(shape=(nyrs, nj, ni))
        for iyr, year in enumerate(year_model):
            if year in year_in:
                iyr_in = np.argwhere(year_in == year)[0][0]
                sst_model[iyr, :, :] = sst_in[iyr_in, :, :]
            else:
                print(iyr, year)
                if ((year-1) in year_in) and ((year+1) in year_in):
                    # If data either side exists then interpolate
                    print(" ++ INTERPOLATING MISSING DATA")
                    iyr_in_m1 = np.argwhere(year_in == (year-1))[0][0]
                    iyr_in_p1 = np.argwhere(year_in == (year+1))[0][0]
                    sst_model[iyr, :, :] = (sst_in[iyr_in_m1, :, :] + sst_in[iyr_in_p1, :, :]) / 2.
    else:
        sst_model = sst_in
        year_model = year_in
    return sst_model, year_model

def find_climatology_for_model(model):
    climatology_file = os.path.join(datadir, 'CMIP_{:s}_{:s}_historical-EnsMn_TM{:d}-{:d}_Annual.pkl'.format(analogue_var, model, clim_start, clim_end))
    if os.path.isfile(climatology_file):
        with open(climatology_file, 'rb') as handle:
            sst_clim = pickle.load(handle,encoding='latin')
    else:
        print("No climatology file exists, filling with missing data: {:s}".format(climatology_file))
        sst_clim = np.ma.masked_all(shape=(nj, ni))
    return sst_clim

# ==================
# Now finally process the data
# ==================
print('Padding MODEL data if required')
sst_model, year_model = check_and_pad(sst_in, year_model_in)

print('Regridding MODEL {:s} map'.format(analogue_var))
sst_regridded = regrid_sst(sst_model, year_model)

print('Masking MODEL {:s}'.format(analogue_var))
sst_masked = mask_by_domain(sst_regridded, year_model)

print('Masking TARGET {:s}'.format(analogue_var))
if 'target_masked' in target_keys:
    target_masked = target_saved['target_masked']
else:
    target_masked = mask_by_domain(target_sst_regridded, year_ann)
    target_saved['target_masked'] = target_masked

print("Computing trends for TARGET")
if 'target_masked_trend' in target_keys:
    # Note: If smoothing this will actually be the smoothed version, but it doesn't matter
    target_masked_trend = target_saved['target_masked_trend']
else:
    target_masked_trend = compute_trend_in_windows(target_masked, year_ann, window)
    target_saved['target_masked_trend'] = target_masked_trend

print("Computing trends for MODEL")
sst_masked_trend = compute_trend_in_windows(sst_masked, year_model, window)

print("Computing window-means for TARGET")
if 'target_masked_mean' in target_keys:
    target_masked_mean = target_saved['target_masked_mean']
else:
    target_masked_mean = compute_mean_in_windows(target_masked, year_ann, window)
    target_saved['target_masked_mean'] = target_masked_mean

print("Computing window-means for MODEL")
sst_masked_mean = compute_mean_in_windows(sst_masked, year_model, window)

print("Making anomaly w.r.t. {:d}-{:d} mean for TARGET".format(clim_start, clim_end))
if 'target_masked_mean_anom' in target_keys:
    # Note: If smoothing this will actually be the smoothed version, but it doesn't matter
    target_masked_mean_anom = target_saved['target_masked_mean_anom']
else:
    target_masked_mean_anom = make_anomaly(target_masked_mean, year_ann, target=True)
    target_saved['target_masked_mean_anom'] = target_masked_mean_anom

print("Making anomaly w.r.t. {:d}-{:d} mean for MODEL".format(clim_start, clim_end))
sst_clim = find_climatology_for_model(model)
sst_masked_mean_anom = make_anomaly(sst_masked_mean, year_model, model=True)

# Smooth first if necessary. Don't need to do this for each window as we will
# cut off the ends each time anyway
if smoothing >  1:
    print("Smoothing TARGET {:s} trend".format(analogue_var))
    if 'target_masked' in target_keys:
        target_masked_trend = target_saved['target_masked_trend']
    else:
        target_masked_trend = smooth(target_masked_trend)
        target_saved['target_masked_trend'] = target_masked_trend

    print("Smoothing MODEL {:s} trend".format(analogue_var))
    sst_masked_trend = smooth(sst_masked_trend)

    print("Smoothing TARGET {:s} means".format(analogue_var))
    if 'target_masked_mean_anom' in target_keys:
        target_masked_mean_anom = target_saved['target_masked_mean_anom']
    else:
        target_masked_mean_anom = smooth(target_masked_mean_anom)
        target_saved['target_masked_mean_anom'] = target_masked_mean_anom

    print("Smoothing MODEL {:s} means".format(analogue_var))
    sst_masked_mean_anom = smooth(sst_masked_mean_anom)

print("Computing spatial correlation (of annual means)")
corr_ann = compute_spatial_correlation(target_masked_mean_anom, year_ann, sst_masked_mean_anom, year_model)

print("Computing spatial correlation (of trends)")
corr_trend = compute_spatial_correlation(target_masked_trend, year_ann, sst_masked_trend, year_model)

print("Attempting to open file for writing:  {:s}".format(processed_file))
with open(processed_file, 'wb') as handle:
    print("Writing save file: {:s}".format(processed_file))
    processed_data = [corr_ann, corr_trend, year_ann, year_model]
    pickle.dump(processed_data, handle,  protocol=pickle.HIGHEST_PROTOCOL)

if save_trends:
    print("Attempting to open file for writing:  {:s}".format(trends_file))
    with open(trends_file, 'wb') as handle:
        print("Writing save file: {:s}".format(trends_file))
        processed_data = [target_masked_trend, sst_masked_trend, target_masked_mean_anom,
                          sst_masked_mean_anom, year_ann, year_model]
        pickle.dump(processed_data, handle,  protocol=pickle.HIGHEST_PROTOCOL)

if not os.path.isfile(target_saved_file):
    print("Attempting to create time-saver target file:  {:s}".format(target_saved_file))
    with open(target_saved_file, 'wb') as handle:
        pickle.dump(target_saved, handle,  protocol=pickle.HIGHEST_PROTOCOL)

print("COMPLETE!")
