#!/usr/bin/env python2.7
# The above is required for sbatch submission

# As AnalyseAnalogueSource_Jasmin.py but written better!

testing = False ######################################################################################
testing2 = False
clever_skill = True
skill_only = False
recreate_skill = True
do_final_sd_scaling = False

do_random = True
only_do_random = False

# chosen_target_domain = '+65+10+45-60'
analogue_var = 'SST'
forecast_var = 'SST'
chosen_target_region = 'subpolar_gyre'

trends = False
annual = True

clim_start, clim_end = 1960, 1990
# clim_start, clim_end = 1980, 1990

# The past window over which to normalise the forecast_var. Doesn't necessarily need to
# be the same as the window length over which the analogues were made
import sys
import numpy as np
print len(sys.argv)
if len(sys.argv) == 4:
    norm_window = np.long(sys.argv[1])
    divide_by_sd_in = sys.argv[2]
    chosen_target_domain = sys.argv[3]

    chosen_window = 35
    chosen_num_mems = 100
    pass_string = ''
    chosen_smoothing = 1
    rmse_method = True
    method = 'CleverD' ####################################################################################
elif len(sys.argv) == 5:
    infile = sys.argv[1]
    norm_window = np.long(sys.argv[2])
    divide_by_sd_in = sys.argv[3]
    method = sys.argv[4]

    split_file = infile.split('_')
    print split_file
    chosen_target_domain = split_file[2][6:]
    if split_file[4] in ['gyre', 'atlantic']:  # For regions including an underscore, remove the second part
        split_file.pop(4)
    chosen_window = np.long(split_file[4][6:])
    chosen_num_mems = np.long(split_file[5][4:])
    if split_file[7][:3] == 'Smo':
        chosen_smoothing = np.long(split_file[6][3:])
        split_file.pop(7)
    else:
        chosen_smoothing = 1
    rmse_method = False
    if len(split_file) == 8:
        if split_file[7] == 'RMSEmethod.pkl':
            rmse_method = True
    pass_string = ''

if chosen_num_mems not in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
    raise ValueError('Will not bother with chosen_num_mems={:d}'.format(chosen_num_mems))
if chosen_smoothing not in [1, 21]:
    raise ValueError('Will not bother with chosen_smoothing={:d}'.format(chosen_smoothing))

if divide_by_sd_in == 'True':
    divide_by_sd = True
elif divide_by_sd_in == 'False':
    divide_by_sd = False

if do_random:
    only_do_random = False

if only_do_random and not do_random:
    raise ValueError('You''ve set only_do_random but also unset do_random')

if method[:7] == "CleverD":
    cleverd_norm_window = 5

import pickle
import numpy as np
import os
from scipy import interpolate
from scipy import stats
from analogue import *

def read_target_domain(in_string):
    out_list = []
    for ii in range(4):
        out_list.append(np.int(in_string[ii*3:(ii+1)*3]))
    return out_list

def check_files(path, files, output_prefix):
    # Processed files
    all_files_exist = True
    unique_files = sorted(list(set(files.flatten().data.tolist())))[1:]
    unique_files_newlines = unique_files[:]
    for ifile, this_file in enumerate(unique_files):
        unique_files_newlines[ifile] = os.path.basename(this_file) + '\n'
        if not os.path.isfile(os.path.join(path, os.path.basename(this_file))):
            all_files_exist = False

    if all_files_exist:
        print " == All files have already been copied to: {:s}".format(path)

    out_list = '{:s}/{:s}_{:s}.txt'.format(text_file_dir, output_prefix, base)
    with open(out_list, 'wb') as handle:
        if not all_files_exist:
            print "Writing text file to:\n   {:s}".format(out_list)
        handle.writelines(unique_files_newlines)
    return all_files_exist

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
#                 print iyr, year
                if ((year-1) in year_in) and ((year+1) in year_in):
                    # If data either side exists then interpolate
                    print " ++ INTERPOLATING MISSING DATA"
                    iyr_in_m1 = np.argwhere(year_in == (year-1))[0][0]
                    iyr_in_p1 = np.argwhere(year_in == (year+1))[0][0]
                    sst_model[iyr, :, :] = (sst_in[iyr_in_m1, :, :] + sst_in[iyr_in_p1, :, :]) / 2.
    else:
        sst_model = sst_in
        year_model = year_in
    return sst_model, year_model

nesw = read_target_domain(chosen_target_domain)

if chosen_smoothing > 1:
    smo_len = chosen_smoothing
    smoothing_string = '_Smo{:d}'.format(smo_len)
else:
    smoothing_string = ''

if rmse_method:
    rmse_string = '_RMSEmethod'
else:
    rmse_string = ''

assert trends or annual
if trends:
    trends_or_annual = ''
elif annual:
    trends_or_annual = '_ANNUAL'

if not divide_by_sd:
    sd_string = '_NOSD'
else:
    sd_string = ''

if do_final_sd_scaling:
    scaling_string = ''
else:
    scaling_string = 'NoFinalSDScale'

max_mems_to_take = 20
if chosen_num_mems > max_mems_to_take:
    max_mems_to_take = chosen_num_mems
nlead = 11
lead_times = np.arange(nlead)

myhost = os.uname()[1]
print("myhost = {:s}".format(myhost))
usr = os.environ["USER"]
print("{.s}".format(usr))

if 'ciclad' in myhost:
    raise ValueError("deprecated")
    scripts_dir = '/home/mmenary/python/scripts/'
    processed_output_dir = '/modfs/ipslfs/dods/mmenary/AnalogueCache'
    map_output_dir = '/modfs/ipslfs/dods/mmenary/AnalogueCache'
    raw_analogue_output_dir = '/data/mmenary/python_saves/CMIP_{:s}/'.format(analogue_var)
    raw_forecast_output_dir = '/data/mmenary/python_saves/CMIP_{:s}/'.format(forecast_var)
    hadisst_save_file = '/data/mmenary/python_saves/HadISST_AnnualMapCMIPStyleRegridded.pkl'
    en4_save_file = '/data/mmenary/python_saves/EN4_0-500m_AnnualMapCMIPStyleRegridded.pkl'
    hadcrut4_save_file = '/data/mmenary/python_saves/HadCRUT4_AnnualMapCMIPStyleRegridded.pkl'
    hadcrut4_save_file_djfm = '/data/mmenary/python_saves/HadCRUT4_DJFMMapCMIPStyleRegridded.pkl'
    text_file_dir = '/data/mmenary/text_files'
    hist_map_dir = '/data/mmenary/python_saves'
else:
    scripts_dir = '/home/users/{:s}/scripts/'.format(usr)
    processed_output_dir = '/work/scratch-nopw/{:s}/AnalogueCache'.format(usr)
    map_output_dir = '/work/scratch-nopw/{:s}/SkillMaps'.format(usr)
    raw_analogue_output_dir = '/work/scratch-nopw/{:s}/CMIP_{:s}/'.format(usr, analogue_var)
    raw_forecast_output_dir = '/work/scratch-nopw/{:s}/CMIP_{:s}/'.format(usr, forecast_var)
    hadisst_save_file = '/home/users/{:s}/data/HadISST_AnnualMapCMIPStyleRegridded.pkl'.format(usr)
    en4_save_file = '/home/users/{:s}/data/EN4_0-500m_AnnualMapCMIPStyleRegridded.pkl'.format(usr)
    hadcrut4_save_file = '/home/users/{:s}/data/HadCRUT4_AnnualMapCMIPStyleRegridded.pkl'.format(usr)
    hadcrut4_save_file_djfm = '/home/users/{:s}/data/HadCRUT4_DJFMMapCMIPStyleRegridded.pkl'.format(usr)
    text_file_dir = '/home/users/{:s}/text_files'.format(usr)
    hist_map_dir = '/home/users/{:s}/data/python_saves'.format(usr)

if (clim_start != 1960) or (clim_end != 1990):
    processed_output_dir += '_{:d}-{:d}'.format(clim_start, clim_end)
    map_output_dir += '_{:d}-{:d}'.format(clim_start, clim_end)

# Read the skill data
skill_base = 'ANALOGUE{:s}_FORECAST{:s}_DOMAIN{:s}_TARGET{:s}_WINDOW{:d}_MEMS{:d}{:s}_SpatialSkill{:s}{:s}{:s}.pkl'
skill_base = skill_base.format(analogue_var, forecast_var, chosen_target_domain, chosen_target_region, chosen_window, max_mems_to_take,
                               smoothing_string, '', '', rmse_string)
skill_file = os.path.join(processed_output_dir, skill_base)
print 'skill_file', skill_file

base_template = 'ANALOGUE{:s}_DOMAIN{:s}_TARGET{:s}_WINDOW{:d}_MEMS{:d}_SpatialSkill{:s}{:s}{:s}'
base = base_template.format(analogue_var, chosen_target_domain, chosen_target_region, chosen_window, chosen_num_mems, smoothing_string, pass_string, rmse_string)

processed_template = '{:s}_{:s}_'.format(analogue_var, chosen_target_domain) + '{:s}_{:s}-{:s}' + '_Window{:d}{:s}_SpatialProcessed{:s}{:s}.pkl'.format(chosen_window, smoothing_string, '', rmse_string)
print 'processed_template', processed_template

raw_analogue_template = '{:s}' + '_{:s}_'.format(analogue_var) + '{:s}_{:s}' + '_Annual_Regridded.pkl'
print 'raw_analogue_template', raw_analogue_template

raw_forecast_template = '{:s}' + '_{:s}_'.format(forecast_var) + '{:s}_{:s}' + '_Annual_Regridded.pkl'
print 'raw_forecast_template', raw_forecast_template

trends_template = '{:s}_SavedTrends_{:s}_'.format(analogue_var, chosen_target_domain) + '{:s}_{:s}-{:s}' + '_Window{:d}{:s}_SpatialProcessed{:s}{:s}.pkl'.format(chosen_window, smoothing_string, '', rmse_string)
print 'trends_template', trends_template

source_file = os.path.join(processed_output_dir, 'Source' + trends_or_annual + '_' + base + '.pkl')
print 'source_file', source_file
#
# forecast_file = os.path.join(processed_output_dir, 'ForecastMaps' + trends_or_annual + '_' + base + '.pkl')
forecast_file = os.path.join(processed_output_dir, 'ForecastMaps' + trends_or_annual + '_' + base + '.pkl.{:s}Method{:d}'.format(method, norm_window) + sd_string + scaling_string)
print 'forecast_file', forecast_file

if clever_skill:
    clever_skill_string = 'Clever'
else:
    clever_skill_string = ''
# skill_map_file = os.path.join(processed_output_dir, '{:s}SkillMaps'.format(clever_skill_string) + trends_or_annual + '_' + base + '.pkl')
skill_map_file = os.path.join(map_output_dir, '{:s}SkillMaps'.format(clever_skill_string) + trends_or_annual + '_' + base + '.pkl.{:s}Method{:d}'.format(method, norm_window) + sd_string + scaling_string)
if testing2:
    skill_map_file += '.TESTING2'
print 'skill_map_file', skill_map_file

if os.path.isfile(skill_map_file) and not recreate_skill:
    print "Skipping - already created: {:s}".format(skill_map_file)

# forecast_file_ensmn_expanded = os.path.join(processed_output_dir, 'ForecastMaps_EnsMnExpanded' + trends_or_annual + '_' + base + '.pkl')
forecast_file_ensmn_expanded = os.path.join(processed_output_dir, 'ForecastMaps_EnsMnExpanded' + trends_or_annual + '_' + base + '.pkl.{:s}Method{:d}'.format(method, norm_window) + sd_string + scaling_string)
forecast_file_ensmn_expanded_random = os.path.join(processed_output_dir, 'ForecastMapsRandom_EnsMnExpanded' + trends_or_annual + '_' + base + '.pkl.{:s}Method{:d}'.format(method, norm_window) + sd_string + scaling_string)
if testing:
    forecast_file_ensmn_expanded += '.TEST'
    forecast_file_ensmn_expanded_random += '.TEST'
print 'forecast_file_ensmn_expanded', forecast_file_ensmn_expanded
print 'forecast_file_ensmn_expanded_random', forecast_file_ensmn_expanded_random

if not os.path.isfile(skill_file):
    raise ValueError("No save file: {:s}".format(skill_file))

with open(skill_file, 'rb') as handle:
    print "Reading: {:s}".format(skill_file)
    if trends:
        print "Doing TRENDS version"
        _, _, _, _, trend_corr_info, trend_forecast, trend_forecast_means, trend_forecast_sds = pickle.load(handle)
    elif annual:
        print "Doing ANNUAL version"
        trend_corr_info, trend_forecast, trend_forecast_means, trend_forecast_sds, corr_info_mask, _, _, _ = pickle.load(handle)
        # Because I haven't masked ann_corr_info (which this really is) properly
        ind = np.argwhere(corr_info_mask.mask[:, 0, 0] == True)  # Find where masked
        if len(ind) > 0:  # If masked, then copy this mask
            trend_corr_info.mask[:ind[-1][0]+1, :, :] = corr_info_mask.mask[:ind[-1][0]+1, :, :].copy()

if forecast_var == 'SST':
    forecast_save_file = hadisst_save_file
elif forecast_var == 'DepthAverageT':
    forecast_save_file = en4_save_file
elif forecast_var == 'SAT':
    forecast_save_file = hadcrut4_save_file
with open(forecast_save_file, 'rb') as handle:
    print "Loading save file: {:s}".format(forecast_save_file)
    obs_map_expanded, _, _, _, _, year_forecast_obs = pickle.load(handle)
    nyrs_forecast_obs = len(year_forecast_obs)

if analogue_var == 'SST':
    analogue_save_file = hadisst_save_file
elif analogue_var == 'DepthAverageT':
    analogue_save_file = en4_save_file
elif analogue_var == 'SAT':
    analogue_save_file = hadcrut4_save_file
with open(analogue_save_file, 'rb') as handle:
    print "Loading save file: {:s}".format(analogue_save_file)
    analogue_obs_map_expanded, _, _, _, _, year_analogue_obs = pickle.load(handle)
    nyrs_analogue_obs = len(year_analogue_obs)

# Make climatology for later
t0 = np.argwhere(year_analogue_obs == clim_start)[0][0]
t1 = np.argwhere(year_analogue_obs == clim_end)[0][0]
obs_clim = np.ma.mean(analogue_obs_map_expanded[t0:t1, :, :], axis=0)

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

processed_files = np.ma.masked_all(shape=(nyrs_forecast_obs, chosen_num_mems), dtype=object)
trends_files = np.ma.masked_all(shape=(nyrs_forecast_obs, chosen_num_mems), dtype=object)
raw_analogue_files = np.ma.masked_all(shape=(nyrs_forecast_obs, chosen_num_mems), dtype=object)
raw_forecast_files = np.ma.masked_all(shape=(nyrs_forecast_obs, chosen_num_mems), dtype=object)

for iyr_forecast, year in enumerate(year_forecast_obs):
    for imem in range(chosen_num_mems):
        if np.ma.is_masked(trend_corr_info[iyr_forecast, imem, :]):
            continue
        model = trend_corr_info[iyr_forecast, imem, 1]
        expt = trend_corr_info[iyr_forecast, imem, 2]
        ens_mem = trend_corr_info[iyr_forecast, imem, 3]
        index = trend_corr_info[iyr_forecast, imem, 4]
        processed_files[iyr_forecast, imem] = processed_template.format(model, expt, ens_mem)
        trends_files[iyr_forecast, imem] = trends_template.format(model, expt, ens_mem)
        if expt == 'piControl':
            expt_ens_mem = expt
        else:
            expt_ens_mem = expt + '-' + ens_mem
        if model in cmip5_models:
            project = 'CMIP5'
        elif model in cmip6_models:
            project = 'CMIP6'
        else:
            raise ValueError('Unknown model')
        raw_analogue_files[iyr_forecast, imem] = raw_analogue_template.format(project, model, expt_ens_mem)
        raw_forecast_files[iyr_forecast, imem] = raw_forecast_template.format(project, model, expt_ens_mem)

all1 = check_files(processed_output_dir, processed_files, 'InputFilesList'+trends_or_annual)
all2 = check_files(raw_analogue_output_dir, raw_analogue_files, 'InputFilesList2'+trends_or_annual)
# all3 = check_files(processed_output_dir, trends_files, 'InputFilesList3'+trends_or_annual)
all4 = check_files(raw_forecast_output_dir, raw_forecast_files, 'InputFilesList4'+trends_or_annual)

if not all1:
    print "\n\n++ Missing SpatialProcessed files ++"
if not all2:
    # Not actually used currently
    print "\n\n++ Missing Raw Analogue ({:s}) files ++".format(analogue_var)
# if not all3:
#     print "\n\n++ Missing SpatialProcessed TRENDS files ++"
if not all4:
    print "\n\n++ Missing Raw Forecast ({:s}) files ++".format(forecast_var)

if not all1:
    raise ValueError('The SpatialProcessed files are required.')

# Surprisingly complicated. Make a shuffled set of the files/corr info so I can test whether the skill
# I get is just due to the method of creating the forecasts, rather than the actual input data
first_year_index = chosen_window + (nyrs_forecast_obs - nyrs_analogue_obs) - 1
nyrs = nyrs_forecast_obs - first_year_index

rand_ind = np.arange(nyrs * chosen_num_mems)
np.random.shuffle(rand_ind)

raw_forecast_files_shuffled = raw_forecast_files.copy()
processed_files_shuffled = processed_files.copy()
trend_corr_info_shuffled = trend_corr_info.copy()

iyears = np.repeat(np.arange(nyrs) + first_year_index, chosen_num_mems)
imems = np.repeat(np.arange(chosen_num_mems)[np.newaxis, :], nyrs, axis=0).flatten()

count = 0
for iyr, year in enumerate(year_forecast_obs):
    if np.ma.is_masked(raw_forecast_files[iyr, 0]):
        continue
    for imem in range(chosen_num_mems):
        iyr2 = iyears[rand_ind[count]]
        imem2 = imems[rand_ind[count]]
        processed_files_shuffled[iyr, imem] = processed_files[iyr2, imem2]
        raw_forecast_files_shuffled[iyr, imem] = raw_forecast_files[iyr2, imem2]
        trend_corr_info_shuffled[iyr, imem, :] = trend_corr_info[iyr2, imem2, :].copy()
        count += 1

list1 = list(set(sorted(processed_files[first_year_index:, :].flatten())))
list2 = list(set(sorted(processed_files_shuffled[first_year_index:, :].flatten())))
assert len(list1) == len(list2)
for a, b in zip(list1, list2):
    assert a == b

list1 = list(set(sorted(raw_forecast_files[first_year_index:, :].flatten())))
list2 = list(set(sorted(raw_forecast_files_shuffled[first_year_index:, :].flatten())))
assert len(list1) == len(list2)
for a, b in zip(list1, list2):
    assert a == b

for iyr, year in enumerate(year_forecast_obs):
    if np.ma.is_masked(raw_forecast_files_shuffled[iyr, 0]):
        continue
    for imem in range(chosen_num_mems):
        raw_file = raw_forecast_files_shuffled[iyr, imem]
        model, expt, ens_mem = trend_corr_info_shuffled[iyr, imem, 1:4]
        assert model in raw_file
        assert expt in raw_file
        if expt != 'piControl':
            assert ens_mem in raw_file

def find_climatology_for_model(model):
    climatology_file = os.path.join(raw_analogue_output_dir, 'CMIP_{:s}_{:s}_historical-EnsMn_TM{:d}-{:d}_Annual.pkl'.format(analogue_var, model, clim_start, clim_end))
    if os.path.isfile(climatology_file):
        with open(climatology_file, 'rb') as handle:
            sst_clim = pickle.load(handle)
    else:
        print "No climatology file exists, filling with missing data: {:s}".format(climatology_file)
        sst_clim = np.ma.masked_all(shape=(nj, ni))
    return sst_clim

# Full size
nj = 180
ni = 360
lon_re_full = np.repeat((np.arange(-180, 180) + 0.5)[np.newaxis, :], nj, axis=0)
lat_re_full = np.repeat((np.arange(-90, 90) + 0.5)[:, None], ni, axis=1)

# Coordinates for reduced-size analogue trends maps
ii0 = 70
ii1 = 210
jj0 = 80
jj1 = 171
nj_sub = jj1 - jj0
ni_sub = ii1 - ii0

lon_re = lon_re_full[jj0:jj1, ii0:ii1]
lat_re = lat_re_full[jj0:jj1, ii0:ii1]

# Coordinates for reduced-size forecast maps
ii0b = 110
ii1b = 250
jj0b = 120
jj1b = 160
nj_sub2 = jj1b - jj0b
ni_sub2 = ii1b - ii0b

lon_re2 = lon_re_full[jj0b:jj1b, ii0b:ii1b]
lat_re2 = lat_re_full[jj0b:jj1b, ii0b:ii1b]

tol = 1e-2  # For when I check that the correlations are the same

if testing:
    # Overwrite this after having used it to create filenames above. Hopefully this way when we run in interactive
    # mode we won't run out of memory and crash
    chosen_num_mems = 2

# Make and store the analogue source (trend) maps
if os.path.isfile(source_file) and os.path.isfile(forecast_file) and os.path.isfile(forecast_file_ensmn_expanded) and os.path.isfile(forecast_file_ensmn_expanded_random) and not testing:
    print "READING:\n  {:s}\n  {:s}\n  {:s}".format(source_file, forecast_file, forecast_file_ensmn_expanded)
    with open(source_file, 'rb') as handle:
        obs_trend, analogue_trend = pickle.load(handle)
    with open(forecast_file, 'rb') as handle:
        obs_map, forecast_map = pickle.load(handle)
    with open(forecast_file_ensmn_expanded, 'rb') as handle:
        forecast_map_ensmn_stdsd_expanded = pickle.load(handle)
    with open(forecast_file_ensmn_expanded_random, 'rb') as handle:
        forecast_map_ensmn_stdsd_expanded_random = pickle.load(handle)
    forecast_map_ensmn_expanded = forecast_map_ensmn_expanded_random = 0
elif os.path.isfile(forecast_file_ensmn_expanded) and os.path.isfile(forecast_file_ensmn_expanded_random) and skill_only:
    with open(forecast_file_ensmn_expanded, 'rb') as handle:
        forecast_map_ensmn_stdsd_expanded = pickle.load(handle)
    with open(forecast_file_ensmn_expanded_random, 'rb') as handle:
        forecast_map_ensmn_stdsd_expanded_random = pickle.load(handle)
    forecast_map_ensmn_expanded = forecast_map_ensmn_expanded_random = 0
else:
    # if all3:
    #     analogue_trend = np.ma.masked_all(shape=(nyrs_forecast_obs, chosen_num_mems, nj_sub, ni_sub))
    if all4:
        obs_map = obs_map_expanded[:, jj0b:jj1b, ii0b:ii1b].copy()
        forecast_map = 0
        forecast_map_ensmn_expanded = 0
        if not only_do_random:
            forecast_map_ensmn_stdsd_expanded = np.ma.masked_all(shape=(nyrs_forecast_obs, nlead, nj, ni))
            if method[:7] == "CleverD":
                sat_in_MeanInCleverDWindow_reshaped = np.ma.masked_all(shape=(nyrs_forecast_obs, nj, ni))
                obs_MeanInCleverDWindow = np.ma.masked_all(shape=(nyrs_forecast_obs, nj, ni))
        forecast_map_ensmn_expanded_random = 0
        if do_random:
            forecast_map_ensmn_stdsd_expanded_random = np.ma.masked_all(shape=(nyrs_forecast_obs, nlead, nj, ni))

        if method[:7] == 'CleverB':
            if not only_do_random:
                mns_saved = np.ma.masked_all(shape=(nyrs_forecast_obs, nj, ni))
            if do_random:
                mns_saved_random = np.ma.masked_all(shape=(nyrs_forecast_obs, nj, ni))

    first_trends_file = True
    for iyr_forecast, year in enumerate(year_forecast_obs):
        print iyr_forecast, year
        if testing:
            if iyr_forecast > 50: continue
#         if iyr_forecast != 34: continue ###########################################################
        if trend_corr_info[iyr_forecast, :, :].count() == 0:
            continue
        if not only_do_random:
            sat_in_full = np.ma.masked_all(shape=(chosen_num_mems, nlead, nj, ni))
            sat_in_MeanInWindow = np.ma.masked_all(shape=(chosen_num_mems, nj, ni))
            if method[:7] == "CleverD":
                sat_in_MeanInCleverDWindow = np.ma.masked_all(shape=(chosen_num_mems, nj, ni))
            sat_in_SDInWindow = np.ma.masked_all(shape=(chosen_num_mems, nj, ni))
        if do_random:
            sat_in_full_random = np.ma.masked_all(shape=(chosen_num_mems, nlead, nj, ni))
            sat_in_MeanInWindow_random = np.ma.masked_all(shape=(chosen_num_mems, nj, ni))
            sat_in_SDInWindow_random = np.ma.masked_all(shape=(chosen_num_mems, nj, ni))
        for imem in range(chosen_num_mems):
            if np.ma.is_masked(trend_corr_info[iyr_forecast, imem, :]):
                continue
            this_processed_file = os.path.join(processed_output_dir, processed_files[iyr_forecast, imem])
            this_processed_file_random = os.path.join(processed_output_dir, processed_files_shuffled[iyr_forecast, imem])
            # if all3: this_trends_file = os.path.join(processed_output_dir, trends_files[iyr_forecast, imem])
            if all4:
                this_forecast_file = os.path.join(raw_forecast_output_dir, raw_forecast_files[iyr_forecast, imem])
                this_forecast_file_random = os.path.join(raw_forecast_output_dir, raw_forecast_files_shuffled[iyr_forecast, imem])

            if all2:
                this_analogue_file = os.path.join(raw_analogue_output_dir, raw_analogue_files[iyr_forecast, imem])
            else:
                raise ValueError("all2 shouldn't be false!?")

            # Analogue info
            index = trend_corr_info[iyr_forecast, imem, 4]
            index_random = trend_corr_info_shuffled[iyr_forecast, imem, 4]

            # For verification. This is already processed so shouldn't require padding again
            with open(this_processed_file, 'rb') as handle:
                if trends:
                    _, corr_trend, _, year_analogue_model = pickle.load(handle)
                elif annual:
                    corr_trend, _, _, year_analogue_model = pickle.load(handle)
            with open(this_processed_file_random, 'rb') as handle:
                _, _, _, year_analogue_model_random = pickle.load(handle)

            # Convert obs years between forecast var and analogue var so we can check that we're
            # reading in the correct processed_file for this Skill file (trend_corr_info)
            iyr_analogue = np.argwhere(year_analogue_obs == year_forecast_obs[iyr_forecast])[0][0]
            if rmse_method:
                assert np.abs(trend_corr_info[iyr_forecast, imem, 0] - corr_trend[iyr_analogue, index]) / np.abs(trend_corr_info[iyr_forecast, imem, 0]) < tol  # should be same
            else:
                assert np.abs(trend_corr_info[iyr_forecast, imem, 0] - corr_trend[iyr_analogue, index]) < tol  # should be same

            # # If the ASSERT works, then these have the same validity times as "trends_file"s are created
            # # in the same place as the processed_files
            # if all3:
            #     with open(this_trends_file, 'rb') as handle:
            #         data = pickle.load(handle)
            #         print len(data)
            #         target_masked_trend, sst_masked_trend, target_masked_mean_anom, sst_masked_mean_anom, year_ann, year_model = data
            #         if first_trends_file:
            #             obs_trend = target_masked_trend[:, jj0:jj1, ii0:ii1]
            #             first_trends_file = False
            #         analogue_trend[iyr_forecast, imem, :, :] = sst_masked_trend[index, jj0:jj1, ii0:ii1]

            if all4 and not only_do_random:
                # Now get the forecast data - matching filename
                with open(this_forecast_file, 'rb') as handle:
                    print this_forecast_file
                    sat_in, year_forecast_model  = pickle.load(handle)
                if len(year_forecast_model) == 0:
                    print " ++ Skipping (zero length time array): {:s}".format(this_forecast_file)
                    continue
                sat_in, year_forecast_model = check_and_pad(sat_in, year_forecast_model)  # These are "raw" so unpadded.

                # if analogue_var == forecast_var:
                #     analogue_in, year_analogue_model = sat_in, year_forecast_model
                # else:
                #     raise ValueError("Haven't coded this yet")

                # Check that the MODEL years in the analogue (e.g. SST) and forecast (e.g. SAT) can be translated
                chosen_year = year_analogue_model[index]
                if chosen_year in year_forecast_model:
                    index2 = np.argwhere(year_forecast_model == chosen_year)[0][0]
                    if index != index2:
                        print " -- Changing index ({:d}): {:d} -> {:d} Y{:d}   {:s}".format(imem, index, index2, chosen_year, this_forecast_file)
                else:
                    print " ++ Skipping (year does not exist): {:s}".format(this_forecast_file)
                    continue

                ntimes = np.min([nlead, len(year_forecast_model[index2:])])
                if ntimes < 11:
                    print " ++ Unfull forecast times, nlead={:d}, ntimes={:d}".format(nlead, ntimes)
                    print index2, this_forecast_file
                # forecast_map[iyr_forecast, imem, :ntimes, :, :] = sat_in[index2:index2+ntimes, jj0b:jj1b, ii0b:ii1b]
                sat_in_full[imem, :ntimes, :, :] = sat_in[index2:index2+ntimes, :, :]
                sat_in_MeanInWindow[imem, :, :] = np.ma.mean(sat_in[index2-(norm_window-1):index2+1, :, :], axis=0)
                if method[:7] == "CleverD":
                    sat_in_MeanInCleverDWindow[imem, :, :] = np.ma.mean(sat_in[index2-(cleverd_norm_window-1):index2+1, :, :], axis=0)
                sat_in_SDInWindow[imem, :, :] = np.ma.std(sat_in[index2-(norm_window-1):index2+1, :, :], axis=0)
                if method[7:] == 'Clim':
                    model_clim = find_climatology_for_model(trend_corr_info[iyr_forecast, imem, 1])
                    sat_in_MeanInWindow[imem, :, :] += (obs_clim - model_clim)  # As in analogue construction
                sat_in = 0

            # As above but for the random/shuffled data
            if all4 and do_random:
                # Now get the forecast data - matching filename
                with open(this_forecast_file_random, 'rb') as handle:
                    print this_forecast_file
                    sat_in, year_forecast_model  = pickle.load(handle)
                if len(year_forecast_model) == 0:
                    print " ++ Skipping (zero length time array): {:s}".format(this_forecast_file)
                    continue
                sat_in, year_forecast_model = check_and_pad(sat_in, year_forecast_model)  # These are "raw" so unpadded.

                # Check that the MODEL years in the analogue (e.g. SST) and forecast (e.g. SAT) can be translated
                chosen_year = year_analogue_model_random[index_random]
                if chosen_year in year_forecast_model:
                    index2 = np.argwhere(year_forecast_model == chosen_year)[0][0]
                    if index_random != index2:
                        print " -- Changing index ({:d}): {:d} -> {:d} Y{:d}   {:s}".format(imem, index_random, index2, chosen_year, this_forecast_file_random)
                else:
                    print " ++ Skipping (year does not exist): {:s}".format(this_forecast_file_random)
                    continue

                ntimes = np.min([nlead, len(year_forecast_model[index2:])])
                if ntimes < 11:
                    print " ++ Unfull forecast times, nlead={:d}, ntimes={:d}".format(nlead, ntimes)
                    print index2, this_forecast_file_random
                sat_in_full_random[imem, :ntimes, :, :] = sat_in[index2:index2+ntimes, :, :]
                sat_in_MeanInWindow_random[imem, :, :] = np.ma.mean(sat_in[index2-(norm_window-1):index2+1, :, :], axis=0)
                sat_in_SDInWindow_random[imem, :, :] = np.ma.std(sat_in[index2-(norm_window-1):index2+1, :, :], axis=0)
                sat_in = 0

        if all4:
            # After all members for this year have been read, average them together and save the full size map
            # forecast_map_ensmn_expanded[iyr_forecast, :, :, :] = np.ma.mean(sat_in_full, axis=0)
            # forecast_map_ensmn_expanded_random[iyr_forecast, :, :, :] = np.ma.mean(sat_in_full_random, axis=0)

            # And a cleverer way that first standardises the input but this results in the input
            # analogue info not being that important in the final "skill" outcome. This could be
            # tested by using random inputs to the analogue and seeing how skilful or not it is
            obs_sd = np.ma.std(obs_map_expanded[(iyr_forecast+1)-norm_window:iyr_forecast+1, :, :], axis=0)  # j, i
            obs_sd = np.repeat(np.repeat(obs_sd[np.newaxis, np.newaxis, :, :], chosen_num_mems, axis=0), nlead, axis=1)
            obs_mn = np.ma.mean(obs_map_expanded[(iyr_forecast+1)-norm_window:iyr_forecast+1, :, :], axis=0)  # j, i
            obs_mn = np.repeat(np.repeat(obs_mn[np.newaxis, np.newaxis, :, :], chosen_num_mems, axis=0), nlead, axis=1)

            if method[:7] == "CleverD":
                sat_in_MeanInCleverDWindow_reshaped[iyr_forecast, :, :] = np.ma.mean(sat_in_MeanInCleverDWindow, axis=0)  # MMM (yr, j, i)
                obs_MeanInCleverDWindow[iyr_forecast, :, :] = np.ma.mean(obs_map_expanded[(iyr_forecast+1)-cleverd_norm_window:iyr_forecast+1, :, :], axis=0)  # yr j, i

            # The question is, why do these end up being less skilful than the area-averaged versions??
            # This is actually normalising by FUTURE forecast, not using the means/sds I saved, which is bad.
            # Have to use a map though, so would need to read in SST in prior window...
            # TODO: Try instead by removing the 60-90 clim as well as using the s.d. of that clim period
            if not only_do_random:
                mns = np.repeat(sat_in_MeanInWindow[:, np.newaxis, :, :], nlead, axis=1)  #mem, leads, j, i
                if divide_by_sd:
                    sds = np.repeat(sat_in_SDInWindow[:, np.newaxis, :, :], nlead, axis=1)  #mem, leads, j, i
                    sat_in_full_norm = (sat_in_full - mns) * (obs_sd / sds) + obs_mn
                else:
                    sat_in_full_norm = (sat_in_full - mns) + obs_mn
                forecast_map_ensmn_stdsd_expanded[iyr_forecast, :, :, :] = np.ma.mean(sat_in_full_norm, axis=0)

            if do_random:
                mns_random = np.repeat(sat_in_MeanInWindow_random[:, np.newaxis, :, :], nlead, axis=1)  #mem, leads, j, i
                if divide_by_sd:
                    sds_random = np.repeat(sat_in_SDInWindow_random[:, np.newaxis, :, :], nlead, axis=1)  #mem, leads, j, i
                    sat_in_full_norm = (sat_in_full_random - mns_random) * (obs_sd / sds_random) + obs_mn
                else:
                    sat_in_full_norm = (sat_in_full_random - mns_random) + obs_mn
                forecast_map_ensmn_stdsd_expanded_random[iyr_forecast, :, :, :] = np.ma.mean(sat_in_full_norm, axis=0)  #yr, leads, j, i

        if method[:7] == 'CleverB':
            if not only_do_random:
                mns_saved[iyr_forecast, :, :] = np.ma.mean(mns[:, 0, :, :], axis=0)  # yr, j, i
            if do_random:
                mns_saved_random[iyr_forecast, :, :] = np.ma.mean(mns_random[:, 0, :, :], axis=0)

    if testing:
        print forecast_map_ensmn_stdsd_expanded[:, 0, 150, 150]
        with open('/work/scratch-nopw/mmenary/DEL.pkl', 'wb') as handle:
            pickle.dump(forecast_map_ensmn_stdsd_expanded, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if method[:6] == 'Clever':
        # Find the mn/sd for each lead time at each grid point
        if not only_do_random:
            print "Calculating LTD SD"
            forecast_sd = np.ma.std(forecast_map_ensmn_stdsd_expanded, axis=1)[:, np.newaxis, :, :]
            print "Calculating LTD Mean"
            forecast_mn = np.ma.mean(forecast_map_ensmn_stdsd_expanded, axis=1)[:, np.newaxis, :, :]

        if do_random:
            print "Calculating LTD SD for random"
            forecast_sd_random = np.ma.std(forecast_map_ensmn_stdsd_expanded_random, axis=1)[:, np.newaxis, :, :]
            print "Calculating LTD Mean for random"
            forecast_mn_random = np.ma.mean(forecast_map_ensmn_stdsd_expanded_random, axis=1)[:, np.newaxis, :, :]

        print "Calculating Obs SD and Mean in windows and reshaping"
        obs_map_mn_reshaped = np.ma.masked_all_like(forecast_mn)
        obs_map_sd_reshaped = np.ma.masked_all_like(forecast_mn)
        nyrs = obs_map_mn_reshaped.shape[0]
        for iyr_o in range(norm_window-1, nyrs):
            obs_map_mn_reshaped[iyr_o, :, :, :] = np.ma.mean(obs_map_expanded[iyr_o-norm_window+1:iyr_o+1, :, :], axis=0)
            obs_map_sd_reshaped[iyr_o, :, :, :] = np.ma.std(obs_map_expanded[iyr_o-norm_window+1:iyr_o+1, :, :], axis=0)

        if not only_do_random:
            print "Combining..."
            if method[:7] == 'CleverB':
                forecast_map_ensmn_stdsd_expanded = (forecast_map_ensmn_stdsd_expanded - forecast_mn) * (obs_map_sd_reshaped / forecast_sd) + mns_saved[:, np.newaxis, :, :]
            elif method[:7] in ['CleverC', 'CleverD']:
                if do_final_sd_scaling:
                    forecast_map_ensmn_stdsd_expanded = (forecast_map_ensmn_stdsd_expanded - forecast_mn) * (obs_map_sd_reshaped / forecast_sd) + forecast_mn
                if method[:7] == 'CleverC':
                    forecast_map_ensmn_stdsd_expanded = forecast_map_ensmn_stdsd_expanded + (obs_map_expanded - forecast_map_ensmn_stdsd_expanded[:, 0, :, :])[:, np.newaxis, :, :]
                elif method[:7] == 'CleverD':
                    # obs_map_expanded and sat_in_MeanInCleverDWindow_reshaped have dimensions [yr, j, i], so we need to add lead-> [yr, lead, j, i]
                    forecast_map_ensmn_stdsd_expanded = forecast_map_ensmn_stdsd_expanded + (obs_MeanInCleverDWindow - sat_in_MeanInCleverDWindow_reshaped)[:, np.newaxis, :, :]

        if do_random:
            print "Combining random..."
            if method[:7] == 'CleverB':
                forecast_map_ensmn_stdsd_expanded_random = (forecast_map_ensmn_stdsd_expanded_random - forecast_mn_random) * (obs_map_sd_reshaped / forecast_sd_random) + mns_saved_random[:, np.newaxis, :, :]
            elif method[:7] == 'CleverC':
                forecast_map_ensmn_stdsd_expanded_random = (forecast_map_ensmn_stdsd_expanded_random - forecast_mn_random) * (obs_map_sd_reshaped / forecast_sd_random) + forecast_mn_random
                forecast_map_ensmn_stdsd_expanded_random = forecast_map_ensmn_stdsd_expanded_random + (obs_map_expanded - forecast_map_ensmn_stdsd_expanded_random[:, 0, :, :])[:, np.newaxis, :, :]

        if testing:
            print forecast_map_ensmn_stdsd_expanded[:, 0, 150, 150]
    # if not testing:
        # if all3:
        #     with open(source_file, 'wb') as handle:
        #         pickle.dump([obs_trend, analogue_trend], handle, protocol=pickle.HIGHEST_PROTOCOL)
    if all4:
        # with open(forecast_file, 'wb') as handle:
        #     pickle.dump([obs_map, forecast_map], handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(forecast_file_ensmn_expanded, 'wb') as handle:
        #     print "Writing save file: {:s}".format(forecast_file_ensmn_expanded)
        #     pickle.dump([obs_map_expanded, forecast_map_ensmn_expanded, forecast_map_ensmn_stdsd_expanded,
        #                 forecast_map_ensmn_expanded_random, forecast_map_ensmn_stdsd_expanded_random],
        #                 handle, protocol=pickle.HIGHEST_PROTOCOL)
        if not only_do_random:
            with open(forecast_file_ensmn_expanded, 'wb') as handle:
                print "Writing save file: {:s}".format(forecast_file_ensmn_expanded)
                pickle.dump(forecast_map_ensmn_stdsd_expanded, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if do_random:
            with open(forecast_file_ensmn_expanded_random, 'wb') as handle:
                print "Writing save file: {:s}".format(forecast_file_ensmn_expanded_random)
                pickle.dump(forecast_map_ensmn_stdsd_expanded_random, handle, protocol=pickle.HIGHEST_PROTOCOL)

print "DONE!"

if testing:
    raise ValueError("STOP HERE FOR NOW WHILE TESTING")

start_lead = [1, 2, 3, 4, 5, 6,  1,  2]  # For the multiannual skill
end_lead =   [5, 6, 7, 8, 9, 10, 10, 10]
nlead_multi = len(start_lead)

if os.path.isfile(skill_map_file) and not recreate_skill:
    print "Skipping - already created: {:s}".format(skill_map_file)
else:
    if clever_skill:
        this_forecast_map_ensmn_expanded = forecast_map_ensmn_stdsd_expanded
        this_random_forecast_map_ensmn_expanded = forecast_map_ensmn_stdsd_expanded_random
    else:
        this_forecast_map_ensmn_expanded = forecast_map_ensmn_expanded
        this_random_forecast_map_ensmn_expanded = forecast_map_ensmn_expanded_random

    # For doing residual skill
    hist_map_file = '{:s}/CMIP_{:s}_Historical_EnsMn_Annual_Regridded.pkl'.format(hist_map_dir, forecast_var)
    if os.path.isfile(hist_map_file):
        with open(hist_map_file, 'rb') as handle:
            historicals, hist_models = pickle.load(handle)

    print "Calculating residuals"
    if testing2:
        obs_map_expanded_res = obs_map_expanded.copy()
        this_forecast_map_ensmn_expanded_res = this_forecast_map_ensmn_expanded.copy()
        this_random_forecast_map_ensmn_expanded_res = this_random_forecast_map_ensmn_expanded.copy()
    else:
        obs_map_expanded_res = calculate_residual3d(obs_map_expanded, 1, historicals)
        this_forecast_map_ensmn_expanded_res = calculate_residual3d(this_forecast_map_ensmn_expanded, nlead, historicals)
        this_random_forecast_map_ensmn_expanded_res = calculate_residual3d(this_random_forecast_map_ensmn_expanded, nlead, historicals)

    print "Doing normal skill"
    skill_expanded = calculate_skill3d(this_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs, testing=testing2)
    skill_expanded1960 = calculate_skill3d(this_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs, testing=testing2, since1960=True)
    skill_expanded1990 = calculate_skill3d(this_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs, testing=testing2, before1990=True)

    skill_expanded_multi = calculate_skill3d(this_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs,
                                             testing=testing2, multi=True, start_lead=start_lead, end_lead=end_lead)
    skill_expanded1960_multi = calculate_skill3d(this_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs,
                                                 testing=testing2, since1960=True, multi=True, start_lead=start_lead, end_lead=end_lead)
    skill_expanded1990_multi = calculate_skill3d(this_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs,
                                                 testing=testing2, before1990=True, multi=True, start_lead=start_lead, end_lead=end_lead)

    print "Doing RANDOM skill"
    random_skill_expanded = calculate_skill3d(this_random_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs, testing=testing2)
    random_skill_expanded1960 = calculate_skill3d(this_random_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs, testing=testing2, since1960=True)
    random_skill_expanded1990 = calculate_skill3d(this_random_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs, testing=testing2, before1990=True)

    random_skill_expanded_multi = calculate_skill3d(this_random_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs,
                                                    testing=testing2, multi=True, start_lead=start_lead, end_lead=end_lead)
    random_skill_expanded1960_multi = calculate_skill3d(this_random_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs,
                                                        testing=testing2, since1960=True, multi=True, start_lead=start_lead, end_lead=end_lead)
    random_skill_expanded1990_multi = calculate_skill3d(this_random_forecast_map_ensmn_expanded, nlead, obs_map_expanded, year_forecast_obs,
                                                        testing=testing2, before1990=True, multi=True, start_lead=start_lead, end_lead=end_lead)

    print "Doing residual skill"
    res_skill_expanded = calculate_skill3d(this_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs, testing=testing2)
    res_skill_expanded1960 = calculate_skill3d(this_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs, testing=testing2, since1960=True)
    res_skill_expanded1990 = calculate_skill3d(this_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs, testing=testing2, before1990=True)

    res_skill_expanded_multi = calculate_skill3d(this_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs,
                                                 testing=testing2, multi=True, start_lead=start_lead, end_lead=end_lead)
    res_skill_expanded1960_multi = calculate_skill3d(this_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs,
                                                     testing=testing2, since1960=True, multi=True, start_lead=start_lead, end_lead=end_lead)
    res_skill_expanded1990_multi = calculate_skill3d(this_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs,
                                                     testing=testing2, before1990=True, multi=True, start_lead=start_lead, end_lead=end_lead)

    # Now all again for the RANDOM (shuffled) data
    print "Doing RANDOM residual skill"
    random_res_skill_expanded = calculate_skill3d(this_random_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs, testing=testing2)
    random_res_skill_expanded1960 = calculate_skill3d(this_random_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs, testing=testing2, since1960=True)
    random_res_skill_expanded1990 = calculate_skill3d(this_random_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs, testing=testing2, before1990=True)

    random_res_skill_expanded_multi = calculate_skill3d(this_random_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs,
                                                        testing=testing2, multi=True, start_lead=start_lead, end_lead=end_lead)
    random_res_skill_expanded1960_multi = calculate_skill3d(this_random_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs,
                                                            testing=testing2, since1960=True, multi=True, start_lead=start_lead, end_lead=end_lead)
    random_res_skill_expanded1990_multi = calculate_skill3d(this_random_forecast_map_ensmn_expanded_res, nlead, obs_map_expanded_res, year_forecast_obs,
                                                            testing=testing2, before1990=True, multi=True, start_lead=start_lead, end_lead=end_lead)

    # Attempting to make Extended Data Fig 2 from Doug's Nature paper:
    # https://www.nature.com/articles/s41586-020-2525-0.pdf
    # Not completely sure I understand what he has done though. This is a bit messy as I'm trying different ways to
    # recreate his figure
    start_dates = 1973 + np.arange(17)
    chosen_leads = [2, 9]

    print "Doing DJFM skill"
    with open(hadcrut4_save_file_djfm, 'rb') as handle:
        obs_map_expanded_djfm, _, _, _, _, _ = pickle.load(handle)

    # Assuming he means "average together everything with a validity time equal to each of those years"
    t0 = np.argwhere(year_forecast_obs == start_dates[0])[0][0]
    t1 = np.argwhere(year_forecast_obs == start_dates[-1])[0][0] + 1
    ilead0 = np.argwhere(lead_times == chosen_leads[0])[0][0]
    ilead1 = np.argwhere(lead_times == chosen_leads[-1])[0][0] + 1

    forecast_map_lagens = this_forecast_map_ensmn_expanded[t0:t1, ilead0:ilead1, :, :].mean(axis=1)

    obs_map_lagens = np.ma.masked_all_like(forecast_map_lagens)
    obs_map_djfm_lagens = np.ma.masked_all_like(forecast_map_lagens)
    for tt, year in enumerate(start_dates):
        obs_map_lagens[tt, :, :] = obs_map_expanded[t0+tt+ilead0:t0+tt+ilead1, :, :].mean(axis=0)
        obs_map_djfm_lagens[tt, :, :] = obs_map_expanded_djfm[t0+tt+ilead0:t0+tt+ilead1, :, :].mean(axis=0)

    forecast_map_lagens_grad = np.ma.masked_all(shape=(nj, ni))
    obs_map_grad = np.ma.masked_all(shape=(nj, ni))
    obs_map_djfm_grad = np.ma.masked_all(shape=(nj, ni))
    for jj in range(nj):
        print jj, nj
        if testing:
            if jj != 100: continue
        for ii in range(ni):
            if testing2:
                if jj < 130: continue
                if jj > 160: continue
                if ii < 120: continue
                if ii > 190: continue
            forecast_map_lagens_ts = forecast_map_lagens[:, jj, ii]
            if not forecast_map_lagens_ts.mask.all():
                real = np.nonzero(start_dates * forecast_map_lagens_ts)
                if len(real[0]) > 10:
                    grad, _, _, _, _ = stats.linregress(start_dates[real], forecast_map_lagens_ts[real])
                    forecast_map_lagens_grad[jj, ii] = grad

            obs_map_expanded_ts = obs_map_djfm_lagens[:, jj, ii]
            if not obs_map_expanded_ts.mask.all():
                real = np.nonzero(start_dates * obs_map_expanded_ts)
                if len(real[0]) > 10:
                    grad, _, _, _, _ = stats.linregress(start_dates[real], obs_map_expanded_ts[real])
                    obs_map_djfm_grad[jj, ii] = grad

            obs_map_expanded_ts = obs_map_lagens[:, jj, ii]
            if not obs_map_expanded_ts.mask.all():
                real = np.nonzero(start_dates * obs_map_expanded_ts)
                if len(real[0]) > 10:
                    grad, _, _, _, _ = stats.linregress(start_dates[real], obs_map_expanded_ts[real])
                    obs_map_grad[jj, ii] = grad

    if not testing:
        with open(skill_map_file, 'wb') as handle:
            print "Writing to: {:s}".format(skill_map_file)
            pickle.dump([skill_expanded, skill_expanded1960, skill_expanded1990, skill_expanded_multi,
                         skill_expanded1960_multi, skill_expanded1990_multi, res_skill_expanded,
                         res_skill_expanded1960, res_skill_expanded1990, res_skill_expanded_multi,
                         res_skill_expanded1960_multi, res_skill_expanded1990_multi, random_skill_expanded,
                         random_skill_expanded1960, random_skill_expanded1990, random_skill_expanded_multi,
                         random_skill_expanded1960_multi, random_skill_expanded1990_multi, 0,
                         forecast_map_lagens_grad, obs_map_djfm_grad, obs_map_grad, random_res_skill_expanded_multi,
                         random_res_skill_expanded1960_multi, random_res_skill_expanded1990_multi], handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
print "DONE!"
