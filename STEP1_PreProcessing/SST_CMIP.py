#!/usr/bin/env python3
# host = 'ciclad'
host = 'jasmin'

import netCDF4
import numpy as np
import os
import sys

usr = os.environ["USER"]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

import pickle
import glob
from cmip import *
import time as time_module  # Because I use "time" elsewhere
from scipy import interpolate
import xarray as xr

if host == 'ciclad':
    raise ValueError("This is now deprecated")
    save_dir = '/home/users/{}/data/python_saves/CMIP_SST'.format(usr)
    # save_dir_netcdf = '/home/users/mmenary/data/python_saves/CMIP_SST_NetCDF'
    # list_location = '/home/{}/python/scripts'.format(usr)
elif host == 'jasmin':
    # save_dir = '/gws/nopw/j04/acsis/mmenary/python_saves/CMIP_SST'
    save_dir = '/work/scratch-nopw/{}/CMIP_SST'.format(usr)
    # save_dir_netcdf = '/gws/nopw/j04/acsis/mmenary/python_saves/CMIP_SST_NetCDF'
else:
    raise ValueError("Unknown host")

# ==================
# Input data
# ==================
experiment = sys.argv[1]
ens_mem = sys.argv[2]
model = sys.argv[3]
period_string = sys.argv[4]
time_series_only = sys.argv[5]
testing = sys.argv[6]
list_location = sys.argv[7]

print('Inputs: ', sys.argv)

seasonal = False
annual = False
JJA = False
MAM = False

if period_string == 'Seasonal':
    seasonal = True
    save_dir += '_Seas'
elif period_string == 'annual':
    annual = True
elif period_string == 'JJA':
    JJA = True
elif period_string == 'MAM':
    MAM = True
else:
    raise ValueError('period_string unknown')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

if time_series_only == 'True':
    time_series_only = True
    time_series_only_string = '_TimeSeries'
else:
    time_series_only = False
    time_series_only_string = ''

if testing == 'True':
    TESTING = True
    max_times = 36
else:
    TESTING = False

if TESTING: print("\n==========\nTESTING mode !!\n==========\n")

# ==================
# Constants
# ==================
# save_netcdfs = True
already_read_sftof = False
already_read_basin = False
already_read_salinity = False
radius_earth = 6371229.
deg2rad = np.pi / 180.
regions = ['north_atlantic', 'subpolar_gyre', 'intergyre', 'tropical_atlantic_north',
           'tropical_atlantic_south', 'global', 'global60', 'nh', 'spg_rahmstorf',
           'spg_menary18', 'spg_leo20']

cmip5_list_file = os.path.join(list_location, 'cmip5_list.txt')
cmip6_list_file = os.path.join(list_location, 'cmip6_list.txt')

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

# Models for which we have to use the regridded (not by us, by the centres, which is why we don't want to use it normally) data
# because the non-regridded is just too large.
models_requiring_gr = ['GFDL-CM4']

# This is for the DAMIP in CMIP5, where the "p" was used to determine which experiment was being run
# Unfortunately, this is model dependant. Fortunately, Laura Wilcox made a very helpful table!
# These "p" values are just for the historicalAA experiment (and historicalAero for CCSM4)
if (project == 'CMIP5') and (experiment == 'historicalMisc'):
    if model == 'CSIRO-Mk3-6-0':
        perturbed = '4'
    elif model == 'CanESM2':
        perturbed = '4'
    elif model == 'IPSL-CM5A-LR':
        perturbed = '3'
    elif model == 'CCSM4':
        perturbed = '10'
    elif model == 'GISS-E2-R':
        perturbed = '107'
    elif model == 'NorESM1-M':
        perturbed = '1'
    elif model == 'GFDL-CM3':
        perturbed = '1'
    elif model == 'GFDL-ESM2M':
        perturbed = '5'
    elif model == 'CESM1-CAM5-1-FV2':
        perturbed = '10'
    else:
        raise ValueError("There is no historicalAA (or similar) in CMIP5 for this model:  {:s}".format(model))
elif (project == 'CMIP6') and (experiment[:7] == 'decadal') and (model == 'CanESM5'):
    perturbed = '2'
else:
    perturbed = '1'

dcppa = False
if project == 'CMIP5':
    print(" == PROJECT=CMIP5")
    if host == 'ciclad':
        base_path_data = base_path_coords = '/bdd/CMIP5/output'
    elif host == 'jasmin':
        base_path_data = base_path_coords = '/badc/cmip5/data/cmip5/output1'
    if (model == 'HadCM3') and (experiment[:7] == "decadal"):
        ic_val = 2
    else:
        ic_val = 1
    suffices = ['{:s}/mon/ocean/Omon/r{:s}i{:d}p{:s}/latest/thetao'.format(experiment, ens_mem, ic_val, perturbed)]
    thk_suffices = ['{:s}/fx/ocean/fx/r0i0p0/latest/deptho'.format(experiment)]
    Ofx_suffices = ['{:s}/fx/ocean/fx/r0i0p0/latest/areacello'.format(experiment)]
else:
    print(" == PROJECT=CMIP6")
    if experiment in ['piControl', 'historical']:
        if host == 'ciclad': base_path_data = '/bdd/CMIP6/CMIP'
        if host == 'jasmin': base_path_data = '/badc/cmip6/data/CMIP6/CMIP'
    elif experiment in ['hist-nat', 'hist-GHG', 'hist-aer', 'hist-stratO3', 'ssp245-GHG', 'ssp245-nat']:
        if host == 'ciclad': base_path_data = '/bdd/CMIP6/DAMIP'
        if host == 'jasmin': base_path_data = '/badc/cmip6/data/CMIP6/DAMIP'
    elif experiment[:3] == 'ssp':
        if host == 'ciclad': base_path_data = '/bdd/CMIP6/ScenarioMIP'
        if host == 'jasmin': base_path_data = '/badc/cmip6/data/CMIP6/ScenarioMIP'
    elif experiment[:7] == 'decadal':
        if host == 'ciclad': base_path_data = '/bdd/CMIP6/DCPP'
        if host == 'jasmin': base_path_data = '/badc/cmip6/data/CMIP6/DCPP'
        dcppa = True
    else:
        raise ValueError(" !! Not sure where this experiment lives")
    if host == 'ciclad': base_path_coords = '/bdd/CMIP6/CMIP'
    if host == 'jasmin': base_path_coords = '/badc/cmip6/data/CMIP6/CMIP'
    if model in models_requiring_gr:
        print("Defaulting to using data regridded by model institution (gr grid instead of gn)")
        gn_or_gr = "gr"
    else:
        gn_or_gr = "gn"
    suffices = ['{:s}/r{:s}i1p{:s}f1/Omon/thetao/{:s}/latest'.format(experiment, ens_mem, perturbed, gn_or_gr),
                '{:s}/r{:s}i1p{:s}f2/Omon/thetao/{:s}/latest'.format(experiment, ens_mem, perturbed, gn_or_gr),
                '{:s}/r{:s}i1p{:s}f3/Omon/thetao/{:s}/latest'.format(experiment, ens_mem, perturbed, gn_or_gr),
                '{:s}/r{:s}i1p{:s}f1/Omon/thetao/gr/latest'.format(experiment, ens_mem, perturbed),
                '{:s}/r{:s}i1p{:s}f1/Omon/thetao/gr1/latest'.format(experiment, ens_mem, perturbed)]
    Ofx_suffices = ['{:s}/r1i1p{:s}f1/Ofx/areacello/{:s}/latest'.format(experiment, perturbed, gn_or_gr),
                    '{:s}/r1i1p{:s}f2/Ofx/areacello/{:s}/latest'.format(experiment, perturbed, gn_or_gr),
                    '{:s}/r1i1p{:s}f1/Ofx/areacello/gr/latest'.format(experiment, perturbed)]
    thk_suffices = ['{:s}/r1i1p{:s}f1/Ofx/deptho/{:s}/latest'.format(experiment, perturbed, gn_or_gr),
                    '{:s}/r1i1p{:s}f2/Ofx/deptho/{:s}/latest'.format(experiment, perturbed, gn_or_gr),
                    '{:s}/r1i1p{:s}f1/Ofx/deptho/gr/latest'.format(experiment, perturbed)]
    if experiment != 'piControl':
        Ofx_suffices_piControl = ['{:s}/r1i1p1f1/Ofx/areacello/{:s}/latest'.format('piControl', gn_or_gr),
                                  '{:s}/r1i1p1f2/Ofx/areacello/{:s}/latest'.format('piControl', gn_or_gr),
                                  '{:s}/r1i1p1f1/Ofx/areacello/gr/latest'.format('piControl', gn_or_gr)]
        for extra_suffix in Ofx_suffices_piControl:
            Ofx_suffices.append(extra_suffix)

if model in ['NorESM2-LM', 'NorCPM1']:
    # Have to choose "gr" here, which I think means "regridded" rather than "native"
    suffices = ['{:s}/r{:s}i1p1f1/Omon/thetao/gr/latest'.format(experiment, ens_mem)]
    thk_suffices = ['{:s}/r1i1p1f1/Ofx/deptho/gr/latest'.format(experiment)]

if dcppa:
    suffices = edit_suffices_for_dcpp(suffices)
    Ofx_suffices = edit_suffices_for_dcpp(Ofx_suffices)
    thk_suffices = edit_suffices_for_dcpp(thk_suffices)

if experiment == 'piControl':
    ens_mem_string = ''
else:
    ens_mem_string = '-{:s}'.format(ens_mem)

# save_file = '{:s}/{:s}_SST_{:s}_{:s}{:s}_Monthly{:s}.pkl'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
if annual:
    save_file_ann_field = '{:s}/{:s}_SSTfield_{:s}_{:s}{:s}_Annual{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_ann_timeser = '{:s}/{:s}_SSTtimeser_{:s}_{:s}{:s}_Annual{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_ann_mask = '{:s}/{:s}_SSTmask_{:s}_{:s}{:s}_Annual{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
elif JJA:
    save_file_ann_field = '{:s}/{:s}_SSTfield_{:s}_{:s}{:s}_JJA{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_ann_timeser = '{:s}/{:s}_SSTtimeser_{:s}_{:s}{:s}_JJA{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_ann_mask = '{:s}/{:s}_SSTmask_{:s}_{:s}{:s}_JJA{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
elif MAM:
    save_file_ann_field = '{:s}/{:s}_SSTfield_{:s}_{:s}{:s}_MAM{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_ann_timeser = '{:s}/{:s}_SSTtimeser_{:s}_{:s}{:s}_MAM{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_ann_mask = '{:s}/{:s}_SSTmask_{:s}_{:s}{:s}_MAM{:s}.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
# save_file_regridded = '{:s}/{:s}_SST_{:s}_{:s}{:s}_{:s}_Regridded{:s}.pkl'.format(save_dir, project, model, experiment, ens_mem_string, period_string, time_series_only_string)
# netcdf_save_file = '{:s}/{:s}_SST_{:s}_{:s}{:s}_Annual{:s}.nc'.format(save_dir_netcdf, project, model, experiment, ens_mem_string, time_series_only_string)
if TESTING:
    # save_file += '.TEST'
    save_file_ann += '.TEST'
    # save_file_regridded += '.TEST'
    # netcdf_save_file += '.TEST'

# print "Will save to:\n{:s}\n{:s}\n{:s}\n{:s}\n".format(save_file, save_file_ann, save_file_regridded, netcdf_save_file)
print("Will save to: {:s} and {:s}".format(save_file_ann_field,save_file_ann_timeser))


# ==================
# Find the institute associated with the input model
# ==================
institutes = os.listdir(base_path_data)
inst_model = {}  # Store the institutes with the models as the keys
for institute in institutes:
    these_models = os.listdir(os.path.join(base_path_data, institute))
    for this_model in these_models:
        inst_model[this_model] = institute
institute = inst_model[model]

# ==================
# Find the thetao files
# ==================
# print institute, model, suffices, base_path_data
thetao_files = get_files(institute, model, suffices=suffices, base_path=base_path_data)
thetao_files.sort()
if thetao_files == None:
    raise ValueError("No thetao files found !")
else:
    data_model = thetao_files

# ==================
# Read the coordinate data in
# ==================
thetao_files = get_files(institute, model, suffices=suffices, base_path=base_path_data)
thetao_files.sort()
loaded = netCDF4.Dataset(thetao_files[0])

lon_name, _ = find_var_name(loaded, 'longitude')
lat_name, _ = find_var_name(loaded, 'latitude')

bounds_file = get_files(institute, model, suffices=Ofx_suffices, base_path=base_path_coords)
if bounds_file != None:
    loaded3 = netCDF4.Dataset(bounds_file[0])
else:
    loaded3 = False
lon_bnds_name, lon_bnds_pointer = find_var_name(loaded, 'longitude_bounds', second_loaded=loaded3)
lat_bnds_name, lat_bnds_pointer = find_var_name(loaded, 'latitude_bounds', second_loaded=loaded3)

lon_bnds_vertices = lon_bnds_pointer.variables[lon_bnds_name][:]
lat_bnds_vertices = lat_bnds_pointer.variables[lat_bnds_name][:]
lon = loaded.variables[lon_name][:]
lat = loaded.variables[lat_name][:]
loaded.close()

if lon.ndim == lon_bnds_vertices.ndim:  # When the vertices are missing a dimension
    print(" == Inflating lon_bnds dimension as it is (wrongly) missing j-coords for: {:s}".format(model))
    nj, ni = lon.shape  # Must have 2D by definition
    lon_bnds_vertices = np.repeat(lon_bnds_vertices[np.newaxis, :, :], nj, axis=0)

if lat.ndim == lat_bnds_vertices.ndim:  # When the vertices are missing a dimension
    print(" == Inflating lat_bnds dimension as it is (wrongly) missing i-coords for: {:s}".format(model))
    nj, ni = lon.shape  # Must have 2D by definition
    lat_bnds_vertices = np.repeat(lat_bnds_vertices[:, np.newaxis, :], ni, axis=1)

if lon.ndim < 2:
    print(" == Converting coordinates to 2D for {:s}".format(model))
    nj = len(lat)
    ni = len(lon)
    lon = np.repeat(lon[np.newaxis, :], nj, axis=0)
    lat = np.repeat(lat[:, np.newaxis], ni, axis=1)
    lon_bnds_vertices = np.repeat(lon_bnds_vertices[np.newaxis, :, :], nj, axis=0)
    lat_bnds_vertices = np.repeat(lat_bnds_vertices[:, np.newaxis, :], ni, axis=1)

# Make the longitudes -180 -> +180
lon = (lon + 360.) % 360.
lon[lon > 180] -= 360
lon_bnds_vertices = (lon_bnds_vertices + 360.) % 360.
lon_bnds_vertices[lon_bnds_vertices > 180] -= 360

# ==================
# Figure out the size of the output SST array
# ==================
ntimes_total = 0
years = []
for ifile, thetao_file in enumerate(thetao_files):
    loaded = netCDF4.Dataset(thetao_file)
    time = loaded.variables['time'][:]
    ntimes = len(time)
    ntimes_total += ntimes

    t0, t1 = os.path.basename(thetao_file).split('_')[-1].split('.nc')[0].split('-')
    y0, y1 = int(t0[:4]), int(t1[:4])
    m0, m1 = int(t0[4:]), int(t1[4:])

    for tt in range(ntimes):
        year = y0 + (tt // 12)
        years.append(year)

# For both annual AND seasonal, we are going to ignore years that aren't full (12 months)
year_ann, counts = np.unique(years, return_counts=True)
year_ann = year_ann[counts == 12].astype('int')
print("Unique, full years are:", year_ann)

if seasonal:
    ntimes_to_keep_while_working = 3
    nseasons = 4
elif annual:
    ## Annual or single season
    ntimes_to_keep_while_working = 12
    nseasons = 1
elif JJA:
    ntimes_to_keep_while_working = 3
    nseasons = 1
elif MAM:
    ntimes_to_keep_while_working = 3
    nseasons = 1


ntimes_output = len(year_ann) * nseasons

# ntimes_total_max = ntimes_total
# ntimes_total = int(np.min([ntimes_total, 12 * 500]))  # Max 500 years (memory)
# print("ntimes_total could be as high as {:d}. It is set to {:d}".format(ntimes_total_max, ntimes_total))

nj, ni = lon.shape
print("Creating large numpy arrays: ntimes_output, nj, ni = ", ntimes_output, nj, ni)
## Removed to save memory. It was only being used for time series,
## so "sst" has been moved to within the loop now
# sst = np.ma.masked_all(shape=(ntimes_total, nj, ni))
sst_ann = np.ma.masked_all(shape=(ntimes_output, nj, ni))

# ==================
# Prepare the sst_ts dictionary (to avoid reading in both
## "sst_global" and "sst" array, to save memory)
# ==================
sst_ts_ann = {}
for region in regions:
    sst_ts_ann[region] = np.ma.masked_all(shape=(ntimes_output))

# These are the working arrays
sst_atlantic_working = np.ma.masked_all(shape=(ntimes_to_keep_while_working, nj, ni))
sst_global_working = np.ma.masked_all(shape=(ntimes_to_keep_while_working, nj, ni))
year_working = np.ma.masked_all(shape=(ntimes_to_keep_while_working))
mon_working = np.ma.masked_all(shape=(ntimes_to_keep_while_working))

# ==================
# Read the thetao data and make the SST
# ==================
tt2 = -1
exit_loops = False
for ifile, thetao_file in enumerate(thetao_files):
    if exit_loops: break
    print(thetao_file)
    loaded = netCDF4.Dataset(thetao_file)

    time = loaded.variables['time'][:]
    ntimes = len(time)

    t0, t1 = os.path.basename(thetao_file).split('_')[-1].split('.nc')[0].split('-')
    y0, y1 = int(t0[:4]), int(t1[:4])
    m0, m1 = int(t0[4:]), int(t1[4:])

    if JJA:
        y0 += 5
    if MAM:
        y0 += 2

    for tt in range(ntimes):
        tt2 += 1
        if exit_loops: break
        print("{:4d} of {:4d} in this file, {:4d} of {:4d} overall".format(tt, ntimes, tt2, ntimes_total))
        thetao = loaded.variables['thetao'][tt, 0, :, :]

        peak_to_peak = thetao.ptp()
        if peak_to_peak < 0.1:
            print("WARNING: This data looks to all be missing")
            continue

        if model in ['GISS-E2-1-H', 'GISS-E2-H', 'MIROC5', 'GISS-E2-R']:
            thetao = np.ma.masked_greater(thetao, 1e10)
            if not already_read_sftof:
                if model == 'MIROC5':
                    guess_at_mask_file = base_path_data + '/MIROC/MIROC5/historical/fx/ocean/fx/r0i0p0/' + \
                                         'latest/sftof/sftof_fx_MIROC5_historical_r0i0p0.nc'
                    mask_less_than_this = 0.99
                elif model == 'GISS-E2-R':
                    guess_at_mask_file = base_path_data + '/NASA-GISS/GISS-E2-R/piControl/fx/ocean/fx/r0i0p0/' + \
                                         'latest/sftof/sftof_fx_GISS-E2-R_piControl_r0i0p0.nc'
                    mask_less_than_this = 0.99
                else:
                    guess_at_mask_file = base_path_data + '/NASA-GISS/GISS-E2-1-H/piControl/r1i1p1f1/' + \
                                         'Ofx/sftof/gr/latest/sftof_Ofx_GISS-E2-1-H_piControl_r1i1p1f1_gr.nc'
                    mask_less_than_this = 99.9
                loaded4 = netCDF4.Dataset(guess_at_mask_file)
                sftof = np.ma.masked_less(loaded4.variables['sftof'][:], mask_less_than_this)
                loaded4.close()
                already_read_sftof = True
            thetao = np.ma.array(thetao, mask=sftof.mask)
        elif model == 'EC-EARTH':
            if not already_read_salinity:
                guess_at_mask_file = base_path_data + '/ICHEC/EC-EARTH/piControl/mon/ocean/Omon/r1i1p1/' + \
                                     'latest/so/so_Omon_EC-EARTH_piControl_r1i1p1_210001-255112.nc'
                loaded4 = netCDF4.Dataset(guess_at_mask_file)
                so_mask = np.ma.masked_less(loaded4.variables['so'][0, 0, :, :], 0.1).mask
                loaded4.close()
                already_read_salinity = True
            thetao = np.ma.array(thetao, mask=so_mask)

        thetao_global = thetao.copy()

        if model in ['MRI-ESM1', 'MRI-CGCM3', 'MIROC4h']:
            if not already_read_basin:
                if model in ['MRI-ESM1', 'MRI-CGCM3']:
                    basin_mask_file = base_path_data + '/MRI/MRI-ESM1/historical/fx/ocean/fx/r0i0p0/' + \
                                      'latest/basin/basin_fx_MRI-ESM1_historical_r0i0p0.nc'
                elif model == 'MIROC4h':
                    basin_mask_file = base_path_data + '/MIROC/MIROC4h/historical/fx/ocean/fx/r0i0p0/' + \
                                      'latest/basin/basin_fx_MIROC4h_historical_r0i0p0.nc'
                loaded4 = netCDF4.Dataset(basin_mask_file)
                basin = np.ma.masked_not_equal(loaded4.variables['basin'], 2)
                basin_global = np.ma.masked_not_equal(loaded4.variables['basin'], 0)
                loaded4.close()
                already_read_basin = True
            thetao = np.ma.array(thetao, mask=basin.mask)
            thetao_global = np.ma.array(thetao_global, mask=(1 - basin_global.mask))

        thetao_masked = apply_mask(thetao, lon, lat, model=model)

        # Make masked lon and dlon (one time only) and check if data is in Kelvin
        if ifile == 0 and tt == 0:
            xdist = arc_length(lon_bnds_vertices, lat_bnds_vertices)
            xdist_global = np.ma.array(xdist, mask=thetao_global.mask)
            xdist = np.ma.array(xdist, mask=thetao_masked.mask)
            ydist = arc_length(lon_bnds_vertices, lat_bnds_vertices, j_direction=True)
            ydist_global = np.ma.array(ydist, mask=thetao_global.mask)
            ydist = np.ma.array(ydist, mask=thetao_masked.mask)
            area = xdist * ydist
            area_global = xdist_global * ydist_global
            if thetao_masked.mean() > 200:
                kelvin_offset = 273.15
            else:
                kelvin_offset = 0.

        # Make the SST and store it in the working arrays. After these reach "full" then take the time mean
        # and then compute the area averages
        tt2_mod = tt2 % ntimes_to_keep_while_working
        sst_atlantic_working[tt2_mod, :, :] = thetao_masked - kelvin_offset
        sst_global_working[tt2_mod, :, :]   = thetao_global - kelvin_offset

        year_working[tt2_mod] = y0 + (tt // 12)      # Goes up by 1 every 12 indices (i.e. months)
        mon_working[tt2_mod]  = m0 + (tt % 12)       # Goes up by 1, looping every 12 (i.e. annually)

        # Check and see if we have enough data to make a seasonal or annual mean
        indices_for_meaning = [-1, -1, -1]
        index_of_timemean = -1
        reached_meaning_window = False
        if seasonal:
            # Check each season. If we have the correct months for a seasonal mean stored in the working arrays then
            # store the indices of these and work out the index in the target TIME MEAN array. Then stop searching.
            for season_index in range(nseasons):
                ind = np.argwhere((mon_working == (season_index * 3 + 1)) | (mon_working == (season_index * 3 + 2)) | (mon_working == (season_index * 3 + 3)))
                if len(ind) == ntimes_to_keep_while_working:
                    assert year_working[ind[0]] == year_working[ind[-1]]  ## Should both be the same year (as we aren't using Dec in winter means)
                    indices_for_meaning = ind.flatten()
                    year_index = np.argwhere(year_ann == year_working[ind[0]])
                    index_of_timemean = year_index + season_index
                    reached_meaning_window = True
                    break
        elif annual:
            # Check each year. If we have the correct months for an annual mean stored in the working arrays then
            # store the indices of these and work out the index in the target TIME MEAN array. Then stop searching.
            for year_index, possible_year in enumerate(year_ann):
                ind = np.argwhere(year_working == possible_year)
                if len(ind) == ntimes_to_keep_while_working:
                    indices_for_meaning = ind.flatten()
                    index_of_timemean = year_index
                    reached_meaning_window = True
                    break
        elif JJA:
            # Check each year. If we have the correct months for a summer mean stored in the working arrays then
            # store the indices of these and work out the index in the target TIME MEAN array. Then stop searching.
            for season_index in range(nseasons):
                ind = np.argwhere((mon_working == (season_index * 3 + 1)) | (mon_working == (season_index * 3 + 2)) | (mon_working == (season_index * 3 + 3)))
                if len(ind) == ntimes_to_keep_while_working:
                    indices_for_meaning = ind.flatten()
                    year_index = np.argwhere(year_ann == year_working[ind[0]])
                    index_of_timemean = year_index
                    reached_meaning_window = True
                    break
        elif MAM:
            # Check each year. If we have the correct months for a summer mean stored in the working arrays then
            # store the indices of these and work out the index in the target TIME MEAN array. Then stop searching.
            for season_index in range(nseasons):
                ind = np.argwhere((mon_working == (season_index * 3 + 1)) | (mon_working == (season_index * 3 + 2)) | (mon_working == (season_index * 3 + 3)))
                if len(ind) == ntimes_to_keep_while_working:
                    indices_for_meaning = ind.flatten()
                    year_index = np.argwhere(year_ann == year_working[ind[0]])
                    index_of_timemean = year_index
                    reached_meaning_window = True
                    break


        if reached_meaning_window:
            sst_atlantic_masked = sst_atlantic_working[indices_for_meaning, :, :].mean(axis=0, keepdims=True)
            sst_global_masked = sst_global_working[indices_for_meaning, :, :].mean(axis=0, keepdims=True)

            print("sst_global_masked.shape", sst_global_masked.shape, sst_global_working.shape)

            for region in regions:
                if region in ['global', 'global60', 'nh']:
                    sst_area_averaged = make_ts_2d(sst_global_masked, lon, lat, area, region)
                else:
                    sst_area_averaged = make_ts_2d(sst_atlantic_masked, lon, lat, area, region)
                sst_ts_ann[region][index_of_timemean] = sst_area_averaged

            sst_ann[index_of_timemean, :, :] = sst_global_masked

        # year[tt2] = y0 + (tt // 12)
        # mon[tt2] = m0 + (tt % 12)
        # tt2 += 1

        # if tt2 >= ntimes_total:
        #     exit_loops = True
        #
        # if TESTING:
        #     if tt2 >= max_times:
        #         exit_loops = True

    loaded.close()

for region in regions:
    if region == 'north_atlantic':
        sst_timesers = xr.DataArray(sst_ts_ann[region], name=region, dims = ['time'], coords = {'time': (['time'],year_ann)}).to_dataset(name='north_atlantic')
    else:
        sst_timesers[region] = xr.DataArray(sst_ts_ann[region], name=region, dims = ['time'], coords = {'time': (['time'],year_ann)})

print(sst_ann.shape)

sst_field = xr.DataArray(sst_ann, name='SST', dims = ['time','y','x'], coords = {'time': (['time'],year_ann), 'lat': (['y','x'],lat), 'lon': (['y','x'],lon)}).to_dataset(name='SST')

sst_mask = xr.DataArray(sst_ann[:,:,:].mask, name="mask", dims = ['time','y','x'], coords = {'time': (['time'],year_ann), 'lat': (['y','x'],lat), 'lon': (['y','x'],lon)}).to_dataset(name='mask')

# ==================
# Save the data (annual version only?)
# ==================
#with open(save_file_ann, 'wb') as handle:
#    print("Saving SST data: {:s}".format(save_file_ann))
#    print(sst_ann.shape, area.shape, lon.shape, lat.shape, year_ann.shape)
#    if time_series_only:
#        pickle.dump([sst_ts_ann, year_ann], handle, protocol=pickle.HIGHEST_PROTOCOL)
#    else:
#        if seasonal:
#            pickle.dump([sst_ann, sst_ts_ann, area, lon, lat, year_ann, season], handle, protocol=pickle.HIGHEST_PROTOCOL)
#        else:
#            pickle.dump([sst_ann, sst_ts_ann, area, lon, lat, year_ann], handle, protocol=pickle.HIGHEST_PROTOCOL)
#    print("DONE!")

# sst_ts_ann_list = list(sst_ts_ann.items())
# sst_ts_ann_arr = np.array(sst_ts_ann_list,dtype=object)

print("Saving SST data: {:s} and {:s}".format(save_file_ann_field,save_file_ann_timeser))
#print(sst_timesers.shape,sst_ann.shape, area.shape, lon.shape, lat.shape, year_ann.shape)

print(lon)

if time_series_only:
    #ds = xr.Dataset({'SST': (('time'), sst_timesers)}, coords={'region':regions,'time':year_ann})
    sst_timesers.to_netcdf(path=save_file_ann_timeser,format="NETCDF4")
else:
    if seasonal:
        seasonal_cycle = np.arange(nseasons) + 1
        pickle.dump([sst_ann, sst_ts_ann, area, lon, lat, year_ann, seasonal_cycle], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        sst_timesers.to_netcdf(path=save_file_ann_timeser,format="NETCDF4")
        sst_field.to_netcdf(path=save_file_ann_field,format="NETCDF4")
        sst_mask.to_netcdf(path=save_file_ann_mask,format="NETCDF4")

print("DONE!")

# ==================
# Save the data (annual version only?)
# ==================
#if not time_series_only:
#    with open(save_file_regridded, 'wb') as handle:
#        print("Saving SST data: {:s}".format(save_file_regridded))
#        print(sst_regridded.shape, year_ann.shape, seasonal_cycle.shape)
#        pickle.dump([sst_regridded, year_ann, seasonal_cycle], handle, protocol=pickle.HIGHEST_PROTOCOL)
#        print("DONE!")

# if save_netcdfs:
#     print "Saving SST data: {:s}".format(netcdf_save_file)
#     save_to_netcdf(netcdf_save_file, sst_ts_ann['spg_leo20'], year_ann)
#     print "DONE!"

print("Program finished sucessfully")
