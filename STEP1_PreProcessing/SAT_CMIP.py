#!/usr/bin/env python3

# ===============
# Set variables & modules
# ===============

# host = 'ciclad'
host = 'jasmin'

import netCDF4
import numpy as np
import os
import sys
import xarray as xr

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

if host == 'ciclad':
    raise ValueError("deprecated")
    save_dir = '/data/{.s}/python_saves/CMIP_SAT'.format(usr)
    # list_location = '/home/{.s}/python/scripts'.format(usr)
elif host == 'jasmin':
    #save_dir = '/gws/nopw/j04/acsis/mmenary/python_saves/CMIP_SAT'
    save_dir = '/work/scratch-nopw/{}/CMIP_SAT'.format(usr)
    # list_location = '/home/users/{}/python/scripts'.format(usr)
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

# These set boundary variables; DO NOT CHANGE. The actual season is read in from period_string above.
seasonal = False
annual = False
JJA = False

if period_string == 'Seasonal':
    seasonal = True
    save_dir += '_Seas'
elif period_string == 'annual':
    annual = True
elif period_string == 'JJA':
    JJA = True
else:
    raise ValueError('period_string unknown')

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
already_read_sftof = False
already_read_basin = False
already_read_salinity = False
radius_earth = 6371229.
deg2rad = np.pi / 180.
regions = ['europe1', 'south_europe']

cmip5_list_file = list_location + '/cmip5_list.txt'
cmip6_list_file = list_location + '/cmip6_list.txt'

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
    suffices = ['{:s}/mon/atmos/Amon/r{:s}i1p{:s}/latest/tas'.format(experiment, ens_mem, perturbed)]
    # thk_suffices = ['{:s}/fx/atmos/fx/r0i0p0/latest/deptho'.format(experiment)]
    fx_suffices = ['{:s}/fx/atmos/fx/r0i0p0/latest/areacella'.format(experiment)]
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
    suffices = ['{:s}/r{:s}i1p{:s}f1/Amon/tas/gn/latest'.format(experiment, ens_mem, perturbed),
                '{:s}/r{:s}i1p{:s}f2/Amon/tas/gn/latest'.format(experiment, ens_mem, perturbed),
                '{:s}/r{:s}i1p{:s}f3/Amon/tas/gn/latest'.format(experiment, ens_mem, perturbed),
                '{:s}/r{:s}i1p{:s}f1/Amon/tas/gr/latest'.format(experiment, ens_mem, perturbed),
                '{:s}/r{:s}i1p{:s}f1/Amon/tas/gr1/latest'.format(experiment, ens_mem, perturbed)]
    fx_suffices = ['{:s}/r1i1p{:s}f1/fx/areacella/gn/latest'.format(experiment, perturbed),
                    '{:s}/r1i1p{:s}f2/fx/areacella/gn/latest'.format(experiment, perturbed),
                    '{:s}/r1i1p{:s}f1/fx/areacella/gr/latest'.format(experiment, perturbed)]
    # thk_suffices = ['{:s}/r1i1p{:s}f1/fx/deptho/gn/latest'.format(experiment, perturbed),
    #                 '{:s}/r1i1p{:s}f2/fx/deptho/gn/latest'.format(experiment, perturbed),
    #                 '{:s}/r1i1p{:s}f1/fx/deptho/gr/latest'.format(experiment, perturbed)]
    if experiment != 'piControl':
        fx_suffices_piControl = ['{:s}/r1i1p1f1/fx/areacella/gn/latest'.format('piControl'),
                                  '{:s}/r1i1p1f2/fx/areacella/gn/latest'.format('piControl'),
                                  '{:s}/r1i1p1f1/fx/areacella/gr/latest'.format('piControl')]
        for extra_suffix in fx_suffices_piControl:
            fx_suffices.append(extra_suffix)

if model == 'NorESM2-LM':
    # Have to choose "gr" here, which I think means "regridded" rather than "native"
    suffices = ['{:s}/r{:s}i1p1f1/Amon/tas/gr/latest'.format(experiment, ens_mem)]
    # thk_suffices = ['{:s}/r1i1p1f1/fx/thkcello/gr/latest'.format(experiment)]

if dcppa:
    suffices = edit_suffices_for_dcpp(suffices)
    fx_suffices = edit_suffices_for_dcpp(fx_suffices)
    # thk_suffices = edit_suffices_for_dcpp(thk_suffices)

if experiment == 'piControl':
    ens_mem_string = ''
else:
    ens_mem_string = '-{:s}'.format(ens_mem)

if annual:
    save_file_regridded = '{:s}/{:s}_SATfield_{:s}_{:s}{:s}_Annual.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_mask = '{:s}/{:s}_SATmask_{:s}_{:s}{:s}_Annual.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_timeseries = '{:s}/{:s}_SATtimeser_{:s}_{:s}{:s}_Annual.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
elif JJA:
    save_file_regridded = '{:s}/{:s}_SATfield_{:s}_{:s}{:s}_JJA.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_mask = '{:s}/{:s}_SATmask_{:s}_{:s}{:s}_JJA.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
    save_file_timeseries = '{:s}/{:s}_SATtimeser_{:s}_{:s}{:s}_JJA.nc'.format(save_dir, project, model, experiment, ens_mem_string, time_series_only_string)
if TESTING:
    # save_file += '.TEST'
    # save_file_ann += '.TEST'
    save_file_regridded += '.TEST'

print("Will save to: {:s}".format(save_file_regridded))

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
# Find the tas files
# ==================
# print institute, model, suffices, base_path_data
tas_files = get_files(institute, model, suffices=suffices, base_path=base_path_data)
if tas_files == None:
    raise ValueError("No tas files found !")
else:
    tas_files.sort()
    data_model = tas_files

# ==================
# Read the coordinate data in
# ==================
tas_files = get_files(institute, model, suffices=suffices, base_path=base_path_data)
tas_files.sort()
loaded = netCDF4.Dataset(tas_files[0])

lon_name, _ = find_var_name(loaded, 'longitude')
lat_name, _ = find_var_name(loaded, 'latitude')

bounds_file = get_files(institute, model, suffices=fx_suffices, base_path=base_path_coords)
if bounds_file != None:
    loaded3 = netCDF4.Dataset(bounds_file[0])
else:
    loaded3 = False
lon_bnds_name, lon_bnds_pointer = find_var_name(loaded, 'longitude_bounds', second_loaded=loaded3)
lat_bnds_name, lat_bnds_pointer = find_var_name(loaded, 'latitude_bounds', second_loaded=loaded3)
area_name, area_pointer = find_var_name(loaded, 'areacella', second_loaded=loaded3)

lon = loaded.variables[lon_name][:]
lat = loaded.variables[lat_name][:]

if area_pointer:
    area = area_pointer.variables[area_name][:]

    if lon.ndim < 2:
        print(" == Converting coordinates to 2D for {:s}".format(model))
        nj = len(lat)
        ni = len(lon)
        lon = np.repeat(lon[np.newaxis, :], nj, axis=0)
        lat = np.repeat(lat[:, np.newaxis], ni, axis=1)
else:
    print(" -++- NO AREA DATA -++-")
    if lon_bnds_pointer:
        lon_bnds_vertices = lon_bnds_pointer.variables[lon_bnds_name][:]
    else:
        lon_bnds_vertices = guess_bounds(lon)
    if lat_bnds_pointer:
        lat_bnds_vertices = lat_bnds_pointer.variables[lat_bnds_name][:]
    else:
        lat_bnds_vertices = guess_bounds(lat)

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
    lon_bnds_vertices = (lon_bnds_vertices + 360.) % 360.
    lon_bnds_vertices[lon_bnds_vertices > 180] -= 360

# Make the longitudes -180 -> +180
lon = (lon + 360.) % 360.
lon[lon > 180] -= 360

loaded.close()

# ==================
# Figure out the size of the output SAT array
# ==================
ntimes_total = 0
for ifile, tas_file in enumerate(tas_files):
    loaded = netCDF4.Dataset(tas_file)
    time = loaded.variables['time'][:]
    ntimes_total += len(time)

ntimes_total = int(np.min([ntimes_total, 12 * 500]))  # Max 500 years (memory)

nj, ni = lon.shape
sat = np.ma.masked_all(shape=(ntimes_total, nj, ni))
# sat_global = np.ma.masked_all(shape=(ntimes_total, nj, ni))
year = np.ma.masked_all(shape=(ntimes_total))
mon = np.ma.masked_all(shape=(ntimes_total))

# ==================
# Read the tas data and make the SAT
# ==================
tt2 = 0
exit_loops = False
for ifile, tas_file in enumerate(tas_files):
    if exit_loops: break
    print(tas_file)
    loaded = netCDF4.Dataset(tas_file)

    time = loaded.variables['time'][:]
    ntimes = len(time)

    t0, t1 = os.path.basename(tas_file).split('_')[-1].split('.nc')[0].split('-')
    y0, y1 = int(t0[:4]), int(t1[:4])
    m0, m1 = int(t0[4:]), int(t1[4:])

    for tt in range(ntimes):
        if exit_loops: break
        print("{:4d} of {:4d} in this file, {:4d} of {:4d} overall".format(tt, ntimes, tt2, ntimes_total))
        tas = loaded.variables['tas'][tt, :, :]

        peak_to_peak = tas.ptp()
        if peak_to_peak < 0.1:
            print("WARNING: This data looks to all be missing/0")
            continue

        # if model in ['GISS-E2-1-H', 'GISS-E2-H', 'GISS-E2-R']:
        #     tas = np.ma.masked_greater(tas, 1e10)
        #     if not already_read_sftof:
        #         if model == 'MIROC5':
        #             guess_at_mask_file = base_path_data + '/MIROC/MIROC5/historical/fx/atmos/fx/r0i0p0/' + \
        #                                  'latest/sftof/sftof_fx_MIROC5_historical_r0i0p0.nc'
        #             mask_less_than_this = 0.99
        #         elif model == 'GISS-E2-R':
        #             guess_at_mask_file = base_path_data + '/NASA-GISS/GISS-E2-R/piControl/fx/atmos/fx/r0i0p0/' + \
        #                                  'latest/sftof/sftof_fx_GISS-E2-R_piControl_r0i0p0.nc'
        #             mask_less_than_this = 0.99
        #         else:
        #             guess_at_mask_file = base_path_data + '/NASA-GISS/GISS-E2-1-H/piControl/r1i1p1f1/' + \
        #                                  'fx/sftof/gr/latest/sftof_fx_GISS-E2-1-H_piControl_r1i1p1f1_gr.nc'
        #             mask_less_than_this = 99.9
        #         loaded4 = netCDF4.Dataset(guess_at_mask_file)
        #         sftof = np.ma.masked_less(loaded4.variables['sftof'][:], mask_less_than_this)
        #         loaded4.close()
        #         sftof = np.repeat(sftof[np.newaxis, :, :], nk, axis=0)
        #         already_read_sftof = True
        #     tas = np.ma.array(tas, mask=sftof.mask)
        # elif model in ['MIROC5', 'EC-EARTH', 'MRI-CGCM3']:
        #     if not already_read_salinity:
        #         if model == 'MIROC5':
        #             guess_at_mask_file = base_path_data + '/MIROC/MIROC5/piControl/mon/atmos/Amon/r1i1p1/' + \
        #                                  'latest/so/so_Amon_MIROC5_piControl_r1i1p1_200001-200012.nc'
        #         elif model == 'EC-EARTH':
        #             guess_at_mask_file = base_path_data + '/ICHEC/EC-EARTH/piControl/mon/atmos/Amon/r1i1p1/' + \
        #                                  'latest/so/so_Amon_EC-EARTH_piControl_r1i1p1_210001-255112.nc'
        #         elif model == 'MRI-CGCM3':
        #             print 'HERE!!!!!!'
        #             guess_at_mask_file = base_path_data + '/MRI/MRI-CGCM3/piControl/mon/atmos/Amon/r1i1p1/' + \
        #                                  'latest/so/so_Amon_MRI-CGCM3_piControl_r1i1p1_185101-185512.nc'
        #         loaded4 = netCDF4.Dataset(guess_at_mask_file)
        #         so_mask = np.ma.masked_less(loaded4.variables['so'][0, k0:k1, :, :], 0.1).mask
        #         loaded4.close()
        #         already_read_salinity = True
        #     tas = np.ma.array(tas, mask=so_mask)
        #
        # tas_global = tas.copy()
        #
        # if model in ['MRI-ESM1', 'MIROC4h']:
        #     if not already_read_basin:
        #         if model == 'MRI-ESM1':
        #             basin_mask_file = base_path_data + '/MRI/MRI-ESM1/historical/fx/atmos/fx/r0i0p0/' + \
        #                               'latest/basin/basin_fx_MRI-ESM1_historical_r0i0p0.nc'
        #         elif model == 'MIROC4h':
        #             basin_mask_file = base_path_data + '/MIROC/MIROC4h/historical/fx/atmos/fx/r0i0p0/' + \
        #                               'latest/basin/basin_fx_MIROC4h_historical_r0i0p0.nc'
        #         loaded4 = netCDF4.Dataset(basin_mask_file)
        #         basin = np.ma.masked_not_equal(loaded4.variables['basin'], 2)
        #         basin_global = np.ma.masked_not_equal(loaded4.variables['basin'], 0)
        #         loaded4.close()
        #         basin = np.repeat(basin[np.newaxis, :, :], nk, axis=0)
        #         basin_global = np.repeat(basin_global[np.newaxis, :, :], nk, axis=0)
        #         already_read_basin = True
        #     tas = np.ma.array(tas, mask=basin.mask)
        #     tas_global = np.ma.array(tas_global, mask=(1 - basin_global.mask))
        #
        # tas_masked = apply_mask(tas, lon, lat, model=model)
        #
        # Make masked lon and dlon (one time only) and check if data is in Kelvin
        if ifile == 0 and tt == 0:
            print("Doing 1-time-only things")
            if not area_pointer:
                xdist = arc_length(lon_bnds_vertices, lat_bnds_vertices)
                # xdist_global = np.ma.array(xdist, mask=tas_global.mask)
                # xdist = np.ma.array(xdist, mask=tas_masked.mask)
                ydist = arc_length(lon_bnds_vertices, lat_bnds_vertices, j_direction=True)
                # ydist_global = np.ma.array(ydist, mask=tas_global.mask)
                # ydist = np.ma.array(ydist, mask=tas_masked.mask)
                area = xdist * ydist
                # area_global = xdist_global * ydist_global
                # thick3d = np.repeat(np.repeat(thk[:, np.newaxis], nj, axis=1)[:, :, np.newaxis], ni, axis=2)
                # thick3d_global = np.ma.array(thick3d, mask=tas_global.mask)
                # bottom_layer = tas_global[-1, :, :]
                # thick3d_global = np.ma.array(thick3d, mask=np.repeat(bottom_layer[np.newaxis, :, :], nk, axis=0).mask)  # Mask using bottom level
                # thick3d_global_weighted = thick3d / np.repeat(np.ma.sum(thick3d_global, axis=0)[np.newaxis, :, :], nk, axis=0)
                # thick3d_masked_weighted = np.ma.array(thick3d_global_weighted, mask=tas_masked.mask)
            if tas.mean() > 200:
                kelvin_offset = 273.15
            else:
                kelvin_offset = 0.
        #
        # # Make the Depth averaged temperature
        # tas_masked_depavg = np.ma.sum(tas_masked * thick3d_masked_weighted, axis=0)
        # tas_global_depavg = np.ma.sum(tas_global * thick3d_global_weighted, axis=0)

        # Store it
        sat[tt2, :, :] = tas - kelvin_offset
        # sat_global[tt2, :, :] = tas_global_depavg - kelvin_offset

        year[tt2] = y0 + (tt // 12)
        mon[tt2] = m0 + (tt % 12)
        tt2 += 1

        if tt2 >= ntimes_total:
            exit_loops = True

        if TESTING:
            if tt2 >= max_times:
                exit_loops = True

time = year + (mon - 1) / 12.

# ==================
# Premake some time series
# ==================
sat_ts = {}
for region in regions:
    print("Making SAT average for {:s}".format(region))
    # if region in ['global', 'global60', 'nh']:
    #     sat_ts[region] = make_ts_2d(sat_global, lon, lat, area, region)
    # else:
    sat_ts[region] = make_ts_2d(sat, lon, lat, area, region)

# ==================
# Make annual mean versions
# ==================
nt, nj, ni = sat.shape
year_ann, counts = np.unique(year, return_counts=True)
year_ann = year_ann[counts == 12].astype('int')

if seasonal:
    nseasons = 4
else:
    nseasons = 1
seasonal_cycle = np.arange(nseasons) + 1

sat_ann = np.ma.masked_all(shape=(len(year_ann) * nseasons, nj, ni))  # maps
sat_ts_ann = {}  # time series
for region in regions:
    sat_ts_ann[region] = np.ma.masked_all(shape=len(year_ann) * nseasons)

for iyr, yy in enumerate(year_ann):
    for iseason, ss in enumerate(seasonal_cycle):
        if seasonal:
            ind = np.argwhere((year == yy) & ((mon == (iseason * 3 + 1)) | (mon == (iseason * 3 + 2)) | (mon == (iseason * 3 + 3))))
        elif annual:
            ind = np.argwhere(year == yy) 
        elif JJA:
            ind = np.argwhere((year == yy) & ((mon == (6 + 1)) | (mon == (6 + 2)) | (mon == (6 + 3))))

        sat_ann[iyr * nseasons + iseason, :, :] = sat[ind, :, :].mean(axis=0)
        for region in regions:
            sat_ts_ann[region][iyr * nseasons + iseason] = sat_ts[region][ind].mean(axis=0)

# ==================
# Do some regridding
# ==================
if not time_series_only:
    nt_in, nj_in, ni_in = sat_ann.shape
    nj_re = 180
    ni_re = 360

    sat_regridded = np.ma.masked_all(shape=(nt_in, nj_re, ni_re))
    lon_re = np.repeat((np.arange(-180, 180) + 0.5)[np.newaxis, :], nj_re, axis=0)
    lat_re = np.repeat((np.arange(-90, 90) + 0.5)[:, None], ni_re, axis=1)

    for tt in range(nt_in):
        print('Regridding SAT map {:d}'.format(tt))
        sat_regridded[tt, :, :] = interpolate.griddata(np.array([lat.ravel(), lon.ravel()]).T,
                                                       sat_ann[tt, :, :].ravel(), (lat_re, lon_re),
                                                       method='linear')
    mask_regridded1 = interpolate.griddata(np.array([lat.ravel(), lon.ravel()]).T,
                                          sat[0, :, :].mask.ravel(), (lat_re, lon_re),
                                          method='linear')
    mask_regridded = np.repeat(mask_regridded1[np.newaxis, :, :], nt_in, axis=0)
    sat_regridded = np.ma.array(sat_regridded, mask=mask_regridded)

# ==================
# Save the data (annual version only?)
# ==================
# with open(save_file_ann, 'wb') as handle:
#     print "Saving SAT data: {:s}".format(save_file_ann)
#     print sat_ann.shape, area.shape, lon.shape, lat.shape, year_ann.shape
#     if time_series_only:
#         pickle.dump([sat_ts_ann, year_ann], handle, protocol=pickle.HIGHEST_PROTOCOL)
#     else:
#         pickle.dump([sat_ann, sat_ts_ann, area, lon, lat, year_ann], handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print "DONE!"

# ==================
# Save the data (annual version only?)
# ==================
#if not time_series_only:
#    with open(save_file_regridded, 'wb') as handle:
#        print("Saving SAT data: {:s}".format(save_file_regridded))
#        print(sat_regridded.shape, year_ann.shape, seasonal_cycle.shape)
#        pickle.dump([sat_regridded, year_ann, seasonal_cycle], handle, protocol=pickle.HIGHEST_PROTOCOL)
#        print("DONE!")

for region in regions:
    if region == 'europe1':
        sat_timesers = xr.DataArray(sat_ts_ann[region], name=region, dims = ['time'], coords = {'time': (['time'],year_ann)}).to_dataset(name='europe1')
    else:
        sat_timesers[region] = xr.DataArray(sat_ts_ann[region], name=region, dims = ['time'], coords = {'time': (['time'],year_ann)})

print(sat_regridded.shape)
print(mask_regridded.shape)

sat_field = xr.DataArray(sat_regridded, name='SAT', dims = ['time','y','x'], coords = {'time': (['time'],year_ann), 'lat': (['y','x'],lat_re), 'lon': (['y','x'],lon_re)}).to_dataset(name='SAT')
sat_mask = xr.DataArray(mask_regridded, name="mask", dims = ['time','y','x'], coords = {'time': (['time'],year_ann), 'lat': (['y','x'],lat_re), 'lon': (['y','x'],lon_re)}).to_dataset(name='mask')

print("Saving SAT data: {:s} and {:s}".format(save_file_regridded,save_file_timeseries))
#print(sst_timesers.shape,sst_ann.shape, area.shape, lon.shape, lat.shape, year_ann.shape)

if time_series_only:
    #ds = xr.Dataset({'SST': (('time'), sst_timesers)}, coords={'region':regions,'time':year_ann})
    sat_timesers.to_netcdf(path=save_file_timeseries,format="NETCDF4")
else:
    sat_timesers.to_netcdf(path=save_file_timeseries,format="NETCDF4")
    sat_field.to_netcdf(path=save_file_regridded,format="NETCDF4")
    sat_mask.to_netcdf(path=save_file_mask,format="NETCDF4")

print("DONE!")


print("Program finished sucessfully")
