#!/usr/bin/env python3
# The above is required for sbatch submission

import numpy as np
import pickle
import os
import glob
import sys
import xarray as xr
from scipy import interpolate

analogue_var = 'SST'
model = sys.argv[3]

# The problem with using a very early clim period is that the obs are actually just THEIR clim,
# which is 1960-1990 anyway
clim_start, clim_end = 1960, 1990
# clim_start, clim_end = 1940, 1970
# clim_start, clim_end = 1900, 2015
# clim_start, clim_end = 1900, 1990
# clim_start, clim_end = 1980, 1990

myhost = os.uname()[1]
print("myhost = {:s}".format(myhost))
usr = os.environ["USER"]

if 'ciclad' in myhost:
    # Ciclad options
    datadir = '/data/mmenary/python_saves/CMIP_{:s}/'.format(analogue_var)
else:
    # Jasmin options
    datadir = '/work/scratch-nopw/{}/CMIP_{:s}/'.format(usr,analogue_var)

climatology_file = os.path.join(datadir, 'CMIP_{:s}_{:s}_historical-EnsMn_TM{:d}-{:d}_Annual.nc'.format(analogue_var, model, clim_start, clim_end))

tolerance = 0.9 * (clim_end - clim_start)

# Prepare regridding
def regrid_sst(sst_in, year_in):
    nyrs = len(year_in)
    area = get_area()
    sst_regridded = np.ma.masked_all(shape=(nyrs, nj, ni))

    for tt in range(nyrs):
#         if tt not in [97, 98, 99, 100]: continue
        print('{:d}/{:d}'.format(tt, nyrs))
#        print(len(np.array([lat_in.ravel(), lon_in.ravel()]).T))
#        print(len(sst_in[0, 0, :, :].ravel()))
#        print(len(sst_in[0, 0, :, :].mask.ravel()))
        sst_regridded[tt, :, :] = interpolate.griddata(np.array([lat_in.ravel(), lon_in.ravel()]).T,
                                                       sst_in[0, tt, :, :].ravel(), (lat_re, lon_re),
                                                       method='linear')
    mask_regridded = interpolate.griddata(np.array([lat_in.ravel(), lon_in.ravel()]).T,
                                          sst_in[0, 0, :, :].mask.ravel(), (lat_re, lon_re),
                                          method='linear')
    sst_regridded = np.ma.array(sst_regridded, mask=np.repeat(mask_regridded[np.newaxis, :, :], nyrs, axis=0))
    return sst_regridded

def get_area():
    radius_earth = 6.4e6
    dlon = dlat = np.radians(1.)
    area = np.ones(shape=(nj, ni))
    for jj in range(nj):
        area[jj, :] *= radius_earth**2 * np.cos(np.radians(lat_re[jj])) * np.sin(dlon) * np.sin(dlat)
    return area


if os.path.isfile(climatology_file):
    print('Already created climatology file:  {:s}'.format(climatology_file))
else:
    hist_files_field = glob.glob(os.path.join(datadir, 'CMIP?_{:s}field_{:s}_historical-*_Annual.nc'.format(analogue_var, model)))
    hist_files_field.sort()
    hist_files_mask = glob.glob(os.path.join(datadir, 'CMIP?_{:s}mask_{:s}_historical-1_Annual.nc'.format(analogue_var, model)))

    nj, ni = 180, 360
    lon_re = np.repeat((np.arange(-180, 180) + 0.5)[np.newaxis, :], nj, axis=0)
    lat_re = np.repeat((np.arange(-90, 90) + 0.5)[:, None], ni, axis=1)
    climatology = np.zeros(shape=(nj, ni))
    count = 0.
    for ifile, hist_file in enumerate(hist_files_field):
        print(ifile, hist_file)

        ds_mask = xr.open_dataset(hist_files_mask[0]).to_array()
        mask_in = ds_mask.values
        ds_field = xr.open_dataset(hist_file).to_array()
        sst_read = np.ma.masked_array(ds_field.values,mask=mask_in)
        lon_in = ds_field['lon'].values
        lat_in = ds_field['lat'].values
        year_in = ds_field['time'].values

        sst_in = regrid_sst(sst_read,year_in)

        clim_years = np.nonzero((year_in > clim_start) * (year_in <= clim_end))[0]
        nyrs = len(clim_years)
        if nyrs < tolerance:
            continue

        climatology += np.ma.mean(sst_in[clim_years[0]:clim_years[-1], :, :], axis=0)
        count += 1.

    climatology /= count
    climatology = np.ma.masked_array(climatology, mask=sst_in[clim_years[0], :, :].mask)

    clim_print = xr.DataArray(climatology, name='climatology', dims = ['y','x'], coords = {'lat': (['y','x'],lat_re), 'lon': (['y','x'],lon_re)}).to_dataset(name='climatology')

    #with open(climatology_file, 'wb') as handle:
    print('Writing to:  {:s}'.format(climatology_file))
    clim_print.to_netcdf(path=climatology_file,format="NETCDF4")
