import numpy as np
import pickle
import os
import glob
import sys

analogue_var = 'SST'
model = sys.argv[1]

# The problem with using a very early clim period is that the obs are actually just THEIR clim,
# which is 1960-1990 anyway
clim_start, clim_end = 1960, 1990
# clim_start, clim_end = 1940, 1970
# clim_start, clim_end = 1900, 2015
# clim_start, clim_end = 1900, 1990
# clim_start, clim_end = 1980, 1990

myhost = os.uname()[1]
print "myhost = {:s}".format(myhost)

if 'ciclad' in myhost:
    # Ciclad options
    datadir = '/data/mmenary/python_saves/CMIP_{:s}/'.format(analogue_var)
else:
    # Jasmin options
    datadir = '/work/scratch-nopw/mmenary/CMIP_{:s}/'.format(analogue_var)

climatology_file = os.path.join(datadir, 'CMIP_{:s}_{:s}_historical-EnsMn_TM{:d}-{:d}_Annual.pkl'.format(analogue_var, model, clim_start, clim_end))

tolerance = 0.9 * (clim_end - clim_start)

if os.path.isfile(climatology_file):
    print('Already created climatology file:  {:s}'.format(climatology_file))
else:
    hist_files = glob.glob(os.path.join(datadir, 'CMIP?_{:s}_{:s}_historical-*_Annual_Regridded.pkl'.format(analogue_var, model)))
    hist_files.sort()

    nj, ni = 180, 360
    climatology = np.zeros(shape=(nj, ni))
    count = 0.
    for ifile, hist_file in enumerate(hist_files):
        print ifile, hist_file

        with open(hist_file, 'rb') as handle:
            sst_in, year_in = pickle.load(handle)
            clim_years = np.nonzero((year_in > clim_start) * (year_in <= clim_end))[0]
            nyrs = len(clim_years)
            if nyrs < tolerance:
                continue

            climatology += np.ma.mean(sst_in[clim_years[0]:clim_years[-1], :, :], axis=0)
            count += 1.

    climatology /= count
    climatology = np.ma.masked_array(climatology, mask=sst_in[clim_years[0], :, :].mask)

    with open(climatology_file, 'wb') as handle:
        print('Writing to:  {:s}'.format(climatology_file))
        pickle.dump(climatology, handle,  protocol=pickle.HIGHEST_PROTOCOL)
