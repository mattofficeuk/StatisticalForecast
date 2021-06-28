import numpy as np
import os
import sys
from numerics import find_nearest
import glob

radius_earth = 6371229.
deg2rad = np.pi / 180.

# ==================
# Function to return the models in CMIP
# ==================
def get_model_lists(project):
    if project == 'CMIP5':
        models = ['bcc-csm1-1', 'bcc-csm1-1-m', 'BNU-ESM', 'CanAM4', 'CanCM4', 'CanESM2',
                  'CMCC-CESM', 'CMCC-CM', 'CMCC-CMS', 'CNRM-CM5', 'CNRM-CM5-2', 'ACCESS1-0',
                  'ACCESS1-3', 'CSIRO-Mk3-6-0', 'SP-CCSM4', 'FIO-ESM', 'EC-EARTH', 'inmcm4',
                  'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'IPSL-CM5B-LR', 'FGOALS-g2', 'FGOALS-gl',
                  'FGOALS-s2', 'MIROC-ESM', 'MIROC-ESM-CHEM', 'MIROC4h', 'MIROC5', 'HadCM3',
                  'HadGEM2-A', 'HadGEM2-CC', 'HadGEM2-ES', 'MPI-ESM-LR', 'MPI-ESM-MR',
                  'MPI-ESM-P', 'MRI-AGCM3-2H', 'MRI-AGCM3-2S', 'MRI-CGCM3', 'MRI-ESM1',
                  'GISS-E2-H', 'GISS-E2-H-CC', 'GISS-E2-R', 'GISS-E2-R-CC', 'GEOS-5', 'CCSM4',
                  'NorESM1-M', 'NorESM1-ME', 'NICAM-09', 'HadGEM2-AO', 'GFDL-CM2p1', 'GFDL-CM3',
                  'GFDL-ESM2G', 'GFDL-ESM2M', 'GFDL-HIRAM-C180', 'GFDL-HIRAM-C360', 'CESM1-BGC',
                  'CESM1-CAM5', 'CESM1-CAM5-1-FV2', 'CESM1-FASTCHEM', 'CESM1-WACCM',
                  'CSIRO-Mk3L-1-2']
    elif project == 'CMIP6':
        models = ['AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'BCC-ESM1', 'CAMS-CSM1-0', 'FGOALS-f3-L',
                  'FGOALS-g3', 'IITM-ESM', 'CanESM5', 'CNRM-CM6-1', 'CNRM-CM6-1-HR',
                  'CNRM-ESM2-1', 'E3SM-1-0', 'E3SM-1-1', 'EC-Earth3', 'EC-Earth3-Veg',
                  'FIO-ESM-2-0', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'MIROC6',
                  'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'UKESM1-0-LL', 'MPI-ESM1-2-HR',
                  'MRI-ESM2-0', 'GISS-E2-1-G', 'GISS-E2-1-H', 'CESM2', 'CESM2-WACCM', 'NorCPM1',
                  'NorESM1-F', 'NorESM2-LM', 'GFDL-AM4',    'GFDL-CM4', 'GFDL-ESM4', 'NESM3',
                  'SAM0-UNICON', 'MCM-UA-1-0']
    elif project == 'CMIP5_subset':
        models = ['CNRM-CM5-2', 'CNRM-CM5', 'GISS-E2-H', 'GISS-E2-H-CC', 'GISS-E2-R',
                  'GISS-E2-R-CC', 'GFDL-CM3', 'bcc-csm1-1', 'bcc-csm1-1-m', 'HadGEM2-CC',
                  'HadGEM2-ES', 'EC-Earth', 'CESM1-CAM5', 'CanESM2', 'MIROC-ESM', 'MIROC-ESM-CHEM',
                  'MIROC5', 'CESM1-WACCM', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR']
    elif project == 'DAMIP':
        models = ['HadGEM3-GC31-LL', 'CNRM-CM6-1', 'GISS-E2-1-G', 'MIROC6', 'CanESM5',
                  'BCC-CSM2-MR', 'IPSL-CM6A-LR']
    elif project == 'CMIP5_Aer':
        models = ['MRI-CGCM3', 'HadGEM2-CC', 'MIROC5', 'MIROC-ESM-CHEM', 'CNRM-CM5', 'GFDL-CM3',
                  'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'NorESM1-M', 'inmcm4', 'CSIRO-Mk3-6-0', 'CanESM2',
                  'HadGEM2-ES', 'MIROC-ESM']
    elif project == 'CMIP5_NoAer':
        models = ['FIO-ESM', 'CESM1-BGC', 'CESM1-FASTCHEM', 'GFDL-ESM2G', 'GFDL-ESM2M', 'CCSM4',
                  'CESM1-WACCM', 'MPI-ESM-LR', 'bcc-csm1-1', 'CESM1-CAM5', 'BNU-ESM', 'EC-EARTH']
    return models

# ==================
# Function to return x0, x1, y0, y1 based on a named region
# ==================
def get_coords(region):
    if region == 'north_atlantic':
        xx = [-100, 10]
        yy = [0, 65]
    elif region == 'subpolar_gyre':
        xx = [-70, 10]
        yy = [45, 65]
    elif region == 'tropical_atlantic_south':  # Previously intergyre
        xx = [-60, 20]
        yy = [-30, 0]
    elif region == 'intergyre':  # Previously tropical Atlantic North
        xx = [-85, 0]
        yy = [30, 45]
    elif region == 'tropical_atlantic_north':  # Previously tropical Atlantic South
        xx = [-100, 20]
        yy = [0, 30]
    elif region == 'labsea':
        xx = [-70, -45]
        yy = [55, 65]
    elif region == 'irminger':
        xx = [-45, -30]
        yy = [58, 65]
    elif region == 'gin':
        xx = [-20, 20]
        yy = [65, 78]
    elif region == 'zhang_tna':
        xx = [-70, 10]
        yy = [0, 15]
    elif region == 'global60':
        xx = [-180, 180]
        yy = [-60, 60]
    elif region == 'nh':
        xx = [-180, 180]
        yy = [0, 90]
    elif region == 'spg_rahmstorf':
        xx = [-55, -20]
        yy = [46, 60]
    elif region == 'spg_menary18':
        xx = [-65, -20]
        yy = [50, 65]
    elif region == 'spg_leo20':
        xx = [-50, -10]
        yy = [45, 60]
    elif region == 'europe1':
        xx = [-10, 40]
        yy = [40, 70]
    else:
        raise ValueError("Unknown mask region")

    return xx, yy

# ==================
# Function to return i1, j2, i2, j2 (indices of 2 centres of action)
# based on a named region (e.g. NAO) and the 2D lat/lon
# ==================
def get_coords_diff(lon_in, lat_in, region):
    if region == 'nao1':
        iceland = [-21.1, 64.3]  # Lon, Lat of Iceland
        azores = [-25.9, 37.6]  # Lon, Lat of Azores
        j1, i1 = np.unravel_index(np.argmin(np.sqrt((lon_in - azores[0])**2 + (lat_in - azores[1])**2)),  lon_in.shape)
        j2, i2 = np.unravel_index(np.argmin(np.sqrt((lon_in - iceland[0])**2 + (lat_in - iceland[1])**2)),  lon_in.shape)
    else:
        raise ValueError("Unknown index region")
    return j1, i1, j2, i2

# ==================
# Function to make time series' from 3D (T, Y, X) data
# ==================
def make_ts_2d(var_tji_in, lon_in, lat_in, area_in, region):
    if region != 'global':
        xx, yy = get_coords(region)
        mask = ((lon_in > xx[1]) | (lon_in < xx[0]) | (lat_in > yy[1]) | (lat_in < yy[0]))
        area = np.ma.array(area_in, mask=mask)
    else:
        area = np.ma.array(area_in)
    area_tot = np.sum(area)

    nt, nj, ni = var_tji_in.shape
    out_ts = np.ma.masked_all(shape=nt)
    for tt in range(nt):
        out_ts[tt] = np.sum(var_tji_in[tt, :, :] * area) / area_tot
    return out_ts

# ==================
# Function to make time series' from 3D (T, Y, X) data
# but this time using difference between two points (e.g. for NAO)
# ==================
def make_ts_diff(var_tji_in, lon_in, lat_in, region):
    j1, i1, j2, i2 = get_coords_diff(lon_in, lat_in, region)
    out_ts = var_tji_in[:, j1, i1] - var_tji_in[:, j2, i2]
    return out_ts

# ==================
# Function to find the relevant files
# ==================
def get_files(institute, model, suffices=None, base_path=None):
    found_path = False
    found_files = None
    for suffix in suffices:
#         print base_path, institute, model, suffix
        full_path = os.path.join(base_path, institute, model, suffix)
        if os.path.isdir(full_path):
            found_path = True
            break
    if found_path:
        files = glob.glob(full_path + '/*.nc')
        found_files = []
        for ff in range(len(files)):
            found_files.append(os.path.realpath(os.path.join(full_path, files[ff])))
            # found_files.append(os.path.join(full_path, files[ff]))
    else:
        print "Could not find files"
        print "Path start: {:s}".format(os.path.join(base_path, institute, model))
        print "Final attempt was: {:s}".format(full_path)
    return found_files

# ==================
# Function to edit the suffices for the DCPP experiments
# which use a different file structure
# ==================
def edit_suffices_for_dcpp(input_suffix_array):
    output_suffix_array = []
    for element in input_suffix_array:
        print element
        split = element.split('/')
        if split[0] == 'piControl':  # If we've added bonus piControl suffices (for cell info etc) then don't alter these
            fixed_element = element
        else:
            year = np.long(split[0][7:])  # Converting it to a long means it should fail if it's not a number
            fixed_element = os.path.join('dcppA-hindcast', 's{:d}-'.format(year) + split[1], '/'.join(split[2:]))
        output_suffix_array.append(fixed_element)
    return output_suffix_array

# ==================
# Functions to make an Atlantic mask
# ==================
def get_indices(lon, lat):
    distance = np.sqrt((lon**2 + lat**2))
    indices = np.unravel_index(np.argmin(distance), lon.shape)
    return indices

def mask_nearby(variable_masked, jj, ii, recur_count=0):
    sys.setrecursionlimit(10000)
    max_recur = 3000
    ndim = variable_masked.ndim
    if ndim == 3:
        variable_masked.mask[:, jj, ii] = True
    elif ndim == 2:
        variable_masked.mask[jj, ii] = True
    else:
        raise ValueError("ndim's is wrong")
    if recur_count < max_recur:
        for jj2 in range(jj-1, jj+2, 2):
            if ndim == 3: surface_val = variable_masked.mask[0, jj2, ii]
            if ndim == 2: surface_val = variable_masked.mask[jj2, ii]
            if surface_val == False:
                variable_masked, recur_count = mask_nearby(variable_masked, jj2, ii,
                                                           recur_count=recur_count+1)
        for ii2 in range(ii-1, ii+2, 2):
            if ndim == 3: surface_val = variable_masked.mask[0, jj, ii2]
            if ndim == 2: surface_val = variable_masked.mask[jj, ii2]
            if surface_val == False:
                variable_masked, recur_count = mask_nearby(variable_masked, jj, ii2,
                                                           recur_count=recur_count+1)
    return variable_masked, recur_count

def apply_mask(variable, lon, lat, model='', north_atlantic_only=False):
    if north_atlantic_only == True:
        # Initial large mask
        mask = ((lon < -110) | (lon > 70)) | \
               ((lat < 40 ) | (lat > 85))
    else:
        # Initial large mask
        mask = ((lon < -100) | (lon > 60)) | \
               ((lat < -31 ) | (lat > 85)) | \
               ((lat < 5) & ((lon < -66.5) | (lon > 23.5))) | \
               ((lon > 23.5) & (lon < 180) & (lat < 29)) | \
               ((lon > 44.6) & (lon < 180) & (lat > 25) & (lat < 49.5))

    # Special rules :-(
    special_models = ['CESM2', 'CESM1-CAM5-2-FV2', 'GISS-E2-1-G', 'EC-Earth3-Veg', 'EC-Earth3',
                      'SAM0-UNICON', 'CESM2-WACCM', 'MIROC6', 'CESM1-BGC', 'CESM1-WACCM', 'CESM1-CAM5',
                      'CESM1-FASTCHEM', 'GFDL-ESM2G', 'CCSM4', 'GISS-E2-R', 'GISS-E2-H-CC', 'MRI-CGCM3']
    if model in special_models:
        print " == Applying special rule for {:s}".format(model)
    if model in ['CESM2', 'CESM1-CAM5-2-FV2', 'SAM0-UNICON', 'CESM2-WACCM', 'CESM1-BGC', 'CESM1-WACCM',
                 'CESM1-CAM5', 'CESM1-FASTCHEM', 'CCSM4', 'CESM1-CAM5-1-FV2']:
        mask[219:224, 280] = True
        mask[224, 279] = True
    elif model in ['GISS-E2-1-G', 'GISS-E2-R-CC', 'GISS-E2-R']:
        mask[98, 220:227] = True
        mask[98:101, 220] = True
    elif model in ['EC-Earth3-Veg', 'EC-Earth3']:
        mask[168, 206:208] = True
    elif model == 'MIROC6':
        mask[141, 217] = True
    elif model == 'GFDL-ESM2G':
        mask[147:154, 308:322] = True
    elif model == 'GISS-E2-H-CC':
        mask[97, 275:284] = True
        mask[97:99, 275] = True
    elif model == 'MRI-CGCM3':
        mask[170:177, 190:197] = True
        mask[170:175, 190:198] = True
        mask[170:180, 190:196] = True
        mask[172:174, 194:200] = True
        mask[171:173, 194:205] = True

    ndim = variable.ndim
    if ndim == 3:
        ndep, nj, ni = variable.shape
        mask = np.repeat(mask[np.newaxis, :, :], ndep, axis=0)

    variable_masked = np.ma.array(variable, mask=mask)

    if north_atlantic_only == False:
        # Now find the i-indices around the difficult Pacific bit
        target_lon = 268.5 - 360.
        target_lat = 7.6
        jj, ii = get_indices(lon - target_lon, lat - target_lat)
        # Mask the data in this patch
        variable_masked, recur_count = mask_nearby(variable_masked, jj, ii)

    return variable_masked

# ==================
# Functions to estimate e1v
# ==================
def find_vertex_indices(lon_bnds_vertices_in, lat_bnds_vertices_in):
    lon_bnds_vertices = lon_bnds_vertices_in.copy()
    lat_bnds_vertices = lat_bnds_vertices_in.copy()
    bottom_left = np.intersect1d(np.argsort(lon_bnds_vertices)[:2], np.argsort(lat_bnds_vertices)[:2])[0]
    top_left = np.intersect1d(np.argsort(lon_bnds_vertices)[:2], np.argsort(lat_bnds_vertices)[2:])[0]
    bottom_right = np.intersect1d(np.argsort(lon_bnds_vertices)[2:], np.argsort(lat_bnds_vertices)[:2])[0]
    top_right = np.intersect1d(np.argsort(lon_bnds_vertices)[2:], np.argsort(lat_bnds_vertices)[2:])[0]
    return [bottom_left, bottom_right, top_right, top_left]

def spherical_to_cartesian(r, la, lo):
    rlo = np.radians(lo)
    rla = np.radians(90 - la)
    x = r * np.sin(rla) * np.cos(rlo)
    y = r * np.sin(rla) * np.sin(rlo)
    z = r * np.cos(rla)
    return np.array([x, y, z])

def arc_length(lon_bnds_vertices, lat_bnds_vertices, j_direction=False):
    nj, ni, n_vertices = lon_bnds_vertices.shape
    jj, ii = 100, 100  # For testing 4-corner data
    if n_vertices == 2:
        print "n_vertices_equals_2: Should check this model"
        bottom, top = 0, 1
        left, right = 0, 1

        if j_direction:
            # This delta_lon should be basically zero unless the cells are wonky
            lat1 = lat_bnds_vertices[:, :, bottom] * deg2rad  # Mean lat at BOTTOM FACE
            lat2 = lat_bnds_vertices[:, :, top] * deg2rad     # Mean lat at TOP FACE
            delta_lon = 0.
        else:
            # These lats should be basically the same unless the cells are wonky
            lat1 = (lat_bnds_vertices[:, :, bottom] + lat_bnds_vertices[:, :, top]) * deg2rad / 2. # Mean lat at LEFT FACE
            lat2 = lat1 # Mean lat at RIGHT FACE
            lon1 = lon_bnds_vertices[:, :, left] * deg2rad   # Mean lon on left FACE
            lon2 = lon_bnds_vertices[:, :, right] * deg2rad  # Mean lon on right FACE
            delta_lon = (lon2 - lon1)
    # elif (len(np.unique(lon_bnds_vertices[jj, ii, :].round(decimals=4))) == 2) and (len(np.unique(lat_bnds_vertices[jj, ii, :].round(decimals=4))) == 2):
    #     # Where there are 4 "corners" but actually they are just (wrongly?) copied
    #     # Appears to be the case for MPI-ESM1-2-HR decadal data
    elif n_vertices == 4:
        print "n_vertices_equals_4"
        bottom_left, bottom_right, top_right, top_left = find_vertex_indices(lon_bnds_vertices[jj, ii, :], lat_bnds_vertices[jj, ii, :])

        if j_direction:
            # This delta_lon should be basically zero unless the cells are wonky
            lat1 = (lat_bnds_vertices[:, :, bottom_left] + lat_bnds_vertices[:, :, bottom_right]) * deg2rad / 2. # Mean lat at BOTTOM FACE
            lat2 = (lat_bnds_vertices[:, :, top_left] + lat_bnds_vertices[:, :, top_right]) * deg2rad / 2. # Mean lat at TOP FACE
            lon1 = (lon_bnds_vertices[:, :, bottom_left] + lon_bnds_vertices[:, :, bottom_right]) * deg2rad / 2.   # Mean lon on bottom FACE
            lon2 = (lon_bnds_vertices[:, :, top_left] + lon_bnds_vertices[:, :, top_right]) * deg2rad / 2. # Mean lon on top FACE
            delta_lon = (lon2 - lon1)
        else:
            # These lats should be basically the same unless the cells are wonky
            lat1 = (lat_bnds_vertices[:, :, bottom_left] + lat_bnds_vertices[:, :, top_left]) * deg2rad / 2. # Mean lat at LEFT FACE
            lat2 = (lat_bnds_vertices[:, :, bottom_right] + lat_bnds_vertices[:, :, top_right]) * deg2rad / 2. # Mean lat at RIGHT FACE
            lon1 = (lon_bnds_vertices[:, :, top_left] + lon_bnds_vertices[:, :, bottom_left]) * deg2rad / 2.   # Mean lon on left FACE
            lon2 = (lon_bnds_vertices[:, :, top_right] + lon_bnds_vertices[:, :, bottom_right]) * deg2rad / 2. # Mean lon on right FACE
            delta_lon = (lon2 - lon1)

    central_angle = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon))

    # https://en.wikipedia.org/wiki/Great-circle_distance
    arclen = radius_earth * central_angle
    return arclen

# ==================
# Function to find the variable names
# ==================
def find_var_name(file_handle_in, variable, second_loaded=False):
    if variable == 'depth':
        potential_vars = ['deptho', 'lev', 'olevel', 'depth']
    elif variable == 'moc':
        potential_vars = ['msftmyz', 'msftmz', 'msftyz']
    elif variable == 'depth_bounds':
        potential_vars = ['lev_bnds', 'lev_bounds', 'olevel_bnds', 'olevel_bounds']
    elif variable == 'longitude':
        potential_vars = ['longitude', 'lon', 'nav_lon']
    elif variable == 'latitude':
        potential_vars = ['latitude', 'lat', 'nav_lat', 'rlat']
    elif variable == 'longitude_bounds':
        potential_vars = ['vertices_longitude', 'lon_bnds', 'bounds_lon',
                          'bounds_nav_lon', 'lon_vertices', 'rlon_bnds']
    elif variable == 'latitude_bounds':
        potential_vars = ['vertices_latitude', 'lat_bnds', 'bounds_lat',
                          'bounds_nav_lat', 'lat_vertices', 'rlat_bnds']
    elif variable == 'areacella':
        potential_vars = ['areacella']
    else:
        raise ValueError("Illegal variable name")

    var_name = ''
    file_handle_out = False
    for potential_var in potential_vars:
        if potential_var in file_handle_in.variables.keys():
            var_name = potential_var
            file_handle_out = file_handle_in
            break

    if file_handle_out == False and second_loaded != False:
        print " == Searching second file handle"
        for potential_var in potential_vars:
            if potential_var in second_loaded.variables.keys():
                var_name = potential_var
                file_handle_out = second_loaded

    return var_name, file_handle_out

# ==================
# Function to guess bounds if the input is 1D
# ==================
def guess_bounds(in_arr):
    nn = len(in_arr)
    bnds = np.zeros((nn, 2))
    edges = (in_arr[1:] + in_arr[:-1]) / 2.
    bnds[:-1, 1] = bnds[1:, 0] = edges
    delta = edges[1] - edges[0]
    bnds[-1, 1] = bnds[-2, 1] + delta
    bnds[0, 0] = bnds[1, 0] - delta
    return bnds
