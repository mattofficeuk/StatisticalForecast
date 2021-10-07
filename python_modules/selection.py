# compute_spatial_correlation.py
# defines the procedure by which the analogues are chosen (e.g. RMSE, spatial correlation, etc.) and returns a metric of the fit of model simulations to observations at all points in time

import numpy as np

def calc_rmse(aa, bb):
    return np.sqrt(np.ma.mean((aa - bb)**2))


def csc(method, sst_obs, year_obs, sst_model, year_model):
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
