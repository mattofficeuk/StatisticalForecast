import numpy as np
import os
import sys
import glob
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import hashlib

# Make a class to store the data. This might save me time in the future...
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

# Make a class to store the output data. This might save me time in the future...
class OutputMetrics:
    def __init__(self):
        self.source_index = []          # The index into the stored_data list
        self.corrs = []                 # The maximum correlation (any window - noted below)
        self.grads = []                 # The grad at maximal corr (for constructing the forecast data)
        self.intercepts = []            # The intercept at maximal corr (for constructing the forecast data)
        self.info = []                  # Currently the model
#         self.source_subvar_index = [] # Could add the region
        self.source_time_index = []     # The temporal index of the maximal corr

def remove_masked(list_in):
    while np.ma.is_masked(list_in[0]) or (list_in[0] == ''):
        list_in = list_in[1:]
    return list_in

def make_info_matrices(corr_info, target_years, num_mems_to_take):
    #models = remove_masked(list(np.unique(corr_info[:, :, 1])))
    #expts_used_unsorted = remove_masked(list(np.unique(corr_info[:, :, 2])))
    models = remove_masked(list(np.unique(corr_info[:, :])))
    expts_used_unsorted = remove_masked(list(np.unique(corr_info[:, :])))
    decadal_list = ['decadal' + str(num) for num in range(1960, 2021)]
    expts = ['piControl', 'historical', 'hist-GHG', 'hist-aer', 'hist-nat', 'hist-stratO3',
             'rcp45', 'ssp126',  'rcp85', 'ssp585'] + decadal_list
    expts_used = expts[:]
    for expt in expts:
        if expt not in expts_used_unsorted:
            expts_used.remove(expt)

    # This way gets rid of all unused models
    model_matrix = np.ma.masked_all(shape=(len(target_years), num_mems_to_take), dtype='int')
    expt_matrix = np.ma.masked_all(shape=(len(target_years), num_mems_to_take), dtype='int')
    for iyr, year in enumerate(target_years):
        for jj in range(num_mems_to_take):
#            if np.isfinite(corr_info[iyr, jj, 0]) == True:
#                model = corr_info[iyr, jj, 1]
#                if model == '': continue
#                expt = corr_info[iyr, jj, 2]
#                imodel = models.index(model)
#                iexpt = expts.index(expt)
#                model_matrix[iyr, jj] = imodel
#                expt_matrix[iyr, jj] = iexpt
            if np.isfinite(corr_info[iyr, jj]) == True:
                model = corr_info[iyr, jj]
                if model == '': continue
                expt = corr_info[iyr, jj]
                imodel = models.index(model)
                iexpt = expts.index(expt)
                model_matrix[iyr, jj] = imodel
                expt_matrix[iyr, jj] = iexpt

    return model_matrix, expt_matrix, models, expts, expts_used

# =======================
# Function to centre the forecast about the obs
# =======================
def recentre_forecast(forecast_3d_in, num_mems_to_take, analogue_means, analogue_sds, target_time_series, window,
                      keep_mems=False, old_recentre_method=False, new_recentre_method=False, new_recentre_method2=False,
                      new_recentre_method2b=False, simple_recentre_method=False, simpleScaled_recentre_method=False,
                      clever_scaling_method=False, clever_scaling_methodb=False, clever_scaling_methodc=False,
                      nlead=11, do_ltbc=False, clever_scaling_methodd=False, map_method=False, map_method_nosd=False,
                      chosen_norm_window=False):
    if clever_scaling_methodd:
        raise ValueError('Not implemented fully yet')
    if not chosen_norm_window:
        chosen_norm_window = window
    nyrs = len(target_time_series)
    max_mems_to_take = 100
    if old_recentre_method:
        print("OLD")
        # Subtract t=0 from analogue to make an anomaly
        forecast_anomt0 = forecast_3d_in - np.repeat(forecast_3d_in[:, :, 0][:, :, None], nlead, axis=2)

        # Average together all these anomalies
        forecast_anomt0_mmm = np.ma.mean(forecast_anomt0[:, max_mems_to_take-num_mems_to_take:, :], axis=1)

        # Add back on the obs - so we really should hope to beat persistence now!
        forecast_anomt0_mmm_recentred = forecast_anomt0_mmm + np.repeat(target_time_series[:, None], nlead, axis=1)

        if keep_mems: return np.repeat(forecast_anomt0_mmm_recentred[:, np.newaxis, :], max_mems_to_take, axis=1)
        return forecast_anomt0_mmm_recentred
    elif new_recentre_method:
        print("NEW")
        # Average over all models - don't care about their biases as we will remove the overall bias at the end
        # NOTE: As different models make up each analogue, it is surprising that removing t=0 isn't a better method
        # but perhaps removing the models ANALOGUE PERIOD mean would work (not tried yet)
        forecast_mmm = np.ma.mean(forecast_3d_in[:, max_mems_to_take-num_mems_to_take:, :], axis=1)

        # Scale by the s.d. of the resultant MMM (non-adjusted) time series
        # NOTE!! This scaling doesn't remove a mean first, so is a bit weird (perhaps indefensible?)
        obs_sd = target_time_series.std()
        forecast_mmm_sd = np.ma.std(forecast_mmm[:, 0])
        forecast_mmm_norm = forecast_mmm * (obs_sd  / forecast_mmm_sd)

        # Here recentre by just adding the obs mean (not the t=0 val)
        forecast_mmm_norm_recentred = forecast_mmm_norm + (target_time_series.mean() - forecast_mmm_norm.mean())

        if keep_mems:
            return np.repeat(forecast_mmm_norm_recentred[:, np.newaxis, :], max_mems_to_take, axis=1)
        return forecast_mmm_norm_recentred
    elif new_recentre_method2 or new_recentre_method2b:
        # This method scales the analogue forecast for each year/model by the mean/sd of that same model
        # during the analogue creation window
        # Remove analogue period  mean
        forecast_anom = forecast_3d_in - np.repeat(analogue_means[:, :, None], nlead, axis=2)

        # Create sds and means for obs for each analogue period and make same shape as model array
        target_time_series_means_rehaped = np.ma.masked_all_like(forecast_anom)
        target_time_series_sds_rehaped = np.ma.masked_all_like(forecast_anom)
        for iyr_o in range(window, nyrs):
            target_time_series_means_rehaped[iyr_o-1, :, :]  = target_time_series[iyr_o-window:iyr_o].mean()
            target_time_series_sds_rehaped[iyr_o-1, :, :]  = target_time_series[iyr_o-window:iyr_o].std()

        # Scale each model anomaly forecast by this
        forecast_anom_norm = forecast_anom * (target_time_series_sds_rehaped / \
                                              np.repeat(analogue_sds[:, :, None], nlead, axis=2))

        # Re-add analogue period OBS mean BUT THIS MAKES IT WORSE (sometimes...)
        # But is probably a more justifiable choice...
        if new_recentre_method2:
            print("NEW2")
            forecast_anom_norm += target_time_series_means_rehaped

        # Re-add the analogue period ANALOGUE mean
        elif new_recentre_method2b:
            print("NEW2b")
            forecast_anom_norm += np.repeat(analogue_means[:, :, None], nlead, axis=2)

        # Re-add full obs period mean (has no effect but makes ts look better)
#         forecast_anom_norm += target_time_series.mean()

        if keep_mems: return forecast_anom_norm

        # Finally take the MMM
        forecast_anom_norm_mmm = np.ma.mean(forecast_anom_norm[:, max_mems_to_take-num_mems_to_take:, :], axis=1)

        # Recentre on the obs time series?
#         forecast_anom_norm_mmm_recentred = forecast_anom_norm_mmm + np.repeat(target_time_series[:, None], nlead, axis=1)
#         return forecast_anom_norm_mmm_recentred
#         if do_ltbc:
#             forecast_anom_norm_mmm = lead_time_bias_correction(forecast_anom_norm_mmm)
        return forecast_anom_norm_mmm
    elif simple_recentre_method:
        print("SIMPLE")
        # This method just averages together the raw forecast data, so we could potentially use the
        # same models to create an "analysis".
        if keep_mems: return forecast_3d_in
        forecast_mmm = np.ma.mean(forecast_3d_in[:, max_mems_to_take-num_mems_to_take:, :], axis=1)
        if do_ltbc:
            forecast_mmm = lead_time_bias_correction(forecast_mmm)
        return forecast_mmm
    elif simpleScaled_recentre_method:
        print("SIMPLESCALED")
        # This method just averages together the raw forecast data, but also scales by the OBS SST variance
        forecast_mmm = np.ma.mean(forecast_3d_in[:, max_mems_to_take-num_mems_to_take:, :], axis=1)

        # Scale by the OBS s.d. (have to remove the mean first, before re-adding it)
        obs_sd = target_time_series.std()
        forecast_mmm_sd = np.ma.std(forecast_mmm[:, 0])
        forecast_mmm_anom = forecast_mmm - forecast_mmm.mean()
        forecast_mmm_norm = forecast_mmm_anom * (obs_sd  / forecast_mmm_sd) + forecast_mmm.mean()

        if keep_mems: return np.repeat(forecast_mmm_norm[:, np.newaxis, :], max_mems_to_take, axis=1)

        if do_ltbc:
            forecast_mmm_norm = lead_time_bias_correction(forecast_mmm_norm)
        return forecast_mmm_norm
    elif clever_scaling_method or clever_scaling_methodb or clever_scaling_methodc or clever_scaling_methodd:
        # This method scales the analogue forecast for each year/model by the mean/sd of that same model
        # during the analogue creation window, AND THEN scales the resulting ensemble mean forecast by
        # the s.d. of the ensemble mean in the prior obs period
        # Remove analogue period  mean
        print("Clever scaling")
        forecast_anom = forecast_3d_in - np.repeat(analogue_means[:, :, None], nlead, axis=2)

        # Create sds and means for obs for each analogue period and make same shape as model array
        target_time_series_means_rehaped = np.ma.masked_all_like(forecast_anom)
        target_time_series_sds_rehaped = np.ma.masked_all_like(forecast_anom)
        for iyr_o in range(window, nyrs):
            target_time_series_means_rehaped[iyr_o-1, :, :]  = target_time_series[iyr_o-window:iyr_o].mean()
            target_time_series_sds_rehaped[iyr_o-1, :, :]  = target_time_series[iyr_o-window:iyr_o].std()

        # Scale each model anomaly forecast by this. Don't really like this as the scaling is centred
        # about the REAL mean of the bit being used. Instead it used the offset from analogue_mean
        if not clever_scaling_methodd:
            forecast_anom_norm = forecast_anom * (target_time_series_sds_rehaped / \
                                                  np.repeat(analogue_sds[:, :, None], nlead, axis=2))
        else:
            forecast_anom_norm = forecast_anom

        # Now take the MMM of the FORECAST
        print(forecast_anom_norm[:, 14, 2])
        forecast_anom_norm_mmm = np.nanmean(forecast_anom_norm[:, max_mems_to_take-num_mems_to_take:, :], axis=1)
        print(forecast_anom_norm_mmm[:, 2])

        # Now scale AGAIN by the obs. Note I divide by the (future) forecast here when it would be better
        # to divide by the same models etc but in the analogue period (i.e. forecast backwards for the
        # previous "window" years). Not sure if the same goes for the mean that I have removed - presumably
        # this needs to be centred on zero. In either case, I am still multiplying by the PAST obs (i.e. target)
        forecast_sds = np.repeat(np.ma.std(forecast_anom_norm_mmm, axis=1)[:, np.newaxis], nlead, axis=1)
        forecast_means = np.repeat(np.ma.mean(forecast_anom_norm_mmm, axis=1)[:, np.newaxis], nlead, axis=1)
        forecast_anom_norm_mmm_norm = (forecast_anom_norm_mmm - forecast_means) * (target_time_series_sds_rehaped[:, 0, :]) / forecast_sds
        forecast_anom_norm_mmm_norm += forecast_means  # yes.

        # Re-add analogue period OBS mean BUT THIS MAKES IT WORSE (sometimes...)
        # But is probably a more justifiable choice...
        if clever_scaling_method:
            print("CLEVER")
            forecast_anom_norm_mmm_norm += target_time_series_means_rehaped[:, 0, :]

        # Re-add the analogue period ANALOGUE mean
        elif clever_scaling_methodb:
            print("CLEVERb")
            forecast_anom_norm_mmm_norm += np.repeat(np.ma.mean(analogue_means[:, max_mems_to_take-num_mems_to_take:],
                                                                axis=1)[:, None], nlead, axis=1)

        # Re-add the actual OBS (i.e. forecast var) at t=0 (like if we were initialising from "truth")
        # and remove the forecast at t=0
        elif clever_scaling_methodc or clever_scaling_methodd:
            forecast_anom_norm_mmm_norm += (target_time_series - forecast_anom_norm_mmm_norm[:, 0])[:, None]

        # Re-add full obs period mean (has no effect but makes ts look better)
#         forecast_anom_norm += target_time_series.mean()

        if keep_mems: return np.repeat(forecast_anom_norm_mmm_norm[:, np.newaxis, :], max_mems_to_take, axis=1)

        # Recentre on the obs time series?
#         forecast_anom_norm_mmm_recentred = forecast_anom_norm_mmm + np.repeat(target_time_series[:, None], nlead, axis=1)
#         return forecast_anom_norm_mmm_recentred
#         if do_ltbc:
#             forecast_anom_norm_mmm = lead_time_bias_correction(forecast_anom_norm_mmm)
        return forecast_anom_norm_mmm_norm
    elif map_method or map_method_nosd:
        if map_method:
            print("MAP")
        elif map_method_nosd:
            print("MAP_NOSD")
        forecast_anom = forecast_3d_in - np.repeat(analogue_means[:, :, None], nlead, axis=2)

        # Create sds and means for obs for each analogue period and make same shape as model array
        target_time_series_means_rehaped = np.ma.masked_all_like(forecast_anom)
        target_time_series_sds_rehaped = np.ma.masked_all_like(forecast_anom)
        for iyr_o in range(window, nyrs):
            target_time_series_means_rehaped[iyr_o-1, :, :]  = target_time_series[iyr_o-chosen_norm_window:iyr_o].mean()
            target_time_series_sds_rehaped[iyr_o-1, :, :]  = target_time_series[iyr_o-chosen_norm_window:iyr_o].std()

        if map_method:
            forecast_anom_norm = forecast_anom * (target_time_series_sds_rehaped / \
                                                  np.repeat(analogue_sds[:, :, None], nlead, axis=2))
            forecast_anom_norm += target_time_series_means_rehaped
        elif map_method_nosd:
            forecast_anom_norm = forecast_anom + target_time_series_means_rehaped

        if keep_mems: return forecast_anom_norm

        forecast_anom_norm_mmm = np.ma.mean(forecast_anom_norm[:, max_mems_to_take-num_mems_to_take:, :], axis=1)
        return forecast_anom_norm_mmm

# =======================
# Function to calculate the skill
# =======================
def calculate_skill(forecast_in2, nlead, target_time_series, target_years, multi=False, start_lead=[1], end_lead=[5], since1960=False,
                    tol=0., target=None, before1990=False):
    forecast_in = forecast_in2.copy()
    this_target = target_time_series.copy()
    nyrs = len(this_target)
    lead_times = np.arange(nlead)
    # Calculate the skill, either for each validity time (annual mean)
    # or for multi-annual means, using the following pattern:
    # 1-5, 2-6, 3-7, 4-8, 5-9, 6-10, 1-10, 2-10
    if since1960:
        offset1960 = np.argwhere(target_years == 1960)[0][0]
    else:
        offset1960 = 0
    if before1990:
        offset1990 = np.argwhere(target_years == 1991)[0][0]
    else:
        offset1990 = nyrs

    # # num_mems_to_take: This means we are provided with raw data that needs recentring first
    # if num_mems_to_take is not None:
    #     # Then we first need to make the forecasts anomalies wrt themselves at at t=0, then average the
    #     # forecast array over the number of members and then add on the obs t=0
    #     forecast_in = recentre_forecast(forecast_in, num_mems_to_take, means, sds)

    if forecast_in.ndim == 1:  # Inflate the dimensions in order to use same code
        forecast_in = np.repeat(forecast_in[:, None], nlead, axis=1)

    if not multi:
        forecast_skill = np.ma.masked_all(nlead)
        for ilead in lead_times:
            this_target_ts = this_target[offset1960+ilead:offset1990]
            this_source_ts = forecast_in[offset1960:offset1990-ilead,  ilead]
            real = np.nonzero(this_target_ts *  this_source_ts)
            if ((len(real[0]) / np.float(len(this_target_ts))) < tol) or (len(real[0]) == 0):
                continue
            _, _, corr, _, _ = stats.linregress(this_source_ts[real], this_target_ts[real])
            forecast_skill[ilead] = corr
    elif multi:
        forecast_skill = np.ma.masked_all(len(start_lead))
        for iforecast, (ss, ee) in enumerate(zip(start_lead, end_lead)):
            nleads = (ee + 1 - ss)
            this_target_ts = np.zeros(shape=(offset1990 - ee - offset1960))
            for ilead in range(ss, ee+1, 1):  # +1 to include the end time
                this_target_ts += this_target[offset1960+ilead:offset1990-(ee-ilead)]
            this_target_ts /= nleads

            this_source_ts = np.ma.mean(forecast_in[offset1960:offset1990-ee, ss:ee+1], axis=1)

            real = np.nonzero(this_target_ts *  this_source_ts)
            if ((len(real[0]) / np.float(len(this_target_ts))) < tol) or (len(real[0]) == 0):
                continue
            _, _, corr, _, _ = stats.linregress(this_source_ts[real], this_target_ts[real])
            forecast_skill[iforecast] = corr

    return forecast_skill

def calculate_skill3d(forecast_in, nlead, this_target, target_years, multi=False, start_lead=[1], end_lead=[5], since1960=False,
                      tol=0., target=None, before1990=False, testing=False):
    nyrs = len(target_years)
    lead_times = np.arange(nlead)
    nyrsb, nleadb, nj, ni = forecast_in.shape
    assert nlead == nleadb
    assert nyrs == nyrsb
    # Calculate the skill, either for each validity time (annual mean)
    # or for multi-annual means, using the following pattern:
    # 1-5, 2-6, 3-7, 4-8, 5-9, 6-10, 1-10, 2-10
    if since1960:
        offset1960 = np.argwhere(target_years == 1960)[0][0]
    else:
        offset1960 = 0
    if before1990:
        offset1990 = np.argwhere(target_years == 1991)[0][0]
    else:
        offset1990 = nyrs

    if not multi:
        forecast_skill = np.ma.masked_all(shape=(nlead, nj, ni))
        for ilead in lead_times:
            for jj in range(nj):
                print(jj, nj)
                for ii in range(ni):
                    if testing:
                        if jj < 100: continue
                        if jj > 150: continue
                        if ii < 110: continue
                        if ii > 160: continue
                    this_target_ts = this_target[offset1960+ilead:offset1990, jj, ii]
                    this_source_ts = forecast_in[offset1960:offset1990-ilead,  ilead, jj, ii]
                    real = np.nonzero(this_target_ts *  this_source_ts)
                    if ((len(real[0]) / np.float(len(this_target_ts))) < tol) or (len(real[0]) == 0):
                        continue
                    _, _, corr, _, _ = stats.linregress(this_source_ts[real], this_target_ts[real])
                    forecast_skill[ilead, jj, ii] = corr
    elif multi:
        forecast_skill = np.ma.masked_all(shape=(len(start_lead), nj, ni))
        for iforecast, (ss, ee) in enumerate(zip(start_lead, end_lead)):
            nleads = (ee + 1 - ss)
            for jj in range(nj):
                print(jj, nj)
                for ii in range(ni):
                    if testing:
                        if jj < 100: continue
                        if jj > 150: continue
                        if ii < 110: continue
                        if ii > 160: continue
                    this_target_ts = np.zeros(shape=(offset1990 - ee - offset1960))
                    for ilead in range(ss, ee+1, 1):  # +1 to include the end time
                        this_target_ts += this_target[offset1960+ilead:offset1990-(ee-ilead), jj, ii]
                    this_target_ts /= nleads

                    this_source_ts = np.ma.mean(forecast_in[offset1960:offset1990-ee, ss:ee+1, jj, ii], axis=1)

                    real = np.nonzero(this_target_ts *  this_source_ts)
                    if ((len(real[0]) / np.float(len(this_target_ts))) < tol) or (len(real[0]) == 0):
                        continue
                    _, _, corr, _, _ = stats.linregress(this_source_ts[real], this_target_ts[real])
                    forecast_skill[iforecast, jj, ii] = corr
    return forecast_skill

# =======================
# Make Doug's residual skill measure (subtract the forced signal)
# =======================
def calculate_residual(forecast2d_in, nlead, hist_ts_in):
    dim_inflated = False
    if forecast2d_in.ndim == 1:
#         print("Inflating dimensions"
        forecast2d_in = forecast2d_in[:, None]
        dim_inflated = True
    assert forecast2d_in.shape[0] == hist_ts_in.shape[0]
    nyrs = len(hist_ts_in)

    forecast_res = np.ma.masked_all_like(forecast2d_in)
    for jj in range(nlead):
        # Same validity times
        forecast_ts = forecast2d_in[:nyrs-jj, jj] - np.ma.mean(forecast2d_in[:nyrs-jj, jj])
        hist_ts = hist_ts_in[jj:] - np.ma.mean(hist_ts_in[jj:])

        real = np.nonzero(hist_ts * forecast_ts)[0]
        if len(real) < 1: continue
        grad, inte, corr, _, _ = stats.linregress(forecast_ts[real], hist_ts[real])
        std_hist = np.ma.std(hist_ts[real])
        std_forecast = np.ma.std(forecast_ts[real])
        coeffs = corr * (std_forecast / std_hist)
        regression = coeffs * hist_ts
        residual = forecast_ts  - regression
        forecast_res[:nyrs-jj, jj] = residual

    if dim_inflated: forecast_res = forecast_res[:, 0]
    return forecast_res

def calculate_residual3d(forecast4d_in, nlead, hist3d_in):
    #residual_dir = '/data/mmenary/python_saves/'
    residual_dir = '/work/scratch-nopw/mmenary/PreCalc'
    ida = hashlib.md5(forecast4d_in).hexdigest()
    idb = hashlib.md5(np.array(nlead)).hexdigest()
    idc = hashlib.md5(hist3d_in).hexdigest()
    ids = int(ida, 16) + int(idb, 16) + int(idc, 16)
    residual_file = residual_dir + '/PreCalc_Residual_{:d}.pkl'.format(ids)
    if os.path.isfile(residual_file):
        with open(residual_file, 'rb') as handle:
            print("Loading pre-calculated residual file: {:s}".format(residual_file))
            forecast_res = pickle.load(handle)
            return forecast_res
    else:
        print("Creating and saving residual file: {:s}".format(residual_file))

    dim_inflated = False
    if forecast4d_in.ndim == 3:
        print("Inflating dimensions [year, lead, jj, ii]")
        forecast4d_in = forecast4d_in[:, np.newaxis,  :, :]
        dim_inflated = True
    assert forecast4d_in.shape[0] == hist3d_in.shape[0]
    nyrs, nj, ni = hist3d_in.shape

    forecast_res = np.ma.masked_all_like(forecast4d_in)
    for jj in range(nj):
        # if jj != 90: continue ################################################################################
        for ii in range(ni):
            forecast2d_in = forecast4d_in[:, :, jj, ii]
            hist_ts_in = hist3d_in[:, jj, ii]
            if forecast2d_in.mask.all() or hist_ts_in.mask.all():
                continue

            for ilead in range(nlead):
                # Same validity times
                forecast_ts = forecast2d_in[:nyrs-ilead, ilead] - np.ma.mean(forecast2d_in[:nyrs-ilead, ilead])
                hist_ts = hist_ts_in[ilead:] - np.ma.mean(hist_ts_in[ilead:])

                real = np.nonzero(hist_ts * forecast_ts)[0]
                if len(real) < 1: continue
                grad, inte, corr, _, _ = stats.linregress(forecast_ts[real], hist_ts[real])
                std_hist = np.ma.std(hist_ts[real])
                std_forecast = np.ma.std(forecast_ts[real])
                coeffs = corr * (std_forecast / std_hist)
                regression = coeffs * hist_ts
                residual = forecast_ts  - regression
                forecast_res[:nyrs-ilead, ilead, jj, ii] = residual

    if dim_inflated: forecast_res = forecast_res[:, 0, :, :]

    if os.path.isdir(residual_dir):
        with open(residual_file, 'wb') as handle:
            pickle.dump(forecast_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return forecast_res

def simple_running_skill(forecast_recentred, running_skill_window, target_time_series_in, ilead=[1]):
    if not isinstance(ilead, list):
        ilead = [ilead]
    tol = 0.9
    nyrs, nleads = forecast_recentred.shape
    if len(ilead) == 1:
        nt = nyrs - ilead[0]
        forecast_recentred_single_leadtime = forecast_recentred[:nt, ilead[0]]
        target_time_series = target_time_series_in[ilead[0]:]
    elif len(ilead) == 2:
        nt = nyrs - ilead[1]
        forecast_recentred_single_leadtime = forecast_recentred[:nt, ilead[0]:ilead[1]+1].mean(axis=1)
        target_time_series = np.ma.masked_all(shape=(nt))
        for tt in range(nt):
            target_time_series[tt] = target_time_series_in[tt+ilead[0]:tt+ilead[1]+1].mean()
    else:
        raise ValueError('ilead must have 1 or 2 values only')

    running_skill = np.ma.masked_all(shape=nyrs)
    for iyr in range(nyrs):
        if (iyr-(running_skill_window-1)) < 0:
            continue
        if (ilead[0]+iyr) > (nyrs-1):
            continue
        this_source_ts = forecast_recentred_single_leadtime[iyr-(running_skill_window-1):iyr+1]
        this_target_ts = target_time_series[iyr-(running_skill_window-1):iyr+1]  # Same validity times
        real = np.nonzero(this_source_ts * this_target_ts)
        if ((len(real[0]) / np.float(len(this_target_ts))) < tol) or (len(real[0]) == 0):
            continue
        _, _, corr, _, _ = stats.linregress(this_source_ts[real], this_target_ts[real])
        running_skill[ilead[0]+iyr] = corr
    return running_skill

def lead_time_bias_correction(in_arr):
    ndim = in_arr.ndim
    if ndim == 2:  # MMM [year, lead_time]
        print('Bias correcting MMM')
        lead_time_bias = np.ma.mean(in_arr, axis=0)
        lead_time_bias -= lead_time_bias.mean()
        bias_corrected = in_arr - lead_time_bias[None, :]  # Broadcast these together
    elif ndim == 3:  # [model, year, lead_time]
        print('Bias correcting all models')
        lead_time_bias = np.ma.mean(in_arr, axis=1)
        lead_time_bias -= np.ma.mean(lead_time_bias, axis=1, keepdims=True)
        bias_corrected = in_arr - lead_time_bias[:, None, :]  # Broadcast these together
    return bias_corrected

def plot_baseline_data(baseline_file, multi=False, lead_times_multi=False):
    with open(baseline_file, 'rb') as handle:
        print('Reading from {:s}'.format(baseline_file))
        baseline_data = pickle.load(handle)
        hist_models = baseline_data['hist_models']
        hind_models = baseline_data['hind_models']
        historical_mmm = baseline_data['historical_mmm']
        historical_anom_ensmn_corr = baseline_data['historical_anom_ensmn_corr']
        historical_anom_ensmn_mmm_corr = baseline_data['historical_anom_ensmn_mmm_corr']
        historical_anom_ensmn_mmm1960_corr = baseline_data['historical_anom_ensmn_mmm1960_corr']
        historical_anom_ensmn_mmm1990_corr = baseline_data['historical_anom_ensmn_mmm1990_corr']
        hindcast_time_series_ensmn_mmm_corr = baseline_data['hindcast_time_series_ensmn_mmm_corr']
        hindcast_time_series_ensmn_ltbc_mmm_corr = baseline_data['hindcast_time_series_ensmn_ltbc_mmm_corr']
        historical_anom_inf_ensmn_mmm_corr = baseline_data['historical_anom_inf_ensmn_mmm_corr']
        persistence_corr = baseline_data['persistence_corr']
        persistence_res_corr = baseline_data['persistence_res_corr']
        persistence_smoothed_corr = baseline_data['persistence_smoothed_corr']
        persistence1960_corr = baseline_data['persistence1960_corr']
        persistence1990_corr = baseline_data['persistence1990_corr']
        historical_anom_ensmn_multicorr = baseline_data['historical_anom_ensmn_multicorr']
        historical_anom_ensmn_mmm_multicorr = baseline_data['historical_anom_ensmn_mmm_multicorr']
        historical_anom_ensmn_mmm1960_multicorr = baseline_data['historical_anom_ensmn_mmm1960_multicorr']
        # historical_anom_ensmn_mmm1990_multicorr = baseline_data['historical_anom_ensmn_mmm1990_multicorr']
        hindcast_time_series_ensmn_mmm_multicorr = baseline_data['hindcast_time_series_ensmn_mmm_multicorr']
        hindcast_time_series_ensmn_ltbc_mmm_multicorr = baseline_data['hindcast_time_series_ensmn_ltbc_mmm_multicorr']
        historical_anom_inf_ensmn_mmm_multicorr = baseline_data['historical_anom_inf_ensmn_mmm_multicorr']
        persistence_multicorr = baseline_data['persistence_multicorr']
        persistence_res_multicorr = baseline_data['persistence_res_multicorr']
        persistence_smoothed_multicorr = baseline_data['persistence_smoothed_multicorr']
        persistence1960_multicorr = baseline_data['persistence1960_multicorr']
        hindcast_time_series_ensmn_mmm_res_ltbc_corr = baseline_data['hindcast_time_series_ensmn_mmm_res_ltbc_corr']
        hindcast_time_series_ensmn_mmm_res_ltbc_multicorr = baseline_data['hindcast_time_series_ensmn_mmm_res_ltbc_multicorr']
        mpi_ensmn_corr = baseline_data['mpi_ensmn_corr']
        mpi_ensmn1960_corr = baseline_data['mpi_ensmn1960_corr']
        mpi_ensmn1990_corr = baseline_data['mpi_ensmn1990_corr']
        mpi_ensmn_multicorr = baseline_data['mpi_ensmn_multicorr']
        mpi_ensmn1960_multicorr = baseline_data['mpi_ensmn1960_multicorr']
        mpi_ensmn1990_multicorr = baseline_data['mpi_ensmn1990_multicorr']
        # if (baseline_forecast_var == 'SST') and (var == 'DepthAverageT'):  # Another HACK to make sure the dates are the same
        #     historical_mmm = historical_mmm[30:]
        #     historical_mmm = np.concatenate((historical_mmm, np.ma.masked_all(shape=1)))

        if multi:
            index_sets = [[0, 1, 2, 3, 4, 5], [6, 7]]
            for ii, ind in enumerate(index_sets):
                plt.plot(lead_times_multi[ind], historical_anom_ensmn_mmm_multicorr[ind], color='green')
                plt.plot(lead_times_multi[ind], historical_anom_ensmn_mmm1960_multicorr[ind], color='green', linestyle='--')
                # plt.plot(lead_times_multi[ind], historical_anom_ensmn_mmm1990_multicorr[ind], color='green', linestyle='-.')
                plt.plot(lead_times_multi[ind], hindcast_time_series_ensmn_mmm_multicorr[ind], linestyle='--', color='indigo')
                plt.plot(lead_times_multi[ind], hindcast_time_series_ensmn_mmm_res_ltbc_multicorr[ind], linestyle=':', color='indigo')
                plt.plot(lead_times_multi[ind], hindcast_time_series_ensmn_ltbc_mmm_multicorr[ind], linestyle='--', color='indigo')
                plt.plot(lead_times_multi[ind], persistence_multicorr[ind], color='k')
                plt.plot(lead_times_multi[ind], persistence_res_multicorr[ind], color='k', linestyle=':')
                plt.plot(lead_times_multi[ind], persistence1960_multicorr[ind], color='k', linestyle='--')
                # plt.plot(lead_times_multi[ind], persistence1990_multicorr[ind], color='k', linestyle='-.')
                plt.plot(lead_times_multi[ind], mpi_ensmn_multicorr[ind], color='blue')
                plt.plot(lead_times_multi[ind], mpi_ensmn1960_multicorr[ind], color='blue', linestyle='--')
                plt.plot(lead_times_multi[ind], mpi_ensmn1990_multicorr[ind], color='blue', linestyle='-.')
        else:
            plt.plot(baseline_data['lead_times'], historical_anom_ensmn_mmm_corr, color='green', label='Persistence (MMM, hist)')
            plt.plot(baseline_data['lead_times'], historical_anom_ensmn_mmm1960_corr, color='green', linestyle='--', label='Persistence since 1960 (MMM, hist)')
            plt.plot(baseline_data['lead_times'], historical_anom_ensmn_mmm1990_corr, color='green', linestyle='-.', label='Persistence before 1990 (MMM, hist)')
#                     plt.plot(baseline_data['lead_times'], np.repeat(historical_anom_ensmn_mmm_corr[0], len(lead_times)), color='green', linestyle='--')
            plt.plot(baseline_data['lead_times'], persistence_corr, color='k', label='Persistence (obs)')
            plt.plot(baseline_data['lead_times'], persistence_res_corr, color='k', linestyle=':', label='Persistence RESIDUAL (obs)')
            plt.plot(baseline_data['lead_times'], persistence1960_corr, color='k', linestyle='--', label='Persistence since 1960 (obs)')
            plt.plot(baseline_data['lead_times'], persistence1990_corr, color='k', linestyle='-.', label='Persistence before 1990 (obs)')
            # plt.plot(baseline_data['lead_times'], persistence_smoothed_corr, color='k', linestyle=':', label='Persistence (smoothed) (obs)')
            plt.plot(baseline_data['lead_times'], hindcast_time_series_ensmn_mmm_corr, color='indigo', linestyle='--', label='Hindcasts (MMM)')
            plt.plot(baseline_data['lead_times'], hindcast_time_series_ensmn_mmm_res_ltbc_corr, color='indigo', linestyle=':', label='Hindcasts RESIDUAL (MMM)')
            plt.plot(baseline_data['lead_times'], hindcast_time_series_ensmn_ltbc_mmm_corr, linestyle='--', color='indigo', label='Hindcasts (LTBC, MMM)')
            plt.plot(baseline_data['lead_times'], mpi_ensmn_corr, color='blue', label='MPI SPECS')
            plt.plot(baseline_data['lead_times'], mpi_ensmn1960_corr, color='blue', linestyle='--', label='MPI SPECS since 1960')
            plt.plot(baseline_data['lead_times'], mpi_ensmn1990_corr, color='blue', linestyle='-.', label='MPI SPECS before 1990')

def skill_map(in_arr, target_arr, years, after1960=False, before1990=False):
    nyrs = len(years)

    if in_arr.ndim == 3:
        print(" ++ Inflating in_arr dimensions from 3 to 4 by adding LEAD dimension...")
        in_arr = in_arr[:, np.newaxis, :, :]

    assert in_arr.ndim  == 4  # year, lead, jj, ii
    assert target_arr.ndim  == 3  # year, jj, ii

    nyrs2, nlead, nj, ni = in_arr.shape
    assert nyrs == nyrs2

    skill_expanded = np.ma.masked_all(shape=(nlead, nj, ni))
    if after1960:
        offset1960 = np.argwhere(years == 1960)[0][0]
        skill_expanded1960 = np.ma.masked_all(shape=(nlead, nj, ni))
    if before1990:
        offset1990 = np.argwhere(years == 1990)[0][0] - nyrs
        skill_expanded1990 = np.ma.masked_all(shape=(nlead, nj, ni))

    for jj in range(nj):
        # if jj != 90: continue ################################################################################
        print(jj, nj)
        for ii in range(ni):
            obs_ts = target_arr[:, jj, ii]  # year
            forecast_ts = in_arr[:, :, jj, ii]  # year, lead

            if obs_ts.mask.all() or forecast_ts.mask.all():
                continue

            for ilead in range(nlead):
                # Ensure same validity times
                ntimes = nyrs - ilead
                obs_ts2 = obs_ts[ilead:]
                forecast_ts2 = forecast_ts[:ntimes, ilead]
                real = np.nonzero(obs_ts2 * forecast_ts2)[0]
                if len(real) < 5: continue
                _, _, corr, _, _ = stats.linregress(obs_ts2[real], forecast_ts2[real])
                skill_expanded[ilead, jj, ii] = corr

                if after1960:
                    obs_ts2 = obs_ts[ilead+offset1960:]
                    forecast_ts2 = forecast_ts[offset1960:ntimes, ilead]
                    real = np.nonzero(obs_ts2 * forecast_ts2)[0]
                    if len(real) < 5: continue
                    _, _, corr, _, _ = stats.linregress(obs_ts2[real], forecast_ts2[real])
                    skill_expanded1960[ilead, jj, ii] = corr

                if before1990:
                    obs_ts2 = obs_ts[ilead:offset1990]
                    forecast_ts2 = forecast_ts[:ntimes+offset1990, ilead]
                    real = np.nonzero(obs_ts2 * forecast_ts2)[0]
                    if len(real) < 5: continue
                    _, _, corr, _, _ = stats.linregress(obs_ts2[real], forecast_ts2[real])
                    skill_expanded1990[ilead, jj, ii] = corr

    if after1960 and not before1990:
        return [skill_expanded, skill_expanded1960]
    elif not after1960 and before1990:
        return [skill_expanded, skill_expanded1990]
    elif after1960 and before1990:
        return [skill_expanded, skill_expanded1960, skill_expanded1990]
    else:
        return skill_expanded
