from scipy import signal
import numpy as np

#period = [4, 50]  # Bandpass period range
#f1 = 1. / period[1]  # Convert into frequency
#f2 = 1. / period[0]
#samp_freq = 1. # No. of samples per "second"

# =========================
# This butter_bandpass code is taken from here:
# http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
# It gives similar results to conducting the same operation by just smoothing
# =========================
def butter_bandpass_hidden(lowcut, highcut, fs, order=5):
    """
    Used by butter_bandpass_filter - not intended to be called
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data_in, lowcut, highcut, fs=1., order=5):
    """
    Reads in a 1D array and the low and high frequency cutoffs.
    These are just 1/Period (reverse the order). The "fs" is the
    sampling frequency which is likely to be 1.
    """
    data = data_in - data_in.mean()
    b, a = butter_bandpass_hidden(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# =========================
# This is similar top the above but for just a highpass filter. Note that
# using a moving average/smoothing/boxcar (synonymous) is a good smoother
# but a BAD filter, which may be relevant if you're doing power spectra or
# something with the data afterwards
# =========================
def butter_highpass_hidden(cutoff, fs, order=5):
    """
    Used by butter_highpass_filter - not intended to be called
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high')
    return b, a

def butter_highpass_filter(data, cutoff, fs=1., order=5):
    """
    Reads in a 1D array and the frequency cutoff.
    The "fs" is the sampling frequency which is likely to be 1.
    """
    b, a = butter_highpass_hidden(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def smooth1d(data, smooth_len, mask_ends=False, verbose=False, cut_ends=False):
    """
    Reads in a 1D array and the lengeth of a smoothing window and
    returns a smoothed version of the array
    """
    weights = np.ones((smooth_len,)) / (smooth_len)
    smoothed = np.convolve(data, weights, mode='same')

    if not isinstance(smoothed, np.ma.MaskedArray) and isinstance(data, np.ma.MaskedArray):
        if verbose: print("Re-adding mask")
        smoothed = np.ma.array(smoothed, mask=False)
        smoothed = np.ma.array(smoothed, mask=data.mask)  # If I don't do this then the mask might end up being 1 element and weird...
        # print type(smoothed)

    if mask_ends == True or cut_ends == True:
        if not isinstance(smoothed, np.ma.MaskedArray):
            if verbose: print("Adding a mask")
            smoothed = np.ma.array(smoothed, mask=False)
        for ii in range((np.ceil(smooth_len/2.)).astype('int')):
            edges = np.ma.notmasked_edges(smoothed)
            for edge in edges:
                smoothed.mask[edge] = True

    if cut_ends == True:
        edges = np.ma.notmasked_edges(smoothed)
        smoothed = smoothed[edges[0]:edges[1]+1]

    return smoothed

def smooth(data, smooth_len, mask_ends=False):
    """
    Reads in an arbitrarily shaped array and the length of a smoothing window and
    returns a smoothed version of the array - smoothing in all dimensions!.
    If mask_ends is set then it will make the output a masked array and mask the bits at either end
    TODO: I could make smooth_len either 0D then do the current method or nD and
    then use that as the shape - though I'd have to test if the kernel does anything
    weird in that case and also normalise appropriately, presumably each dimension separately
    """

    print("I'm not sure how this behaves for 3+ dimensions. For 1D, best to use smooth1d for now")
    raise KeyboardError

    shape = [smooth_len] + [1] * (data.ndim - 1)
    kernel = np.ones(shape)
    kernel /= kernel.sum()

    smoothed = signal.convolve(data, kernel, mode='same')

    if not isinstance(smoothed, np.ma.MaskedArray) and isinstance(data, np.ma.MaskedArray):
        print("Re-adding mask")
        smoothed = np.ma.array(smoothed, mask=data.mask)

    if mask_ends == True:
        print("Currently does nothing")
        #np.ma.notmasked_edges(smoothed)
        #smoothed = np.ma.array(smoothed, mask=smoothed

    return smoothed

def hp_filtered_by_smoothing(input_arr, low_period):
    """
    Reads in an arbitrary shape array and filters it by removing long
    period variability captured by a simple low period smoothing.
    Assumes the time axis is the first axis
    """

    shape = [low_period / 2] + [1] * (input_arr.ndim - 1)
    kernel = np.ones(shape)
    kernel /= kernel.sum()

    filtered_by_smoothing_highp = signal.convolve(input_arr, kernel, mode='same')
    filtered_by_smoothing = input_arr - filtered_by_smoothing_highp

    filtered_mean = np.mean(filtered_by_smoothing, axis=0, keepdims=True)
    input_mean = np.mean(input_arr, axis=0, keepdims=True)

    filtered_by_smoothing += (input_mean - filtered_mean)
    return filtered_by_smoothing
