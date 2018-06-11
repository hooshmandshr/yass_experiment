import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from scipy.spatial.distance import pdist, squareform

from yass.evaluate.util import *

def align_template(template, temp_len=40, mode='all'):
    window = np.arange(0, temp_len) - temp_len // 2
    n_chan = template.shape[1]
    main_chan = main_channels(template)[-1]
    base_trace = np.zeros(template.shape[0])
    base_trace[:] = template[:, main_chan]
    temp_norm = np.sum(template * template, axis=0)
    base_norm = temp_norm[main_chan]
    aligned_temp = np.zeros([temp_len, n_chan])
    
    if mode == 'neg':
        base_trace[base_trace > 0] = 0

    for c in range(n_chan):
        orig_filt = template[:, c]
        filt = np.zeros(orig_filt.shape)
        filt[:] = orig_filt
        if mode == 'neg':
            filt[filt > 0] = 0
        filt_norm = temp_norm[c]
        conv_dist = -2 * np.convolve(filt, np.flip(base_trace, axis=0), mode='same') + base_norm + filt_norm
        center = np.argmin(conv_dist)
        try:
            aligned_temp[:, c] = orig_filt[center + window]
        except:
            aligned_temp[:, c] = orig_filt[np.arange(0, temp_len)]
    return aligned_temp


def recon(template, rank=3):
    """SVD reconstruction of a template."""
    u, s, vh = np.linalg.svd(template)
    return np.matmul(u[:, :rank] * s[:rank], vh[:rank, :])

def recon_error(template, rank=3):
    """Reconstruction error of SVD with given rank."""
    temp_rec = recon(template, rank=rank)
    return np.linalg.norm((template - temp_rec))

class Geometry(object):
    """Geometry Object for finidng closest channels."""
    def __init__(self, geometry):
        self.geom = geometry
        self.pdist = squareform(pdist(geometry))

    def neighbors(self, channel, size):
        return np.argsort(self.pdist[channel, :])[:size]


def vis_chan(template, min_peak_to_peak=1):
    """Visible channels on a standardized template with given threshold."""
    return np.max(template, axis=0) - np.min(template, axis=0) > min_peak_to_peak


def conv_dist(ref, temp):
    """l2 distance of temp with all windows of ref."""
    return np.sum(ref * ref) - 2 * np.convolve(ref, np.flip(temp, axis=0), mode='valid') + np.sum(temp * temp)


def align_temp_to_temp(ref, temp):
    """Aligns temp with bigger window to ref with smaller window."""
    n_chan = ref.shape[1]
    shifts = np.zeros(n_chan)
    for c in range(n_chan):
        shifts[c] = np.argmin(conv_dist(temp[:, c], ref[:, c]))
        #plt.plot(conv_dist(temp[:, c], ref[:, c]))
    return shifts


def optimal_aligned_compress(template, upsample=5, rank=3, max_shift=6):
    """Greedy local search of alignments for best SVD compression error."""
    upsample = 5
    max_shift = max_shift * upsample
    half_max_shift = max_shift // 2
    
    n_chan = template.shape[1]
    n_times = template.shape[0]
    template = sp.signal.resample(template, n_times * upsample)
    new_times = upsample * n_times

    snip_win = (half_max_shift, -half_max_shift)
    snip_temp = copy.copy(template[snip_win[0]:snip_win[1], :])
    shifts = np.zeros(n_chan, dtype='int')

    #
    obj = recon_error(snip_temp, rank=rank)
    obj_list = []
    for i, k in enumerate(reversed(main_channels(template))):
        if i == 0:
            # main channel do nothing
            continue
        #cand_chan = np.random.randint(0, n_chan)
        cand_chan = k
        # obj of jitter -1, 0, 0 respectively
        new_obj = np.zeros(max_shift + 1)
        for j, jitter in enumerate(range(-half_max_shift, half_max_shift + 1)):
            snip_from, snip_to = snip_win[0] + jitter, snip_win[1] + jitter
            if snip_to == 0:
                snip_to = new_times
            snip_temp[:, cand_chan] = template[snip_from:snip_to, cand_chan]
            new_obj[j] = recon_error(snip_temp, rank=rank)
        #plt.plot(np.arange(- max_shift, max_shift + 1, 1), new_obj)
        # Optimal local jitterupsample
        opt_shift = np.argmin(new_obj) - half_max_shift
        shifts[cand_chan] = opt_shift
        snip_from, snip_to = snip_win[0] + opt_shift, snip_win[1] + opt_shift
        if snip_to == 0:
            snip_to = new_times
        snip_temp[:, cand_chan] = template[snip_from:snip_to, cand_chan]
        obj = min(new_obj)
        obj_list.append(obj)

    return snip_temp, obj_list


def optimal_svd_align(template, geometry, rank=3, upsample=5, chunk=7, max_shift=10):
    """Iterative svd then align approach to alignment."""
    max_shift = upsample * max_shift

    n_times = template.shape[0]
    n_chan = template.shape[1]
    main_chan = np.flip(main_channels(template), axis=0)
    win_len = n_times * upsample - max_shift

    # Upsample
    temp = sp.signal.resample(template, n_times * upsample)
    shifts = np.zeros(n_chan, dtype=int) + max_shift // 2
    #
    chunk_set = 0
    i = 1
    terminate = False
    while not terminate:
        if i * chunk > n_chan:
            cum_chan = main_chan
            terminate = True
        else:
            #cum_chan = main_chan[:i * chunk]
            cum_chan = geometry.neighbors(main_chan[0], size=chunk * i)
        for iteration in range(4):
            temp_ref = []
            for c in cum_chan:
                temp_ref.append(temp[shifts[c]:shifts[c] + win_len, c])
            temp_ref = np.array(temp_ref).T
            temp_ref_rec = recon(temp_ref, rank=rank)
            shifts[cum_chan] = align_temp_to_temp(temp_ref_rec, temp[:, cum_chan])
        i += 1
    aligned_temp = []
    for c in range(n_chan):
            aligned_temp.append(temp[shifts[c]:shifts[c] + win_len, c])
    return np.array(aligned_temp).T


def plot_spatial(geom, temp, ax, color='C0', scale=10., squeeze=8.):
    """Plots template spatially."""
    leng = temp.shape[0]
    for c in range(temp.shape[1]):
        ax.plot(
            np.arange(0, leng, 1) / squeeze + geom[c, 0],
            temp[:, c] * scale + geom[c, 1], alpha=0.7, color=color, lw=2)

def plot_spatial_fill(geom, temp, ax, color='C0', scale=10., squeeze=8.):
    """Plots standard error for each channel spatially."""
    temp_ = temp * 0
    leng = temp.shape[0]
    for c in range(temp.shape[1]):
        ax.fill_between(
            np.arange(0, leng, 1) / squeeze + geom[c, 0],
            temp_[:, c] - scale / 2  + geom[c, 1],
            temp_[:, c] + scale / 2 + geom[c, 1], color=color, alpha=0.3)