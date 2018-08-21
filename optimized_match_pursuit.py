import numpy as np
import scipy


class OptimizedMatchPursuit(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, data, temps, threshold=2., conv_approx_rank=3,
            upsample=1, vis_su=2., keep_iterations=False):
        """Sets up the deconvolution object.

        Parameters:
        -----------
        data: numpy array of shape (T, C)
            Where T is number of time samples and C number of channels.
        temps: numpy array of shape (t, C, K)
            Where t is number of time samples and C is the number of
            channels and K is total number of units.
        conv_approx_rank: int
            Rank of SVD decomposition for approximating convolution
            operations for templates.
        threshold: float
            amount of energy differential that is admissible by each
            spike. The lower this threshold, more spikes are recovered. 
        vis_su: float
            threshold for visibility of template channel in terms
            of peak to peak standard unit.
        keep_iterations: boolean
            Keeps the spike train per iteration if True. Otherwise,
            does not keep the history.
        """
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps.astype(np.float32)
        # Upsample and downsample time shifted versions
        self.up_factor = upsample
        self.threshold = threshold
        self.approx_rank = conv_approx_rank
        self.vis_su_threshold = vis_su
        self.vis_chan = None
        self.visible_chans()
        self.template_overlaps()
        self.spatially_mask_templates()
        # Upsample the templates
        # Index of the original templates prior to
        # upsampling them.
        self.orig_n_unit = self.n_unit
        self.n_unit = self.orig_n_unit * self.up_factor
        self.orig_template_idx = np.arange(0, self.n_unit, self.up_factor)
        # Computing SVD for each template.
        self.compress_templates()
        # Compute pairwise convolution of filters
        self.pairwise_filter_conv()
        # compute norm of templates
        self.norm = np.zeros([self.orig_n_unit, 1], dtype=np.float32)
        for i in range(self.orig_n_unit):
            self.norm[i] = np.sum(np.square(self.temps[:, self.vis_chan[:, i], i]))
        # Setting up data properties
        self.keep_iterations = keep_iterations
        self.update_data(data)
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        # Energey reduction for assigned spikes.
        self.dist_metric = np.array([])
        # Single time preperation for high resolution matches
        # matching indeces of peaks to indices of upsampled templates
        factor = self.up_factor
        radius = factor // 2 + factor % 2
        self.up_window = np.arange(-radius - 1, radius + 1)[:, None]
        self.up_window_len = len(self.up_window)
        off = (factor + 1) % 2
        peak_to_template_idx = np.append(
                np.arange(radius + off, factor),
                np.arange(radius + off))
        self.peak_to_template_idx = np.pad(
                peak_to_template_idx,
                (radius * (factor + 1) + off, radius * (factor - 1) - off),
                'edge')
        peak_time_jitter = np.array([1, 0]).repeat(radius)
        peak_time_jitter[radius - 1] = 0
        self.peak_time_jitter = np.pad(
                peak_time_jitter,
                (radius * (factor + 1) + off, radius * (factor - 1) - off),
                'edge')

    def update_data(self, data):
        """Updates the data for the deconv to be run on with same templates."""
        self.data = data.astype(np.float32)
        self.data_len = data.shape[0]
        # Computing SVD for each template.
        self.obj_len = self.data_len + self.n_time - 1
        self.dot = np.zeros(
                [self.orig_n_unit, self.obj_len],
                dtype=np.float32)
        # Indicator for computation of the objective.
        self.obj_computed = False
        # Resulting recovered spike train.
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])
        self.iter_spike_train = []

    def visible_chans(self):
        if self.vis_chan is None:
            a = np.max(self.temps, axis=0) - np.min(self.temps, 0)
            self.vis_chan = a > self.vis_su_threshold
        return self.vis_chan

    def template_overlaps(self):
        """Find pairwise units that have overlap between."""
        vis = self.vis_chan.T
        self.unit_overlap = np.sum(
            np.logical_and(vis[:, None, :], vis[None, :, :]), axis=2)
        self.unit_overlap = self.unit_overlap > 0
        self.unit_overlap = np.repeat(self.unit_overlap, self.up_factor, axis=0)

    def spatially_mask_templates(self):
        """Spatially mask templates so that non visible channels are zero."""
        idx = np.logical_xor(
                np.ones(self.temps.shape, dtype=bool), self.vis_chan)
        self.temps[idx] = 0.

    def compress_templates(self):
        """Compresses the templates using SVD and upsample temporal compoents."""
        self.temporal, self.singular, self.spatial = np.linalg.svd(
            np.transpose(np.flipud(self.temps), (2, 0, 1)))
        # Keep only the strongest components
        self.temporal = self.temporal[:, :, :self.approx_rank]
        self.singular = self.singular[:, :self.approx_rank]
        self.spatial = self.spatial[:, :self.approx_rank, :]
        # Upsample the temporal components of the SVD
        # in effect, upsampling the reconstruction of the
        # templates.
        if self.up_factor == 1:
            # No upsampling is needed.
            self.temporal_up = self.temporal
            return
        self.temporal_up = scipy.signal.resample(
                self.temporal, self.n_time * self.up_factor, axis=1)
        idx = np.arange(0, self.n_time * self.up_factor, self.up_factor) + np.arange(self.up_factor)[:, None]
        self.temporal_up = np.reshape(
                self.temporal_up[:, idx, :], [-1, self.n_time, self.approx_rank])

    def pairwise_filter_conv(self):
        """Computes pairwise convolution of templates using SVD approximation."""
        conv_res_len = self.n_time * 2 - 1
        self.pairwise_conv = np.zeros(
                [self.n_unit, self.orig_n_unit, conv_res_len], dtype=np.float32)
        for unit1 in range(self.orig_n_unit):
            u, s, vh = self.temporal[unit1], self.singular[unit1], self.spatial[unit1]
            vis_chan_idx = self.vis_chan[:, unit1]
            for unit2 in np.where(self.unit_overlap[:, unit1])[0]:
                orig_unit = unit2 // self.up_factor
                masked_temp = np.flipud(np.matmul(
                        self.temporal_up[unit2] * self.singular[orig_unit],
                        self.spatial[orig_unit, :, vis_chan_idx].T))
                for i in range(self.approx_rank):
                    self.pairwise_conv[unit2, unit1, :] += np.convolve(
                        np.matmul(masked_temp, vh[i, vis_chan_idx].T),
                        s[i] * u[:, i].flatten(), 'full')

    def get_upsampled_templates(self):
        """Get the reconstructed upsampled versions of the original templates.

        If no upsampling was requested, returns the SVD reconstructed version
        of the original templates.
        """
        rec = np.matmul(
                self.temporal_up * np.repeat(self.singular, self.up_factor, axis=0)[:, None, :],
                np.repeat(self.spatial, self.up_factor, axis=0))
        return np.fliplr(rec).transpose([1, 2, 0])

    def approx_conv_filter(self, unit):
        """Approximation of convolution of a template with the data.

        Parameters:
        -----------
        unit: int
            Id of the unit whose filter will be convolved with the data.
        """
        conv_res = 0.
        u, s, vh = self.temporal[unit], self.singular[unit], self.spatial[unit]
        for i in range(self.approx_rank):
            vis_chan_idx = self.vis_chan[:, unit]
            conv_res += np.convolve(
                np.matmul(self.data[:, vis_chan_idx], vh[i, vis_chan_idx].T),
                s[i] * u[:, i].flatten(), 'full')
        return conv_res

    def compute_objective(self):
        """Computes the objective given current state of recording."""
        if self.obj_computed:
            return self.obj
        for i in range(self.orig_n_unit):
            self.dot[i, :] = self.approx_conv_filter(i)
        self.obj = 2 * self.dot - self.norm
        # Set indicator to true so that it no longer is run
        # for future iterations in case subtractions are done
        # implicitly.
        self.obj_computed = True
        return self.obj

    def high_res_peak(self, times, unit_ids):
        """Finds best matching high resolution template.

        Given an original unit id and the infered spike times
        finds out which of the shifted upsampled templates of
        the unit best matches at that time to the residual.

        Parameters:
        -----------
        times: numpy.array of numpy.int
            spike times for the unit.
        unit_ids: numpy.array of numpy.int
            Respective to times, id of each spike corresponding
            to the original units.

        Returns:
        --------
            tuple in the form of (numpy.array, numpy.array) respectively
            the offset of shifted templates and a necessary time shift
            to correct the spike time.
        """
        if self.up_factor == 1 or len(times) < 1:
            return 0, 0
        idx = times + self.up_window
        new_peak_idx = np.argmax(scipy.signal.resample(
            self.obj[unit_ids, idx], self.up_window_len * self.up_factor, axis=0),
            axis=0)
        return self.peak_to_template_idx[new_peak_idx], self.peak_time_jitter[new_peak_idx]

    def find_peaks(self):
        """Finds peaks in subtraction differentials of spikes."""
        refrac_period = self.n_time
        max_across_temp = np.max(self.obj, axis=0)
        spike_times = scipy.signal.argrelmax(max_across_temp, order=refrac_period)[0]
        spike_times = spike_times[max_across_temp[spike_times] > self.threshold]
        dist_metric = max_across_temp[spike_times]
        # TODO(hooshmand): this requires a check of the last element(s)
        # of spike_times only not of all of them since spike_times
        # is sorted already.
        valid_idx = spike_times < self.data_len - self.n_time
        dist_metric = dist_metric[valid_idx]
        spike_times = spike_times[valid_idx]
        # Upsample the objective and find the best shift (upsampled)
        # template.
        spike_ids = np.argmax(self.obj[:, spike_times], axis=0) 
        upsampled_template_idx, time_shift = self.high_res_peak(spike_times, spike_ids)
        spike_ids = spike_ids * self.up_factor + upsampled_template_idx
        spike_times -= time_shift
        result = np.append(
            spike_times[:, None] - self.n_time + 1,
            spike_ids[:, None], axis=1)
        return result, dist_metric

    def enforce_refractory(self, spike_train):
        """Enforces refractory period for units."""
        radius = self.n_time // 2
        window = np.arange(- radius, radius)
        n_spikes = spike_train.shape[0]

        time_idx = spike_train[:, 0:1] + window
        # Re-adjust cluster id's so that they match
        # with the original templates
        unit_idx = spike_train[:, 1:2] // self.up_factor
        self.obj[unit_idx, time_idx] = - np.inf

    def subtract_spike_train(self, spt):
        """Substracts a spike train from the original spike_train."""
        present_units = np.unique(spt[:, 1])
        for i in present_units:
            conv_res_len = self.n_time * 2 - 1
            unit_sp = spt[spt[:, 1] == i, :]
            spt_idx = np.arange(0, conv_res_len) + unit_sp[:, :1] 
            # Grid idx of subset of channels and times
            unit_idx = self.unit_overlap[i]
            idx = np.ix_(unit_idx, spt_idx.ravel())
            self.obj[idx] -= np.tile(2 * self.pairwise_conv[i, unit_idx, :], len(unit_sp))

        self.enforce_refractory(spt)

    def get_iteration_spike_train(self):
        return self.iter_spike_train

    def run(self, max_iter):
        ctr = 0
        tot_max = np.inf
        self.compute_objective()
        while tot_max > self.threshold and ctr < max_iter:
            spt, dist_met = self.find_peaks()
            self.dec_spike_train = np.append(self.dec_spike_train, spt, axis=0)
            self.subtract_spike_train(spt)
            if self.keep_iterations:
                self.iter_spike_train.append(spt)
            self.dist_metric = np.append(self.dist_metric, dist_met)
            ctr += 1
            print "Iteration {0} Found {1} spikes with {2:.2f} energy reduction.".format(
                ctr, spt.shape[0], np.sum(dist_met))
            if len(spt) == 0:
                break
        return self.dec_spike_train, self.dist_metric

