from heapq import merge
import numpy as np
from tqdm import tqdm

from yass.geometry import find_channel_neighbors


def collision_rate(spike_train, templates, geometry, pair_wise=True,
                   collision_threshold=10, neighborhood_distance=100):
    """

    Parameters:
    -----------
    spike_train: numpy.ndarray shape (N, 2)
        First column is spike time and second column unit id
    templates: numpy.ndarray shape (#units, #channels, #timesamples)
    geometry: numpy.ndarray shape (#channels, 2)
    pair_wise: bool
        If True, pairwise collision is reported. Otherwise, total collisions
        with other units is recorded.
    collision_threshold: int
        Threshold for spike time proximity to constitute a collision.
    neighborhood_distance: float
        Maximum distance of any two neighboring channels.

    Returns:
    --------
    numpy.ndarray. If pair_wise is True, shape is (#units, #units).
    Otherwise shape is (#units,).
    """

    def match(seq1, seq2):
        """Matches collisions of two spike time lists."""
        m, n = len(seq1), len(seq2)
        i, j = 0, 0
        res = 0
        while i < m and j < n:
            if abs(seq1[i] - seq2[j]) < collision_threshold:
                res += 1
                i += 1
            elif seq1[i] < seq2[j]:
                i += 1
            else:
                j += 1
        return res
    # Main channel for each template
    main_chan = templates.ptp(axis=-1).argmax(axis=-1)
    chan_neighbors = find_channel_neighbors(geom, neighborhood_distance)
    n_unit = templates.shape[0]
    # Get spike train for each individual unit
    unit_spt = [spike_train[spike_train[:, 1] == i, 0] for i in range(n_unit)]
    # Main part of algorithm.
    if pair_wise:
        collisions = np.zeros([n_unit, n_unit])
        for i in tqdm(range(n_unit)):
            spatial_radius = np.where(chan_neighbors[main_chan[i]])[0]
            for j in range(n_unit):
                if i == j or main_chan[j] not in spatial_radius:
                    continue
                collisions[i, j] = match(unit_spt[i], unit_spt[j])
    else:
        collisions = np.zeros([n_unit])
        for i in tqdm(range(n_unit)):
            spatial_radius = np.where(chan_neighbors[main_chan[i]])[0]
            neighbor_units_spt = []
            for j in range(n_unit):
                if i == j or main_chan[j] not in spatial_radius:
                    continue
                neighbor_units_spt = merge(neighbor_units_spt, unit_spt[j])
            collisions[i] = match(unit_spt[i], list(neighbor_units_spt))
    return collisions

