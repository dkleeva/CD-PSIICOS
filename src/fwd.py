"""
fwd.py
============
Forward modeling  for CD-PSIICOS
------
Daria Kleeva — dkleeva@gmail.com
"""

import numpy as np

def prepare_fwd_2d(fwd, normalize=True):
    """
    Reduce a 3D forward model to a 2D tangential-plane model per cortical source.

    Parameters
    ----------
    fwd : instance of mne.Forward
        MNE forward solution object with free orientations (3 per source).
    
    normalize : bool
        Whether to normalize each resulting column to unit norm.

    Returns
    -------
    G2d : np.ndarray, shape (n_channels, 2 * n_sources)
        Reduced 2D forward model with normalized orientations.
    G2d_raw : np.ndarray, shape (n_channels, 2 * n_sources)
        Same as G2d but without final column-wise normalization.
    """
    G = fwd['sol']['data']
    n_channels, total_dipoles = G.shape
    n_sources = total_dipoles // 3

    G2d = np.zeros((n_channels, 2 * n_sources))
    G2d_raw = np.zeros((n_channels, 2 * n_sources))

    for i in range(n_sources):
        Gi = G[:, 3 * i: 3 * i + 3]
        U, S, Vh = np.linalg.svd(Gi, full_matrices=False)  # Vh: (3, 3)
        V2 = Vh.T[:, :2]  # Take top 2 spatial components (tangential plane)
        Gi2d = Gi @ V2

        # Normalize columns in-place (local tangential directions)
        Gi2d_normalized = Gi2d @ np.diag(1.0 / np.sqrt(np.sum(Gi2d**2, axis=0)))

        G2d[:, 2 * i:2 * i + 2] = Gi2d_normalized
        G2d_raw[:, 2 * i:2 * i + 2] = Gi2d

    if normalize:
        for i in range(G2d.shape[1]):
            G2d[:, i] /= np.linalg.norm(G2d[:, i]) + 1e-10

    return G2d, G2d_raw
