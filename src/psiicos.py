"""
psiicos.py
=======
Projection helpers for CD-PSIICOS
------
Daria Kleeva — dkleeva@gmail.com
"""

import numpy as np

def apply_psiicos_projection(
    CT: np.ndarray,
    U: np.ndarray,
    rank: int,
) -> np.ndarray:
    """
    Apply a PSIICOS or CD-PSIICOS projector of a given rank to a
    sensor-space cross-spectral time series.

    Parameters
    ----------
    CT : ndarray, shape (n_ch**2, n_times)
        Cross-spectral time series in vectorised sensor space.
    U  : ndarray, shape (n_ch**2, 3*n_sources)
        Full left-singular-vector matrix returned by `make_psiicos_projector`.
    rank : int
        Number of leading singular vectors to remove (projection rank).

    Returns
    -------
    CT_proj : ndarray, shape (n_ch**2, n_times)
        Projected cross-spectral time series with spatial-leakage
        contributions of rank `rank` suppressed.
    """
    if rank <= 0:
        return CT.copy()

    Ur        = U[:, :rank]                 
    CT_proj   = CT - Ur @ (Ur.T @ CT)     
    return CT_proj


def make_psiicos_projector(
    G2d: np.ndarray,
    weights: np.ndarray | None = None,
    rank: int | None = None,
):

    G = G2d.copy()
    n_ch, _ = G.shape
    n_src   = G.shape[1] // 2          

    if weights is None:
        weights = np.ones(n_src)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.size != n_src:
            raise ValueError("weights must have length n_sources (=G2d.shape[1]//2)")
        weights = weights / (weights.max() + 1e-12)   

    # ---- Set of auto-topographies ------
    A = np.empty((n_ch**2, n_src * 3))
    k = 0
    for s in range(n_src):
        gx, gy = G[:, 2*s], G[:, 2*s+1]
        w      = weights[s]

        A[:, k] = w * np.kron(gx, gx) / np.linalg.norm(np.kron(gx, gx)); k += 1
        A[:, k] = w * np.kron(gy, gy) / np.linalg.norm(np.kron(gy, gy)); k += 1
        tmp     = np.kron(gx, gy) + np.kron(gy, gx)
        A[:, k] = w * tmp / np.linalg.norm(tmp);                          k += 1

    # ---- SVD → SL ----------------
    U, _, _ = np.linalg.svd(A, full_matrices=False)

    # ---- Projector of the given rank  ----------
    P = None
    if rank is not None:
        Ur = U[:, :rank]
        P  = np.eye(U.shape[0]) - Ur @ Ur.T

    return P, U



