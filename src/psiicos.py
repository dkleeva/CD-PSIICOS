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

    # ---- Set of auto-topographies ---------------------------------------
    A = np.empty((n_ch**2, n_src * 3))
    k = 0
    for s in range(n_src):
        gx, gy = G[:, 2*s], G[:, 2*s+1]
        w      = weights[s]

        A[:, k] = w * np.kron(gx, gx) / np.linalg.norm(np.kron(gx, gx)); k += 1
        A[:, k] = w * np.kron(gy, gy) / np.linalg.norm(np.kron(gy, gy)); k += 1
        tmp     = np.kron(gx, gy) + np.kron(gy, gx)
        A[:, k] = w * tmp / np.linalg.norm(tmp);                          k += 1

    # ---- SVD → SL ---------------------------------------
    U, _, _ = np.linalg.svd(A, full_matrices=False)

    # ---- Projector of the given rank  -----------------------------------
    P = None
    if rank is not None:
        Ur = U[:, :rank]
        P  = np.eye(U.shape[0]) - Ur @ Ur.T

    return P, U

import numpy as np


def _source_power(cs: np.ndarray) -> float:
    """
    Return the largest singular value of a 2×2 complex matrix.

    Parameters
    ----------
    cs : ndarray, shape (2, 2)
        Cross-spectral matrix for a single source in its 2-D tangential basis.

    Returns
    -------
    float
        A scalar power measure (used as a weight in CD-PSIICOS).
    """
    return np.linalg.svd(cs, compute_uv=False)[0]


def power_map_mne(G2d: np.ndarray,
                  CT_avg: np.ndarray,
                  lambd: float = 1.0) -> np.ndarray:
    """
    Estimate source power using an ℓ2-regularised MNE inverse.

    Parameters
    ----------
    G2d : ndarray, shape (n_channels, 2*n_sources)
        Tangential-plane forward matrix.
    CT_avg : ndarray, shape (n_channels, n_channels)
        Real part of the sensor-space cross-spectral matrix averaged over the
        chosen time-frequency window.
    lambd : float, optional
        Tikhonov regularisation parameter (λ).  Default = 1.0.

    Returns
    -------
    power_map : ndarray, shape (n_sources,)
        Normalised power weights in the range [0, 1].
    """
    n_ch, _ = G2d.shape
    n_src   = G2d.shape[1] // 2

    # Minimum-norm linear inverse
    GGT  = G2d @ G2d.T
    lam2 = lambd * np.trace(GGT) / n_ch
    W    = G2d.T @ np.linalg.inv(GGT + lam2 * np.eye(n_ch))

    power_map = np.empty(n_src)
    for s in range(n_src):
        ai = W[2*s:2*s+2, :]       
        cs = ai @ CT_avg @ ai.T    
        power_map[s] = _source_power(cs)

    return power_map / (power_map.max())


def power_map_sloreta(G2d: np.ndarray,
                      CT_avg: np.ndarray,
                      lambd: float = 1.0) -> np.ndarray:
    """
    Estimate source power using an sLORETA-style inverse (MNE + diagonal
    scaling).

    Parameters
    ----------
    G2d : ndarray, shape (n_channels, 2*n_sources)
        Tangential-plane forward matrix.
    CT_avg : ndarray, shape (n_channels, n_channels)
        Real part of the sensor cross-spectral matrix.
    lambd : float, optional
        Regularisation parameter for the underlying MNE step.  Default = 1.0.

    Returns
    -------
    power_map : ndarray, shape (n_sources,)
        Normalised power weights in the range [0, 1].
    """
    n_ch, _ = G2d.shape
    n_src   = G2d.shape[1] // 2

    # MNE inverse
    GGT  = G2d @ G2d.T
    lam2 = lambd * np.trace(GGT) / n_ch
    W_mne = G2d.T @ np.linalg.inv(GGT + lam2 * np.eye(n_ch))

    # Diagonal scaling for sLORETA
    RM        = W_mne @ G2d
    scale     = 1.0 / (np.diag(RM) + 1e-12)
    W_slo     = np.diag(scale) @ W_mne

    power_map = np.empty(n_src)
    for s in range(n_src):
        ai = W_slo[2*s:2*s+2, :]
        cs = ai @ CT_avg @ ai.T
        power_map[s] = _source_power(cs)

    return power_map / (power_map.max())


def power_map_dics(G2d: np.ndarray,
                   CT_avg: np.ndarray) -> np.ndarray:
    """
    Compute a simple DICS-like power map using the numerator term only
    (i.e., without inverse noise normalisation).

    Parameters
    ----------
    G2d : ndarray, shape (n_channels, 2*n_sources)
        Tangential-plane forward matrix.
    CT_avg : ndarray, shape (n_channels, n_channels)
        Real part of the sensor cross-spectral matrix.

    Returns
    -------
    power_map : ndarray, shape (n_sources,)
        Normalised power weights in the range [0, 1].
    """
    n_src = G2d.shape[1] // 2
    power_map = np.empty(n_src)

    for s in range(n_src):
        ai = G2d[:, 2*s:2*s+2].T   
        cs = ai @ CT_avg @ ai.T
        power_map[s] = _source_power(cs)

    return power_map / (power_map.max())
