"""
inv.py
======

Inverse mapping utilities for CD-PSIICOS

------
Daria Kleeva — dkleeva@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import mne

def scan_fast(
    G2d: np.ndarray,
    Cp:  np.ndarray,
    part: str = "real",   # "real" | "imag" | "full"
):
    """
    Fast scanning step for reconstructing the networks in the source space 
    (as in the original PSIICOS approach (Ossadtchi et al., 2018)).

    Parameters
    ----------
    G2d : ndarray, shape (n_channels, 2*n_sources)
        Tangential-plane forward model (no sensor-space reduction applied).
    Cp  : ndarray, shape (n_channels**2,)
        Vectorised sensor-space cross-spectral matrix for a single time slice
        AFTER projection (i.e. `CT_proj[:, t]`).
    part : {"real", "imag", "full"}
        Which component of the projected cross-spectrum to use:
        "real" → real part only
        "imag" → imaginary part only
        "full" → complex value

    Returns
    -------
    C_ij : ndarray, shape (n_connections,)
        Coupling strength for every unique source pair (i < j).
    IND  : ndarray, shape (n_connections, 2)
        Integer indices of the corresponding source pairs.
        IND[k] = [j, i]  with  j > i
    """
    if part not in {"real", "imag", "full"}:
        raise ValueError('`part` must be "real", "imag", or "full".')

    n_ch, _ = G2d.shape
    n_src   = G2d.shape[1] // 2
    n_conn  = n_src * (n_src - 1) // 2

    Cp_sq = Cp.reshape(n_ch, n_ch)         

    T = np.complex128(np.zeros((1, n_src * (n_src - 1) // 2)))
    D = np.complex128(np.zeros((1, n_src * (n_src - 1) // 2)))

    IND = np.zeros((n_src * (n_src - 1) // 2, 2), dtype=int)

    p = 0
    C_ij = []

    for iSrc in range(n_src ):
        ai = G2d[:, iSrc * 2 :iSrc * 2+2].T
        tmp = ai @ Cp_sq
        cslong = tmp @ G2d     
        if part == 'imag':
            cslong = np.imag(cslong)
        if part == 'real':
            cslong = np.real(cslong)
        cs2long = cslong * np.conj(cslong)
        cs2longd = cslong[0, :] * np.conj(cslong[1, :])
        cs2_11_22 = np.array([
        np.sum(np.reshape(cs2long[0, :], (n_src, 2)).T, axis=0),  
            np.sum(np.reshape(cs2long[1, :], (n_src, 2)).T, axis=0)])
        cs2_12_21 = np.sum(np.reshape(cs2longd, (n_src, 2)).T, axis=0)
        Ti = np.sum(cs2_11_22, axis=0)
        Di = np.prod(cs2_11_22, axis=0) - cs2_12_21 * np.conj(cs2_12_21)
        T[0, p: p + n_src - iSrc - 1] = Ti[iSrc + 1: n_src]
        D[0, p: p + n_src - iSrc - 1] = Di[iSrc + 1: n_src]
        IND[p: p + n_src - iSrc - 1, 1] = iSrc
        IND[p: p + n_src- iSrc - 1, 0] = np.arange(iSrc + 1, n_src)
        p += n_src - iSrc - 1
    C_ij = 0.5 * T + np.sqrt(0.25 * T ** 2 - D)
    C_ij = np.max(C_ij, axis=0)
    return C_ij, IND


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
    scale     = 1.0 / (np.diag(RM))
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

def plot_power_map(
    fwd,
    power_map: np.ndarray,
    subject: str = "fsaverage",
    subjects_dir: str | None = None,
    point_size: float = 10,
    cmap: str = "hot",
    figsize: tuple = (12, 4),
):
    """
    Plot axial, sagittal and coronal 2-D projections of a source-space
    power map.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution defining source coordinates.
    power_map : ndarray, shape (n_sources,)
        Values in [0, 1] returned by power_map_* functions.
    subject : str, default "fsaverage"
        Freesurfer subject ID.
    subjects_dir : str or None
        Path to SUBJECTS_DIR (or set via env variable).
    point_size : float, default 10
        Marker size for scatter points.
    cmap : str, default "hot"
        Matplotlib colormap.
    figsize : tuple, default (12, 4)
        Size of the matplotlib figure (width, height).

    Returns
    -------
    fig : matplotlib Figure
        Figure handle.
    axes : ndarray of Axes
        Array with three subplot axes.
    """

    coords = np.concatenate(
        [src["rr"][src["vertno"]] for src in fwd["src"]], axis=0
    )
    coords_mni = mne.head_to_mni(
        coords,
        subject,
        fwd["mri_head_t"],
        subjects_dir=subjects_dir,
        verbose=False,
    )
    x, y, z = coords_mni.T

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    views = [
        ("Axial (top): X vs Y", x, y),
        ("Sagittal (side): Y vs Z", y, z),
        ("Coronal (front): X vs Z", x, z),
    ]

    for ax, (title, xv, yv) in zip(axes, views):
        sc = ax.scatter(
            xv,
            yv,
            c=power_map,
            s=point_size,
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(title.split(":")[1].split("vs")[0].strip())
        ax.set_ylabel(title.split("vs")[1].strip())
        ax.tick_params(labelsize=8)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Normalised power", fontsize=9)

    return fig, axes

