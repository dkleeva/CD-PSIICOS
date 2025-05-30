import numpy as np
import matplotlib.pyplot as plt
import mne
from nilearn import plotting


def compute_sensor_cs(epochs):
    """
    Compute the time-resolved sensor-space cross-spectral matrix 
    from bandpass-filtered epoched MEG/EEG data.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed bandpass-filtered epochs with evoked component removed.

    Returns
    -------
    cross_spectrum : np.ndarray, shape (n_channels**2, n_times)
        Cross-spectral time series (vectorized cross-spectral matrices).
    analytic_signal : np.ndarray, shape (n_channels, n_times, n_trials)
        Complex analytic signal per channel/trial.
    phase_angles : np.ndarray, shape (n_channels, n_times, n_trials)
        Instantaneous phases of the analytic signal.
    """
    data = epochs.get_data() 
    data = np.transpose(data, (1, 2, 0))  

    n_channels, n_times, n_trials = data.shape

    # Hilbert transform via FFT (complex analytic signal)
    Xfft = np.fft.fft(data, axis=1)

    # Construct frequency-domain Hilbert mask
    h = np.zeros(n_times)
    if n_times % 2 == 0:  # even
        h[0] = h[n_times // 2] = 1
        h[1:n_times // 2] = 2
    else:  # odd
        h[0] = 1
        h[1:(n_times + 1) // 2] = 2

    H = h[:, None]  
    H = np.tile(H, (1, n_trials))  

    H = np.broadcast_to(H, Xfft.shape)

    analytic_signal = np.fft.ifft(Xfft * H, axis=1)
    phase_angles = np.angle(analytic_signal)

    X = analytic_signal
    X_conj = np.conj(X)
    cross_spectrum = np.zeros((n_channels ** 2, n_times), dtype=np.complex128)

    for i in range(n_channels):
        mn = np.mean(X * X_conj[i, :, :], axis=2)  
        cross_spectrum[i * n_channels:(i + 1) * n_channels, :] = mn

    return cross_spectrum, analytic_signal, phase_angles

import numpy as np

def build_connectivity_matrix(
    C_ts: np.ndarray,
    pair_idx: np.ndarray,
    times: np.ndarray,
    tmin: float,
    tmax: float,
    keep_top: float = 0.10,
):
    """
    Aggregate a PSIICOS/CD-PSIICOS coupling time-series into a
    symmetric connectivity matrix and apply a relative threshold.

    Parameters
    ----------
    C_ts : ndarray, shape (n_connections, n_times)
        Coupling strength for every source pair over time.
    pair_idx : ndarray, shape (n_connections, 2)
        Integer indices of each pair  (j, i) with j > i.
    times : ndarray, shape (n_times,)
        Epoch time axis (seconds).
    tmin, tmax : float
        Time window (in seconds) over which to average the coupling.
    keep_top : float, default 0.10
        Fraction (0–1) of the strongest absolute connections to keep.
        Everything below that quantile is zeroed out.

    Returns
    -------
    conn_matrix : ndarray, shape (n_sources, n_sources)
        Symmetric matrix with averaged and thresholded coupling values.
    """
    n_src = pair_idx.max() + 1
    conn_matrix = np.zeros((n_src, n_src))

    # --- time mask ---------------------------------------------------------
    win_mask = (times >= tmin) & (times <= tmax)
    if not win_mask.any():
        raise ValueError("No samples fall inside the [tmin, tmax] window.")

    # --- average over the chosen window ------------------------------------
    mean_vals = C_ts[:, win_mask].mean(axis=1)

    # --- fill upper triangle -----------------------------------------------
    for val, (j, i) in zip(mean_vals, pair_idx):
        conn_matrix[i, j] = conn_matrix[j, i] = val

    # --- relative threshold -------------------------------------------------
    if keep_top < 1.0:
        thr = np.quantile(np.abs(conn_matrix), 1.0 - keep_top)
        conn_matrix[np.abs(conn_matrix) < thr] = 0.0

    return conn_matrix





def plot_pairs_3d(
    fwd,
    conn_mat: np.ndarray,
    rel_thr: float = 0.10,
    subject: str = "fsaverage",
    subjects_dir: str | None = None,
    linewidth: float = 3.0,
    node_size: float = 2.0,
):
    """
    Interactive 3-D visualisation of a source-space connectivity matrix.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution that defines source locations.
    conn_mat : ndarray, shape (n_src, n_src)
        Symmetric connectivity matrix (e.g. output of `build_connectivity_matrix`).
    rel_thr : float, default 0.10
        Keep only connections whose absolute value exceeds
        `rel_thr * max(|conn_mat|)`.  Typical range 0.05–0.20.
    subject : str, default "fsaverage"
        Subject ID for MNI conversion (must match `fwd`).
    subjects_dir : str or None
        FreeSurfer `SUBJECTS_DIR`.  If None, MNE will look for the
        `SUBJECTS_DIR` environment variable.
    linewidth : float, default 3.0
        Line width of displayed edges.
    node_size : float, default 2.0
        Size of rendered source nodes.

    Returns
    -------
    nilearn.viewers.surface_viewer.SurfaceView
        An interactive HTML viewer (in-notebook if supported).
    """

    coords = []
    for src in fwd["src"]:
        coords.append(src["rr"][src["vertno"]])
    coords = np.concatenate(coords, axis=0)                  

    coords_mni = mne.head_to_mni(
        coords,
        subject,
        fwd["mri_head_t"],
        subjects_dir=subjects_dir,
        verbose=False,
    )


    abs_max = np.abs(conn_mat).max()
    if abs_max == 0.0:
        raise ValueError("Connectivity matrix is all zeros.")

    thr_val = rel_thr * abs_max
    conn_thr = np.where(np.abs(conn_mat) >= thr_val, conn_mat, 0.0)

    sel = np.abs(conn_thr) > 0
    if sel.any():
        conn_thr[sel] /= np.abs(conn_thr[sel]).max()


    view = plotting.view_connectome(
        conn_thr,
        coords_mni,
        linewidth=linewidth,
        node_size=node_size,
        colorbar=True,
    )
    return view


def plot_mean_coupling_trace(
    C_ts: np.ndarray,
    pair_idx: np.ndarray,
    conn_mat: np.ndarray,
    times: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
    ax: plt.Axes | None = None,
    return_ts: bool = False,
):
    """
    Plot the mean coupling time-series across all retained connections.

    Parameters
    ----------
    C_ts : ndarray, shape (n_connections, n_times)
        Coupling time-series for every (i < j) pair.
    pair_idx : ndarray, shape (n_connections, 2)
        Pair indices from `scan_fast` ( IND[k] = [j, i] ).
    conn_mat : ndarray, shape (n_src, n_src)
        Thresholded symmetric matrix (non-zero entries mark retained pairs).
    times : ndarray, shape (n_times,)
        Epoch time axis (seconds).
    tmin, tmax : float or None
        Optional limits for the x-axis.  If None, use full range.
    ax : matplotlib Axes or None
        Axes to plot into.  If None, a new figure is produced.
    return_ts : bool, default False
        If True, the averaged trace is returned in addition to plotting.

    Returns
    -------
    trace : ndarray, shape (n_times,)   (only if return_ts=True)
        Mean coupling over retained pairs for each time-point.
    """

    retained = []
    for k, (j, i) in enumerate(pair_idx):
        if conn_mat[i, j] != 0.0: 
            retained.append(k)
    if not retained:
        raise RuntimeError("No connections survived thresholding.")

    retained = np.asarray(retained, dtype=int)


    trace = C_ts[retained].mean(axis=0) 


    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    ax.plot(times, trace, color="tab:blue")
    ax.axhline(0, color="gray", lw=0.8)
    if tmin is not None or tmax is not None:
        ax.set_xlim(tmin if tmin is not None else times[0],
                    tmax if tmax is not None else times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean coupling")
    ax.set_title("Averaged coupling")
    ax.grid(True, alpha=0.3)

    if return_ts:
        return trace

