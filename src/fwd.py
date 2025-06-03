"""
fwd.py
============
Forward modeling  for CD-PSIICOS
------
Daria Kleeva — dkleeva@gmail.com
"""

import numpy as np
from scipy import sparse

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
            G2d[:, i] /= np.linalg.norm(G2d[:, i]) 

    return G2d, G2d_raw


def spm_svd(X, U=1e-6):

    """
    Perform sparse Singular Value Decomposition (SVD) with energy-based component selection.


    Parameters
    ----------
    X : ndarray of shape (M, N)
        Input data matrix to decompose.

    U : float, optional (default=1e-6)
        Threshold on the cumulative energy (relative to total) for retaining singular components.
        Must be in the open interval (0, 1). A smaller value retains more components.

    Returns
    -------
    U : ndarray of shape (M, K)
        Left singular vectors (projected back to original row indices),
        where K is the number of retained components.

    S : scipy.sparse.csr_matrix of shape (K, K)
        Diagonal matrix with retained singular values.

    V : ndarray of shape (N, K)
        Right singular vectors (projected back to original column indices).
    """
    if U >= 1:
        U = U - 1e-6
    if U <= 0:
        U = 64 * np.finfo(float).eps

    M, N = X.shape
    p = np.any(X, axis=1).nonzero()[0]
    q = np.any(X, axis=0).nonzero()[0]
    X = X[p[:, np.newaxis], q]

    i, j, s = sparse.find(X)
    m, n = X.shape
    if any(i - j):
        if m > n:
            v, S, v = np.linalg.svd(X.T.dot(X), full_matrices=False)
            v=v.T
            S = sparse.csr_matrix(np.diag(S))
            s = np.diag(S)
            j = np.nonzero(s * len(s) / s.sum() > U)[0]
            v = v[:, j]
            u = spm_en(X.dot(v))
            S = np.sqrt(S[j, j])

        elif m < n:
            u, S, u = np.linalg.svd(X.dot(X.T), full_matrices=False)
            u=u.T
            S = sparse.csr_matrix(np.diag(S))
            s = np.diag(S)
            j = np.nonzero(s * len(s) / s.sum() > U)[0]
            u = u[:, j]
            v = spm_en(X.T.dot(u))
            S = np.sqrt(S[j, j])

        else:
            u, S, v = np.linalg.svd(X, full_matrices=False)
            v=v.T
            S = sparse.csr_matrix(np.diag(S))
            s = np.diag(S.toarray()) ** 2
            j = np.nonzero(s * len(s) / s.sum() > U)[0]
            v = v[:, j]
            u = u[:, j]
            S = S[j, j]

    else:
        S = sparse.csr_matrix((s, (np.arange(n), np.arange(n))), shape=(n, n))
        u = sparse.eye(m, n)
        v = sparse.eye(m, n)
        j = np.argsort(-s)
        S = S[j, j]
        v = v[:, j]
        u = u[:, j]
        s = np.diag(S) ** 2
        j = np.nonzero(s * len(s) / s.sum() > U)[0]
        v = v[:, j]
        u = u[:, j]
        S = S[j, j]

    j = len(j)
    U_sparse = sparse.csr_matrix((M, j))
    V_sparse = sparse.csr_matrix((N, j))
    if j:
        U_sparse[p, :] = u
        V_sparse[q, :] = v

    U = U_sparse.toarray()
    V = V_sparse.toarray()
    return U, S, V

