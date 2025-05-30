import numpy as np

def scan_fast(
    G2d: np.ndarray,
    Cp:  np.ndarray,
    part: str = "real",   # "real" | "imag" | "full"
):
    """
    Fast scanning step of PSIICOS / CD-PSIICOS.

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

    C_ij = np.zeros(n_conn, dtype=float if part != "full" else complex)
    IND  = np.zeros((n_conn, 2), dtype=int)

 
    p = 0
    for i in range(n_src):
        a_i = G2d[:, 2*i:2*i+2].T
        tmp = a_i @ Cp_sq                   
        cs_long = tmp @ G2d                   

        if part == "real":
            cs_long = np.real(cs_long)
        elif part == "imag":
            cs_long = np.imag(cs_long)

        cs_long = cs_long.reshape(2, n_src, 2).transpose(0, 2, 1)   

        cs11 = cs_long[0, 0, :]       
        cs22 = cs_long[1, 1, :]
        cs12 = cs_long[0, 1, :]
        cs21 = cs_long[1, 0, :]

        Ti = cs11 + cs22
        Di = cs11*cs22 - cs12*cs21

        lam = 0.5 * (Ti + np.sqrt(np.maximum(0, Ti*Ti - 4*Di)))


        n_left = n_src - i - 1
        C_ij[p : p + n_left] = lam[i+1:]
        IND[p : p + n_left, 0] = np.arange(i+1, n_src)  
        IND[p : p + n_left, 1] = i                    
        p += n_left

    return C_ij, IND
