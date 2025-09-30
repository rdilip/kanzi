import numpy as np

def kabsch_rmsd(P, Q, weights=None, mask=None, return_transform=False):
    """
    Compute RMSD after optimal superposition of P onto Q via the Kabsch algorithm.

    Parameters
    ----------
    P : (L,3) array_like
        "Mobile" coordinates to be rotated/translated.
    Q : (L,3) array_like
        "Target" coordinates to align to.
    weights : (L,) array_like, optional
        Non-negative weights per point. If provided, uses weighted Kabsch and weighted RMSD.
    mask : (L,) boolean or index array, optional
        Subset of rows to use for alignment/RMSD. Applied to both P and Q.
    return_transform : bool, default False
        If True, also return (R, t, P_aligned) where P_aligned = P @ R + t.

    Returns
    -------
    rmsd : float
    (R, t, P_aligned) : only if return_transform=True
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P = P.reshape(-1, 3)
    Q = Q.reshape(-1, 3)


    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected P and Q with shape (L,3) and same shape; got {P.shape=} {Q.shape=}")

    # Optional sub-selection
    if mask is not None:
        P = P[mask]
        Q = Q[mask]
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)[mask]

    # Optional weights
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != P.shape[0]:
            raise ValueError("weights must have shape (L,) matching P/Q (after masking).")
        if np.any(w < 0) or not np.any(w > 0):
            raise ValueError("weights must be non-negative with at least one positive.")
        wsum = w.sum()
        Pc = P - (w[:, None] * P).sum(axis=0) / wsum
        Qc = Q - (w[:, None] * Q).sum(axis=0) / wsum
    else:
        w = None
        Pc = P - P.mean(axis=0, keepdims=True)
        Qc = Q - Q.mean(axis=0, keepdims=True)

    # Covariance (rows = points)
    C = (Pc.T @ Qc) if w is None else (Pc * w[:, None]).T @ Qc

    # SVD and proper rotation (R = U @ Vt for C = P^T Q)
    U, S, Vt = np.linalg.svd(C, full_matrices=True)
    R = U @ Vt
    if np.linalg.det(R) < 0:     # reflection fix
        U[:, -1] *= -1.0
        R = U @ Vt

    # Translation: t = mu_Q - mu_P @ R
    P_centroid = P.mean(axis=0) if w is None else (w[:, None] * P).sum(axis=0) / (w.sum())
    Q_centroid = Q.mean(axis=0) if w is None else (w[:, None] * Q).sum(axis=0) / (w.sum())
    t = Q_centroid - P_centroid @ R

    # Apply transform and compute (weighted) RMSD
    P_aligned = P @ R + t
    diffsq = np.sum((P_aligned - Q) ** 2, axis=1)
    rmsd = np.sqrt(diffsq.mean() if w is None else (w * diffsq).sum() / w.sum())

    if return_transform:
        return rmsd, R, t, P_aligned
    return rmsd