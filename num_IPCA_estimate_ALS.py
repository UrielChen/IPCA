# file: num_IPCA_estimate_ALS.py

# =========================================================================
# Purpose:
#   Performs one iteration of the Alternating Least Squares (ALS) algorithm
#   for Instrumented Principal Components Analysis (IPCA). Given the current
#   estimate of the factor loadings (Gamma_Old), cross-sectional weight matrices (W),
#   managed portfolio returns (X), and cross-sectional counts (Nts), this function
#   updates the loadings (Gamma_New) and latent factors (F_New).
#
# Assumptions:
#   - Gamma_New[:, :K] must be orthonormal; mean of each factor in F_New is positive.
#   - Optionally supports pre-specified factors (PSF) appended to the latent factors.
#
# Inputs:
#   Gamma_Old : numpy.ndarray (L x K or L x Ktilde)
#       Previous loadings estimate.
#   W         : numpy.ndarray (L x L x T)
#       Weighting matrices for each time period.
#   X         : numpy.ndarray (L x T)
#       Managed returns for each time period.
#   Nts       : numpy.ndarray (T,) or (1 x T)
#       Cross-sectional counts for each period.
#   PSF       : numpy.ndarray (Kadd x T), optional
#       Pre-specified factors matrix, if used.
#
# Outputs:
#   Gamma_New : numpy.ndarray (L x Ktilde)
#       Updated loadings matrix (with/without PSF columns).
#   F_New     : numpy.ndarray (K x T)
#       Updated latent factors (latent only, not PSF).
# =========================================================================

import numpy as np

def num_IPCA_estimate_ALS(Gamma_Old, W, X, Nts, PSF=None):
    # -------------------------------------------------------------
    # Handle optional PSF argument and flag for use
    PSF_version = PSF is not None  # True if pre-specified factors provided

    # -------------------------------------------------------------
    # Extract dimensions: L = #chars, K/Ktilde = #factors (+PSF), T = #periods
    # Gamma_Old: (L x K) or (L x Ktilde)
    # W: (L x L x T)
    # X: (L x T)
    # Nts: (T,)
    T = len(Nts)  # Number of time periods
    if PSF_version:
        L, Ktilde = Gamma_Old.shape       # L = #chars, Ktilde = K + Kadd
        Kadd = PSF.shape[0]              # Kadd = #pre-specified factors
        K = Ktilde - Kadd                # K = #latent factors
    else:
        L, K = Gamma_Old.shape
        Ktilde = K

    # -------------------------------------------------------------
    # Step 1: Update latent factors F_New given Gamma_Old
    F_New = None  # (K x T)
    if K > 0:
        # Allocate F_New with shape (K, T)
        F_New = np.full((K, T), np.nan)
        # For each time period, update latent factors
        for t in range(T):
            if PSF_version:
                # Remove PSF effect from X
                # X[:,t]: (L,), W[:,:,t]: (L,L), Gamma_Old[:,K:]: (L,Kadd), PSF[:,t]: (Kadd,)
                residual = X[:, t] - W[:, :, t] @ Gamma_Old[:, K:Ktilde] @ PSF[:, t]
                # Solve (Gamma_Old_latent' * W * Gamma_Old_latent) x = Gamma_Old_latent' * residual
                lhs = Gamma_Old[:, :K].T @ W[:, :, t] @ Gamma_Old[:, :K]
                rhs = Gamma_Old[:, :K].T @ residual
                F_New[:, t] = np.linalg.solve(lhs, rhs)
            else:
                # Standard latent-only update
                lhs = Gamma_Old.T @ W[:, :, t] @ Gamma_Old
                rhs = Gamma_Old.T @ X[:, t]
                F_New[:, t] = np.linalg.solve(lhs, rhs)

    # -------------------------------------------------------------
    # Step 2: Update loadings Gamma_New given F_New (+ PSF)
    # Numer: (L*Ktilde,) ; Denom: (L*Ktilde, L*Ktilde)
    Numer = np.zeros((L * Ktilde,))
    Denom = np.zeros((L * Ktilde, L * Ktilde))
    for t in range(T):
        if PSF_version:
            if K > 0:
                # stackF: (Ktilde,)
                stackF = np.concatenate([F_New[:, t], PSF[:, t]])  # (Ktilde,)
            else:
                # stackF: (K,)
                stackF = PSF[:, t]
        else:
            stackF = F_New[:, t]  # (K,)
        # Kronecker for vectorized OLS update
        Numer += np.kron(X[:, t], stackF) * Nts[t]
        Denom += np.kron(W[:, :, t], np.outer(stackF, stackF)) * Nts[t]
    # Solve for vectorized Gamma'
    Gamma_trans_vec = np.linalg.solve(Denom, Numer)
    # Reshape: (L, Ktilde)
    Gamma_New = Gamma_trans_vec.reshape((L, Ktilde))

    # -------------------------------------------------------------
    # Step 3: Identification and sign conventions for Gamma_New and F_New
    if K > 0:
        # Orthonormalize Gamma_New latent block (L x K)
        # Compute upper Cholesky of Gamma_New'Gamma_New
        R1 = np.linalg.cholesky(Gamma_New[:, :K].T @ Gamma_New[:, :K]).T  # (K, K), upper
        # SVD for additional rotation
        U, _, _ = np.linalg.svd(R1 @ F_New @ F_New.T @ R1.T)
        # Update Gamma_New latent block: (L x K)
        Gamma_New[:, :K] = (np.linalg.solve(R1, Gamma_New[:, :K].T).T) @ U
        # Update factors
        F_New = np.linalg.solve(U, R1 @ F_New)
        # Enforce sign: mean of each factor must be positive
        sg = np.sign(np.mean(F_New, axis=1))  # (K,)
        sg[sg == 0] = 1
        Gamma_New[:, :K] = Gamma_New[:, :K] * sg  # broadcasting over columns
        F_New = F_New * sg[:, None]              # broadcasting over rows

    # Print orthonormalization check
    if K > 0:
        check = Gamma_New[:, :K].T @ Gamma_New[:, :K]

    # -------------------------------------------------------------
    # Step 4: Orthogonalize PSF loadings if present
    if PSF_version and K > 0:
        # Partition Gamma_New into Gbeta (latent, L x K) and Gdelta (PSF, L x Kadd)
        Gbeta = Gamma_New[:, :K]
        Gdelta = Gamma_New[:, K:]
        # Project Gdelta orthogonal to Gbeta
        Gdelta = (np.eye(L) - Gbeta @ Gbeta.T) @ Gdelta
        # Adjust factors to maintain orthogonality
        gamma_coef = Gbeta.T @ Gdelta  # (K x Kadd)
        F_New = F_New + gamma_coef @ PSF
        # Recombine
        Gamma_New = np.concatenate([Gbeta, Gdelta], axis=1)
        # Re-enforce sign on latent block
        sg = np.sign(np.mean(F_New, axis=1))
        sg[sg == 0] = 1
        Gamma_New[:, :K] = Gamma_New[:, :K] * sg
        F_New = F_New * sg[:, None]

    # -------------------------------------------------------------
    # Return updated loadings and factors
    return Gamma_New, F_New

# Example usage
if __name__ == "__main__":
    # Set up a simple synthetic test for verification
    L, K, T = 5, 2, 4   # chars, latent factors, periods
    np.random.seed(0)
    Gamma_Old = np.random.randn(L, K)
    W = np.repeat(np.eye(L)[..., None], T, axis=2)
    X = np.random.randn(L, T)
    Nts = np.ones(T)
    # Optional PSF
    Kadd = 1
    PSF = np.random.randn(Kadd, T)
    # Test with and without PSF
    print("==== ALS update without PSF ====")
    Gamma_New, F_New = num_IPCA_estimate_ALS(Gamma_Old, W, X, Nts)
    print("==== ALS update with PSF ====")
    Gamma_Old_psf = np.random.randn(L, K+Kadd)
    Gamma_New_psf, F_New_psf = num_IPCA_estimate_ALS(Gamma_Old_psf, W, X, Nts, PSF=PSF)
