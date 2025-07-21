# ------------------------------------------------------------------------------
# IPCA_empirical_GB.py
# ------------------------------------------------------------------------------
# Main script to estimate the "restricted" IPCA model (no alpha intercept).
#
# Purpose:
#   - Perform Alternating Least Squares (ALS) estimation of latent factors
#     and loadings given a characteristic-based dataset, using the InstrumentedPCA class.
#   - Compute in-sample fits and R² for both individual assets and managed portfolios.
#
# Assumptions:
#   - Data are stored in a .npz file with variables:
#       X      : (L, T)    managed‑portfolio returns
#       W      : (L, L, T) characteristic‑covariance matrices
#       Z      : (N, L, T) characteristic exposures
#       xret   : (N, T)    asset excess returns
#       LOC    : (N, T)    logical mask for non‑missing obs
#       Nts    : (T,)      # non‑missing stocks per period
#       date   : (T,)      time stamps
#   - Characteristic dimensions: L (number of chars),  
#     Asset cross‑section: N, Time series length: T.
#   - Alternating least squares converges under default tolerances.
#
# Outputs:
#   - GammaBeta (L, K) factor loadings
#   - Factor    (K, T) latent factors
#   - Lambda    (K,)   average factor (risk prices)
#   - RFITS_GB, RFITS_pred_GB (N, T) in-sample and predictive asset fits
#   - XFITS_GB, XFITS_pred_GB (L, T) in-sample and predictive managed fits
#   - RR2_total_GB, RR2_pred_GB, XR2_total_GB, XR2_pred_GB: R² statistics
#   - Saves results per K as .npz files
# ------------------------------------------------------------------------------

import numpy as np                  # For array operations
import pandas as pd                 # For convenience and optional output
import time                         # For timing
from ipca import InstrumentedPCA    # IPCA estimator class
from mySOS import mySOS

# ----------------------------- USER PARAMETERS -------------------------------

# List the range of factor dimensions K to estimate
Krange = range(1, 2)  # e.g., 1 to 6 (inclusive at 1, exclusive at 7)

# Name of data file (update this if needed)
dataname = 'IPCADATA_FNW36_RNKDMN_CON'
datafile = f'../IPCA_KELLY/{dataname}.npz'

# ----------------------------- LOAD DATA -------------------------------------

# Load preprocessed data from .npz file
with np.load(datafile, allow_pickle=True) as D:
    X      = D['X']        # (L, T): managed‑portfolio returns
    W      = D['W']        # (L, L, T): char covariance matrices
    Z      = D['Z']        # (N, L, T): exposures
    xret   = D['xret']     # (N, T): asset returns
    LOC    = D['LOC']      # (N, T): boolean mask
    Nts    = D['Nts']      # (T,): cross-sectional count per time
    date   = D['date']     # (T,): timestamps
    # charnames, if present, can be loaded as D['charnames']

# Get dimensions for reference
L, T = X.shape               # Number of characteristics, time periods
N    = xret.shape[0]         # Number of assets

print(f"Loaded data: N={N}, L={L}, T={T}")  # Print data dimensions

# -------------------------- IPCA ESTIMATION LOOP -----------------------------

for K in Krange:
    # Print current factor dimension
    print(f"\nEstimating IPCA_GB for K = {K}")

    # ------------------------- SETUP & INITIALIZATION ------------------------
    # Record start time
    t_start = time.time()

    # ALS options (these could be tuned if desired)
    max_iter = 10000
    iter_tol = 1e-6

    # For IPCA, data must be stacked as entity-time for fit
    # Build stacked X_char: shape (N*T, L)
    X_char = np.transpose(Z, (0, 2, 1)).reshape(-1, L)  # (N*T, L)
    y_stack = xret.flatten(order='C')                   # (N*T,)
    indices = np.zeros((N*T, 2), dtype=int)             # (N*T, 2): [entity, time]
    for n in range(N):
        indices[n*T:(n+1)*T, 0] = n     # entity
        indices[n*T:(n+1)*T, 1] = np.arange(T)  # time

    # Mask for valid (non-missing) observations
    valid = LOC.flatten(order='C')                     # (N*T,)

    # Keep only valid (non-NaN) rows
    X_char_valid = X_char[valid]
    y_valid = y_stack[valid]
    indices_valid = indices[valid]

    print(f"Valid observations: {X_char_valid.shape[0]}")  # Show sample size

    # ---------------------- FIT IPCA MODEL (no intercept) -------------------
    # Create and fit the InstrumentedPCA model
    model = InstrumentedPCA(
        n_factors=K,
        intercept=True,
        max_iter=max_iter,
        iter_tol=iter_tol,
        alpha=0.0,           # No regularization for "restricted" model
        l1_ratio=1.0,
        n_jobs=1,
        backend="loky"
    )

    # Fit using panel data
    print("Fitting IPCA model ...")
    model.fit(X=X_char_valid, y=y_valid, indices=indices_valid, data_type="panel")
    print("Model fit complete.")

    # Extract estimated loadings and factors
    GammaBeta, Factor = model.get_factors(label_ind=False)  # GammaBeta: (L, K), Factor: (K, T)
    Lambda = np.mean(Factor, axis=1)                       # Lambda: (K,)

    # Report timing
    elapsed = time.time() - t_start
    print(f"ALS completed in {elapsed:.2f} sec for K={K}")

    # ------------------ Wald-like test for Alpha ------

    print("Start ALpha Wald-like test via bootstrap.")
    pval = model.BS_Walpha(ndraws=1000, n_jobs=3, backend='loky')
    print(f"p-value of alpha for K={K}: {pval}")