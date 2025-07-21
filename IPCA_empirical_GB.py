# file: IPCA_empirical_GB.py

# ==============================================================================
# Description:
#   Main script to estimate the "restricted" IPCA model (no alpha intercept).
#   Performs Alternating Least Squares (ALS) estimation of latent factors
#   and loadings, computes in-sample fits and R² for assets and portfolios,
#   and saves results for each factor dimension K.
#
# Assumptions:
#   - Data are stored as .npz with variables:
#       X      : (L, T)    managed-portfolio returns
#       W      : (L, L, T) characteristic-covariance matrices
#       Z      : (N, L, T) characteristic exposures
#       xret   : (N, T)    asset excess returns
#       LOC    : (N, T)    logical mask for non-missing obs
#       Nts    : (T,)      # non-missing stocks per period
#       date   : (T,)      time stamps
#   - num_IPCA_estimate_ALS and mySOS are implemented as described previously.
#
# Inputs:
#   None (all data loaded from .npz files).
# Outputs:
#   Results for each K are saved as .npz in ../Data/Results_GB_<dataname>_K<k>.npz
# ==============================================================================

import datetime
import numpy as np
import os
import time
from scipy.sparse.linalg import svds
from numpy.linalg import svd
from num_IPCA_estimate_ALS import num_IPCA_estimate_ALS  # <-- your ALS function
from mySOS import mySOS  # <-- your sum-of-squares function

# 1) Set estimation parameters and data choice

# Define range of factor dimensions (K) to estimate.
Krange = range(1, 7)  # K = 1 to 6

# Base data filename (without extension)
dataname = 'IPCADATA_FNW36_RNKDMN_CON'

# ALS convergence options
MaxIterations = 10000  # maximum ALS iterations
Tolerance = 1e-6       # convergence threshold

# Directory for data and results
DATA_DIR = os.path.join('..', 'IPCA_Kelly')

# Loop over each desired K value
for K in Krange:
    # Display current factor dimension
    print(f'Estimating IPCA_GB for K = {K}')

    # On first iteration: load data from .npz
    if K == Krange[0]:
        # Clear workspace variables (handled by scoping in Python)
        # Load the input data arrays
        data = np.load(os.path.join(DATA_DIR, f"{dataname}.npz"), allow_pickle=True)
        # Extract all variables from the data file
        X = data['X']             # (L, T)
        W = data['W']             # (L, L, T)
        Z = data['Z']             # (N, L, T)
        xret = data['xret']       # (N, T)
        LOC = data['LOC']         # (N, T)
        Nts = data['Nts']         # (T,)
        date = data['dates']       # (T,)
        print("Loaded input data.")  # Confirm data loaded

    # 2) Initialize estimation

    # Inform user of start
    print(f"IPCA_empirical_GB starting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, K={K}, data={dataname}")

    # Use truncated SVD of X for initial GammaBeta guess
    # GammaBeta_initial: (L, K)
    GammaBeta_initial, s, v = svd(X)
    GammaBeta_initial = GammaBeta_initial[:, :K]
    s = s[:K]
    v = v[:K, :]

    # 3) Alternating Least Squares (ALS)

    # Mark ALS start time
    print(f" ALS started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tic = time.time()  # start timer

    # Warm-start: if previous Factor exists, expand GammaBeta & Factor
    if False:
        # If previous iteration variables exist, use them to warm-start
        GB_Old = np.hstack([GammaBeta, GammaBeta_initial[:, K-1][:, None]])  # (L, K)
        F_Old = np.vstack([Factor, np.ones((1, X.shape[1]))])                # (K, T)
    else:
        GB_Old = GammaBeta_initial        # (L, K)
        F_Old = np.diag(s) @ v            # (K, T)

    # Initialize convergence trackers
    tol = 1.0                             # scalar
    iter = 0                              # scalar
    tols = np.full(500, np.nan)           # (500,)

    # ALS main loop
    while iter <= MaxIterations and tol > Tolerance:
        # One ALS step: update loadings & factors
        GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts)
        # Compute maximum change for convergence
        tol = np.max(np.concatenate([np.abs(GB_New.ravel() - GB_Old.ravel()),
                                     np.abs( F_New.ravel() -  F_Old.ravel())]))
        # Shift tolerance history
        tols = np.append(tols[1:], tol)
        # Update old values for next iteration
        GB_Old = GB_New
        F_Old = F_New
        iter += 1
        # Optionally print iteration progress every 50 steps
        if iter % 50 == 0:
            print(f"  ALS iteration {iter}, tol={tol:.2e}")

    # Final estimates
    GammaBeta = GB_New                    # (L, K)
    Factor = F_New                        # (K, T)
    Lambda = np.mean(Factor, axis=1)      # (K,)

    # Record ALS timing & iterations
    als_time = time.time() - tic
    timing = {'als_xsvd': {'time': als_time, 'iter': iter, 'tols': tols.copy()}}
    print(f" ALS completed in {iter} iters, {als_time:.2f} sec")  # ALS convergence print

    # 4) Compute in-sample fits and R²

    # Pre-allocate fit matrices
    RFITS_GB      = np.full_like(xret, np.nan)   # (N, T)
    RFITS_pred_GB = np.full_like(xret, np.nan)   # (N, T)
    XFITS_GB      = np.full_like(X, np.nan)      # (L, T)
    XFITS_pred_GB = np.full_like(X, np.nan)      # (L, T)

    # Loop over time to compute realized & predictive fits
    for t in range(xret.shape[1]):
        # asset-level realized fit
        RFITS_GB[:, t]      = Z[:, :, t] @ GammaBeta @ Factor[:, t]
        # asset-level predictive fit
        RFITS_pred_GB[:, t] = Z[:, :, t] @ GammaBeta @ Lambda
        # managed portfolio realized fit
        XFITS_GB[:, t]      = W[:, :, t] @ GammaBeta @ Factor[:, t]
        # managed portfolio predictive fit
        XFITS_pred_GB[:, t] = W[:, :, t] @ GammaBeta @ Lambda

    # Mask out missing returns
    xret_masked = xret.copy()
    xret_masked[~LOC] = np.nan

    # Compute sum of squares
    totalsos = mySOS(xret_masked)  # scalar

    # Asset-level R²
    RR2_total_GB = 1 - mySOS(xret_masked[LOC] - RFITS_GB[LOC]) / totalsos
    RR2_pred_GB  = 1 - mySOS(xret_masked[LOC] - RFITS_pred_GB[LOC]) / totalsos

    # Managed portfolio R²
    XR2_total_GB = 1 - mySOS(X - XFITS_GB) / mySOS(X)
    XR2_pred_GB  = 1 - mySOS(X - XFITS_pred_GB) / mySOS(X)

    print(' R² results:')  # Print block for R²
    print(f'  Managed Total/Pred = [{XR2_total_GB:.4f}, {XR2_pred_GB:.4f}]')
    print(f'  Asset   Total/Pred = [{RR2_total_GB:.4f}, {RR2_pred_GB:.4f}]')

    # 5) Save results for current K
    outdict = {
        'xret': xret, 'W': W, 'date': date, 'LOC': LOC, 'Nts': Nts,
        'GammaBeta': GammaBeta, 'Factor': Factor, 'Lambda': Lambda,
        'RFITS_GB': RFITS_GB, 'RFITS_pred_GB': RFITS_pred_GB,
        'XFITS_GB': XFITS_GB, 'XFITS_pred_GB': XFITS_pred_GB,
        'RR2_total_GB': RR2_total_GB, 'RR2_pred_GB': RR2_pred_GB,
        'XR2_total_GB': XR2_total_GB, 'XR2_pred_GB': XR2_pred_GB,
        'timing': timing
    }
    outfilename = os.path.join(DATA_DIR, f'Results_GB_{dataname}_K{K}.npz')
    np.savez(outfilename, **outdict)
    print(f"Saved results to {outfilename}\n")  # Confirm file saved

# End Krange loop