# -----------------------------------------------------------------------------
# Script: IPCA_empirical_GBGA.py
#
# Main purpose:
#   - Estimate the unrestricted IPCA model (includes alpha intercept) for
#     multiple factor dimensions K using ALS (alternating least squares).
#   - Compute factor loadings (GammaBeta), alpha, factors, fitted returns,
#     and R² statistics (in-sample and predictive) for both restricted
#     (no alpha) and unrestricted (with alpha) models.
#
# Key assumptions:
#   1. Input data arrays X, W, Z, xret, LOC, Nts, date, N, L, T are stored
#      in NumPy .npz files under the directory "IPCA_KELLY".
#   2. Alpha is represented as a prespecified factor of ones (1×T).
#   3. ALS converges within the specified tolerance for all K.
#
# Inputs:
#   - NPZ files named IPCADATA_FNW36_RNKDMN_CON.npz (and variants) in
#     "IPCA_KELLY/" directory, containing:
#       X     : (L, T) moment matrix
#       W     : (L, L, T) weight tensor
#       Z     : (N, L, T) instrument tensor
#       xret  : (N, T) excess returns
#       LOC   : (N, T) availability mask
#       Nts   : (T,) count of valid obs per period
#       date  : (T,) date vector
#       N, L, T : integers
#
# Outputs:
#   - For each K, saves Results_GBGA_<dataname>_K<k>.npz in "IPCA_KELLY/"
#     containing all estimated parameters and diagnostics.
#   - Prints ALS diagnostics and R² summaries to console.
#
# Requirements:
#   Python 3.x, numpy, scipy, num_IPCA_estimate_ALS, mySOS
# -----------------------------------------------------------------------------

# %% 0. Imports and setup
import os                                                # file path operations
import numpy as np                                       # numerical arrays
import time                                              # timing utilities
from numpy.linalg import svd                     # truncated SVD
from num_IPCA_estimate_ALS import num_IPCA_estimate_ALS  # ALS estimator
from mySOS import mySOS                                  # sum-of-squares function

# %% 1. Estimation parameter grid
# Define factor dimension values to test
K_range = list(range(1, 7))  # K_range = [1,2,3,4,5,6]

# %% 2. Preallocate result containers
# XR2_total_vec, XR2_pred_vec, RR2_total_vec, RR2_pred_vec: (len(K_range),)
XR2_total_vec = np.full((len(K_range),), np.nan)  # unrestricted total R²
XR2_pred_vec  = np.full((len(K_range),), np.nan)  # unrestricted predictive R²
RR2_total_vec = np.full((len(K_range),), np.nan)  # restricted total R²
RR2_pred_vec  = np.full((len(K_range),), np.nan)  # restricted predictive R²

# %% 3. Loop over each factor dimension K
for idx, K in enumerate(K_range):
    # Print current factor dimension
    print(f"[INFO] Current factor dimension K = {K}")

    # -- Load data and ALS options once, when K=first element --
    if K == K_range[0]:
        # Data file name (base) and directory
        dataname = 'IPCADATA_FNW36_RNKDMN_CON'             # base file name
        data_path = os.path.join('../IPCA_KELLY', dataname + '.npz')
        # Load NPZ data
        data = np.load(data_path, allow_pickle=True)       # load file
        # Extract required arrays
        X    = data['X']    # X: (L, T)
        W    = data['W']    # W: (L, L, T)
        Z    = data['Z']    # Z: (N, L, T)
        xret = data['xret'] # xret: (N, T)
        LOC  = data['LOC']  # LOC: (N, T)
        Nts  = data['Nts']  # Nts: (T,)
        date = data['dates'] # date: (T,)
        # Extract dimensions
        N, L_dim, T = Z.shape  # N obs, T time
        L = L_dim                  # number of characteristics
        # ALS options
        als_opt = {
            'MaxIterations': 5000,  # maximum ALS iterations
            'Tolerance': 1e-6       # convergence tolerance
        }
        # Print loaded data info
        print(f"[INFO] Loaded data: X{X.shape}, W{W.shape}, Z{Z.shape}, xret{ xret.shape }")

    # %% 4. Initialization via truncated SVD
    print(f"[INFO] Starting IPCA_empirical_GBGA at {time.ctime()}: K={K}, data={dataname}")
    # Compute rank-K SVD of X: X ≈ U S V'
    GammaBeta_initial, s, v = svd(X)
    GammaBeta_initial = GammaBeta_initial[:, :K]
    s = s[:K]
    v = v[:K, :]
    GB_Old = GammaBeta_initial        # (L, K)
    F_Old = np.diag(s) @ v            # (K, T)

    # %% 5. ALS for restricted model (no alpha)
    print(f"[INFO] ALS (restricted) started at {time.ctime()}")  # ALS start
    start_time = time.time()  # timer start

    # Initialize ALS state
    tol = 1.0                              # initial tolerance
    it = 0                                 # iteration counter
    tols = np.full((500,), np.nan)         # tol trace: (500,)

    # ALS loop
    while it <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
        # Single ALS update step (no alpha)
        GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts)  # shapes: (L,K),(K,T)
        # Compute max absolute change in GB and F
        tol_GB = np.max(np.abs(GB_New - GB_Old))  # scalar
        tol_F  = np.max(np.abs(F_New - F_Old))    # scalar
        tol = max(tol_GB, tol_F)                  # update tol
        # Update state
        GB_Old = GB_New
        F_Old  = F_New
        # Record tol
        tols = np.roll(tols, -1)   # shift left
        tols[-1] = tol             # append current tol
        it += 1                    # increment iteration

    # Assign restricted results
    GB_GB = GB_New.copy()                      # restricted loadings: (L,K)
    GB_F  = F_New.copy()                       # restricted factors: (K,T)
    GB_L  = np.mean(F_New, axis=1)             # mean over time: (K,)
    timing_restricted = {
        'time': time.time() - start_time,      # elapsed seconds
        'iters': it,                           # iterations count
        'tols': tols.copy()                    # tol trace
    }
    print(f"[INFO] ALS (restricted) completed after {it} iterations in "
          f"{timing_restricted['time']:.2f}s at {time.ctime()}")

    # %% 6. ALS for unrestricted model (with alpha intercept)
    start_time = time.time()
    # Initialize with restricted solution, add zero column for alpha
    GB_Old = np.hstack([GB_GB, np.zeros((L, 1))])  # (L, K+1)
    F_Old = GB_F.copy()             # dummy, not used directly

    tol = 1.0
    it = 0
    tols = np.full((500,), np.nan)

    # ALS loop with alpha factor of ones(1,T)
    alpha_factor = np.ones((1, T))  # shape (1,T)
    while it <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
        GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts, alpha_factor)
        tol_GB = np.max(np.abs(GB_New - GB_Old))
        tol_F  = np.max(np.abs(F_New - F_Old))
        tol = max(tol_GB, tol_F)
        GB_Old = GB_New
        F_Old  = F_New
        tols = np.roll(tols, -1)
        tols[-1] = tol
        it += 1

    # Assign unrestricted results
    GBGA_GB = GB_New[:, :-1].copy()             # loadings (L,K)
    GBGA_GA = GB_New[:, -1].copy()              # alpha intercept (L,)
    GBGA_F  = F_New.copy()                      # factors (K,T)
    GBGA_L  = np.mean(GBGA_F, axis=1)           # mean factors: (K,)
    timing_unrestricted = {
        'time': time.time() - start_time,
        'iters': it,
        'tols': tols.copy()
    }
    print(f"[INFO] ALS (unrestricted) completed after {it} iterations in "
          f"{timing_unrestricted['time']:.2f}s at {time.ctime()}")

    # %% 7. Compute fits for both models
    # Preallocate fit arrays
    RFITS_GB        = np.full((N, T), np.nan)  # restricted asset fits
    RFITS_pred_GB   = np.full((N, T), np.nan)  # restricted mean-factor fits
    RFITS_GBGA      = np.full((N, T), np.nan)  # unrestricted asset fits
    RFITS_pred_GBGA = np.full((N, T), np.nan)  # unrestricted mean-factor fits
    XFITS_GB        = np.full((L, T), np.nan)  # restricted managed portfolio fits
    XFITS_pred_GB   = np.full((L, T), np.nan)
    XFITS_GBGA      = np.full((L, T), np.nan)  # unrestricted managed portfolio fits
    XFITS_pred_GBGA = np.full((L, T), np.nan)

    # Loop over time to fill fits
    for t in range(T):
        Z_t = Z[:, :, t]                       # Z_t: (N, L)
        W_t = W[:, :, t]                       # W_t: (L, L)
        # Restricted model fits
        RFITS_GB[:, t]      = Z_t @ (GB_GB @ GB_F[:, t])
        RFITS_pred_GB[:, t] = Z_t @ (GB_GB @ GB_L)
        XFITS_GB[:, t]      = W_t @ (GB_GB @ GB_F[:, t])
        XFITS_pred_GB[:, t] = W_t @ (GB_GB @ GB_L)
        # Unrestricted model fits
        theta_t = GBGA_GA + GBGA_GB @ GBGA_F[:, t]
        theta_L = GBGA_GA + GBGA_GB @ GBGA_L
        RFITS_GBGA[:, t]      = Z_t @ theta_t
        RFITS_pred_GBGA[:, t] = Z_t @ theta_L
        XFITS_GBGA[:, t]      = W_t @ theta_t
        XFITS_pred_GBGA[:, t] = W_t @ theta_L

    # %% 8. Compute R² statistics
    # Mask out missing returns
    xret_mask = np.where(LOC, xret, np.nan)     # masked returns
    totalsos  = mySOS(xret_mask)                # total sum of squares scalar

    # Restricted R²
    resid_restr = xret_mask - RFITS_GB
    resid_predR = xret_mask - RFITS_pred_GB
    RR2_total_GB = 1 - mySOS(resid_restr) / totalsos
    RR2_pred_GB  = 1 - mySOS(resid_predR)   / totalsos

    # Unrestricted R²
    resid_unrestr   = xret_mask - RFITS_GBGA
    resid_predU     = xret_mask - RFITS_pred_GBGA
    RR2_total_GBGA  = 1 - mySOS(resid_unrestr) / totalsos
    RR2_pred_GBGA   = 1 - mySOS(resid_predU)     / totalsos

    # Cross-sectional R² on X
    XR2_total_GB    = 1 - mySOS(X - XFITS_GB)      / mySOS(X)
    XR2_pred_GB     = 1 - mySOS(X - XFITS_pred_GB) / mySOS(X)
    XR2_total_GBGA  = 1 - mySOS(X - XFITS_GBGA)    / mySOS(X)
    XR2_pred_GBGA   = 1 - mySOS(X - XFITS_pred_GBGA)/ mySOS(X)

    print(f"[INFO] Completed estimation at {time.ctime()}")

    # %% 9. Save results for this K
    results_dir = '../IPCA_KELLY'
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir,
               f"Results_GBGA_{dataname}_K{K}.npz")
    np.savez(
        out_file,
        xret=xret,
        W=W,
        date=date,
        LOC=LOC,
        Nts=Nts,
        GB_GB=GB_GB,
        GB_F=GB_F,
        GB_L=GB_L,
        GBGA_GB=GBGA_GB,
        GBGA_GA=GBGA_GA,
        GBGA_F=GBGA_F,
        GBGA_L=GBGA_L,
        RFITS_GB=RFITS_GB,
        RFITS_pred_GB=RFITS_pred_GB,
        RFITS_GBGA=RFITS_GBGA,
        RFITS_pred_GBGA=RFITS_pred_GBGA,
        XFITS_GB=XFITS_GB,
        XFITS_pred_GB=XFITS_pred_GB,
        XFITS_GBGA=XFITS_GBGA,
        XFITS_pred_GBGA=XFITS_pred_GBGA,
        XR2_total_GB=XR2_total_GB,
        XR2_pred_GB=XR2_pred_GB,
        XR2_total_GBGA=XR2_total_GBGA,
        XR2_pred_GBGA=XR2_pred_GBGA,
        RR2_total_GB=RR2_total_GB,
        RR2_pred_GB=RR2_pred_GB,
        RR2_total_GBGA=RR2_total_GBGA,
        RR2_pred_GBGA=RR2_pred_GBGA,
        timing_restricted=timing_restricted,
        timing_unrestricted=timing_unrestricted
    )
    print(f"[INFO] Results saved to '{out_file}'")

    # %% 10. Print key R² results for monitoring
    print("XR2_total_GB  XR2_total_GBGA  XR2_pred_GB  XR2_pred_GBGA")
    print(f"{XR2_total_GB:.4f}         {XR2_total_GBGA:.4f}            "
          f"{XR2_pred_GB:.4f}         {XR2_pred_GBGA:.4f}")
    print("RR2_total_GB  RR2_total_GBGA  RR2_pred_GB  RR2_pred_GBGA")
    print(f"{RR2_total_GB:.4f}         {RR2_total_GBGA:.4f}            "
          f"{RR2_pred_GB:.4f}         {RR2_pred_GBGA:.4f}")

    # %% 11. Store for summary
    XR2_total_vec[idx] = XR2_total_GBGA
    XR2_pred_vec[idx]  = XR2_pred_GBGA
    RR2_total_vec[idx] = RR2_total_GBGA
    RR2_pred_vec[idx]  = RR2_pred_GBGA

# %% 12. Print summary table
print("\n===============================================================")
print("    Unrestricted IPCA In-Sample R² Results Summary             ")
print("---------------------------------------------------------------")
print("  K  |  XR2_tot  XR2_pred  |  RR2_tot  RR2_pred")
print("---------------------------------------------------------------")
for idx, K in enumerate(K_range):
    print(f" { K:2d} |  {XR2_total_vec[idx]:8.4f}  {XR2_pred_vec[idx]:8.4f}  |  "
          f"{RR2_total_vec[idx]:8.4f}  {RR2_pred_vec[idx]:8.4f}")
print("===============================================================")
