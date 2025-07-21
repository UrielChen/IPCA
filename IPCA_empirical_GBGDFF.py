# -----------------------------------------------------------------------------
# Script: IPCA_empirical_GBGDFF.py
#
# Main purpose:
#   - Estimate three IPCA-based models—
#       1. GB   (latent factors only)
#       2. GBGD (hybrid latent + observable factors)
#       3. GD   (observable factors only)
#   - Evaluate in-sample and predictive R² for:
#       • Returns (RR2)
#       • Managed portfolios (QR2)
#       • Characteristics (XR2)
#   - Repeat for a set of Fama-French factor specifications.
#
# Key assumptions:
#   1. Preprocessed data arrays (xret, Q, X, W, Z, LOC, Nts, date, N, L, T)
#      are stored in 'IPCA_KELLY/IPCADATA_FNW36_RNKDMN_CON.npz'.
#   2. Fama-French factor data is in
#      'IPCA_KELLY/F-F_Research_Data_5_Factors_2x3_plusMOM.mat'.
#   3. ALS converges within 5000 iterations at tolerance 1e-6.
#
# Inputs:
#   - 'IPCA_KELLY/IPCADATA_FNW36_RNKDMN_CON.npz'
#   - 'IPCA_KELLY/F-F_Research_Data_5_Factors_2x3_plusMOM.mat'
#
# Outputs:
#   - Files 'IPCA_KELLY/Results_GBGDFF_<dataname>_K<K>_<FFchoice>.npz'
#     with model estimates and R².
#   - Console prints of R² summaries and overall summary table.
#
# Requirements:
#   Python 3.x, numpy, scipy, num_IPCA_estimate_ALS, mySOS
# -----------------------------------------------------------------------------

# %% 0. Imports and setup
import os                                                        # file operations
import time                                                      # timing
import numpy as np                                               # numerical arrays
from numpy.linalg import svd                                     # SVD
from num_IPCA_estimate_ALS import num_IPCA_estimate_ALS          # ALS estimator
from mySOS import mySOS                                          # sum-of-squares helper

# %% 1. Estimation parameters
# Only latent rank K = 5 is used in this script
K = 5  # K: number of latent factors (scalar)

# Fama-French model specifications to test
FFchoices = ['FF1', 'FF3', 'FF4', 'FF5', 'FF6']  # list of length 5

# SummaryTable will collect results for each FFchoice
SummaryTable = []  # list to hold per-model result dicts

# %% 2. Loop over latent rank (only one iteration here)
# Clear any stray variables (not needed in Python)
# for K in [5]:
#     pass  # K is already set

# %% 3. Loop over each Fama-French specification
for j, FFchoice in enumerate(FFchoices):
    # Print which FF model is being run
    print(f"\n==== Running Fama-French Model: {FFchoice} (K={K}) ====")

    # %% 3a. Load input data and ALS options on first iteration
    if j == 0:
        # Data filename and directory
        dataname = 'IPCADATA_FNW36_RNKDMN_CON'  # base name for NPZ
        data_file = os.path.join('../IPCA_KELLY', dataname + '.npz')
        # Load precomputed NPZ data
        data = np.load(data_file, allow_pickle=True)  # NPZ file object
        # Extract arrays with shape comments
        # xret: (N, T) - excess returns
        xret = data['xret']  
        # Q: (L, T) - managed portfolio returns
        Q = data['Q']      
        # X: (L, T) - moment matrix
        X = data['X']      
        # W: (L, L, T) - weight tensor
        W = data['W']      
        # Z: (N, L, T) - instrument tensor
        Z = data['Z']      
        # LOC: (N, T) - availability mask
        LOC = data['LOC']  
        # Nts: (T,) - count of valid observations per period
        Nts = data['Nts']  
        # date: (T,) - date vector
        date = data['dates']
        # Derive dimensions
        N, T = xret.shape  # N obs, T time
        L, _ = X.shape     # L characteristics
        # ALS options
        als_opt = {
            'MaxIterations': 5000,  # maximum ALS iterations
            'Tolerance': 1e-6       # convergence tolerance
        }
        # Print loaded data info
        print(f"[INFO] Loaded data: xret{ xret.shape }, Q{ Q.shape }, X{ X.shape }, W{ W.shape }, Z{ Z.shape }")

    # %% 3b. Load Fama-French factor data
    # Load FF factor data from .npz
    ffdata = np.load('../IPCA_KELLY/F-F_Research_Data_5_Factors_2x3_plusMOM.npz')
    # Expected: ffdata['dates'] [T_FF×1], Mkt_RF, SMB, HML, MOM, RMW, CMA [T_FF×1]

    # Align dates between FF data and sample data
    date, loc1, loc2 = np.intersect1d(ffdata['dates'], date, return_indices=True)
    # date [T×1], loc1 [T×1], loc2 [T×1]

    # %% 3c. Build FF factor matrix based on choice
    # Select FF factors based on FFchoice
    if FFchoice == 'FF1':
        # Market factor only; [1×T]
        FF = ffdata['Mkt_RF'][loc1].reshape(1, -1)
    elif FFchoice == 'FF3':
        # Market, SMB, HML; [3×T]
        FF = np.hstack((ffdata['Mkt_RF'][loc1], ffdata['SMB'][loc1], ffdata['HML'][loc1])).T
    elif FFchoice == 'FF4':
        # Market, SMB, HML, MOM; [4×T]
        FF = np.hstack((ffdata['Mkt_RF'][loc1], ffdata['SMB'][loc1], ffdata['HML'][loc1],
                        ffdata['MOM'][loc1])).T
    elif FFchoice == 'FF5':
        # Market, SMB, HML, RMW, CMA; [5×T]
        FF = np.hstack((ffdata['Mkt_RF'][loc1], ffdata['SMB'][loc1], ffdata['HML'][loc1],
                        ffdata['RMW'][loc1], ffdata['CMA'][loc1])).T
    elif FFchoice == 'FF6':
        # Market, SMB, HML, RMW, CMA, MOM; [6×T]
        FF = np.hstack((ffdata['Mkt_RF'][loc1], ffdata['SMB'][loc1], ffdata['HML'][loc1],
                        ffdata['RMW'][loc1], ffdata['CMA'][loc1], ffdata['MOM'][loc1])).T
    else:
        raise ValueError('Invalid FFchoice')

    # Compute factor means across time
    FF_L = np.mean(FF, axis=1)  # shape (num_factors,)

    # %% 3d. Restrict data matrices to common dates
    # Align all data to the T_common periods
    xret = xret[:, loc2]     # (N, T_common)
    Q    = Q[:, loc2]        # (L, T_common)
    X    = X[:, loc2]        # (L, T_common)
    W    = W[:, :, loc2]     # (L, L, T_common)
    Z    = Z[:, :, loc2]     # (N, L, T_common)
    LOC  = LOC[:, loc2]      # (N, T_common)
    Nts  = Nts[loc2]         # (T_common,)
    date = date[loc2]        # (T_common,)
    # Update T to T_common
    T = xret.shape[1]            # new T

    # Print alignment info
    print(f"[INFO] Aligned data to {T} periods for {FFchoice}")

    # %% 4. SVD initialization for GB-model
    # Compute rank-K SVD of X: X ≈ U S V'
    GammaBeta_initial, s, v = svd(X)
    GammaBeta_initial = GammaBeta_initial[:, :K]
    s = s[:K]
    v = v[:K, :]
    GB_Old = GammaBeta_initial        # (L, K)
    F_Old = np.diag(s) @ v            # (K, T)

    # %% 5. ALS for GB-model (latent factors only)
    print(f"[INFO] ALS for GB-model started at {time.ctime()}")
    start = time.time()                      # start timer
    tol    = 1.0                             # tolerance tracker
    it     = 0                               # iteration counter
    tols   = np.full((500,), np.nan)         # tol trace (500,)

    # ALS loop for GB-model
    while it <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
        GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts)  # (L,K),(K,T)
        # compute max absolute update
        tol_GB = np.max(np.abs(GB_New - GB_Old))
        tol_F  = np.max(np.abs(F_New - F_Old))
        tol    = max(tol_GB, tol_F)
        # update state
        GB_Old = GB_New
        F_Old  = F_New
        # record tol
        tols = np.roll(tols, -1)
        tols[-1] = tol
        it += 1

    # assign GB-model results
    GammaBeta_GB = GB_New.copy()             # (L, K)
    Factor_GB    = F_New.copy()              # (K, T)
    Lambda_GB    = np.mean(Factor_GB, axis=1) # (K,)
    timing_gb = {
        'time': time.time() - start,
        'iters': it,
        'tols': tols.copy()
    }
    print(f"[INFO] GB-model ALS done: {it} iters, {timing_gb['time']:.2f}s")

    # %% 6. ALS for GBGD-model (latent + observable)
    print(f"[INFO] ALS for GBGD-model started at {time.ctime()}")

    start = time.time()
    # initialize: append zeros for each observable factor
    num_obs = FF.shape[0]                                       # number of FF factors
    GB_Old = np.hstack([GammaBeta_GB, np.zeros((L, num_obs))])  # (L, K+num_obs)
    F_Old  = Factor_GB.copy()                                   # (K, T)
    tol    = 1.0
    it     = 0
    tols   = np.full((500,), np.nan)

    # ALS loop for GBGD-model
    while it <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
        GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts, FF)  # include FF
        tol_GB = np.max(np.abs(GB_New - GB_Old))
        tol_F  = np.max(np.abs(F_New - F_Old))
        tol    = max(tol_GB, tol_F)
        GB_Old = GB_New
        F_Old = F_New
        tols = np.roll(tols, -1)
        tols[-1] = tol
        it += 1

    # assign GBGD-model results
    GBGD_GB = GB_New[:, :K].copy()           # (L, K)
    GBGD_GD = GB_New[:, K:].copy()           # (L, num_obs)
    Factor_GBGD = F_New.copy()               # (K, T)
    Lambda_GBGD = np.mean(Factor_GBGD, axis=1) # (K,)
    timing_gbgd = {
        'time': time.time() - start,
        'iters': it,
        'tols': tols.copy()
    }
    print(f"[INFO] GBGD-model ALS done: {it} iters, {timing_gbgd['time']:.2f}s")

    # %% 7. ALS for GD-model (observable factors only)
    print(f"[INFO] ALS for GD-model started at {time.ctime()}")
    start = time.time()
    # initialize: only zeros for latent part
    GB_Old = np.zeros((L, num_obs))          # (L, num_obs)
    F_Old  = np.empty((0, T))                # empty array
    tol    = 1.0; it = 0; tols = np.full((500,), np.nan)

    # ALS loop for GD-model
    while it <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
        GB_New, _ = num_IPCA_estimate_ALS(GB_Old, W, X, Nts, FF)
        tol_GB = np.max(np.abs(GB_New - GB_Old))
        tol    = tol_GB
        GB_Old = GB_New
        tols = np.roll(tols, -1); tols[-1] = tol
        it += 1

    # assign GD-model results
    GD_GD = GB_New.copy()                    # (L, num_obs)
    timing_gd = {
        'time': time.time() - start,
        'iters': it,
        'tols': tols.copy()
    }
    print(f"[INFO] GD-model ALS done: {it} iters, {timing_gd['time']:.2f}s")

    # %% 8. Compute fits and R² for all three models
    # Preallocate fit matrices
    RFITS_GB      = np.full((N, T), np.nan)  # restricted return fits
    RFITS_pred_GB = np.full((N, T), np.nan)
    RFITS_GBGD      = np.full((N, T), np.nan)
    RFITS_pred_GBGD = np.full((N, T), np.nan)
    RFITS_GD      = np.full((N, T), np.nan)
    RFITS_pred_GD = np.full((N, T), np.nan)
    XFITS_GB      = np.full((L, T), np.nan)  # managed portfolio fits
    XFITS_pred_GB = np.full((L, T), np.nan)
    XFITS_GBGD      = np.full((L, T), np.nan)
    XFITS_pred_GBGD = np.full((L, T), np.nan)
    XFITS_GD      = np.full((L, T), np.nan)
    XFITS_pred_GD = np.full((L, T), np.nan)

    # Compute fits period by period
    for t in range(T):
        Z_t = Z[:, :, t]                             # (N, L)
        W_t = W[:, :, t]                             # (L, L)
        # GB-model fits
        RFITS_GB[:, t]      = Z_t @ (GammaBeta_GB @ Factor_GB[:, t])
        RFITS_pred_GB[:, t] = Z_t @ (GammaBeta_GB @ Lambda_GB)
        XFITS_GB[:, t]      = W_t @ (GammaBeta_GB @ Factor_GB[:, t])
        XFITS_pred_GB[:, t] = W_t @ (GammaBeta_GB @ Lambda_GB)
        # GBGD-model fits
        term_t = GBGD_GB @ Factor_GBGD[:, t] + GBGD_GD @ FF[:, t]
        term_L = GBGD_GB @ Lambda_GBGD + GBGD_GD @ FF_L
        RFITS_GBGD[:, t]      = Z_t @ term_t
        RFITS_pred_GBGD[:, t] = Z_t @ term_L
        XFITS_GBGD[:, t]      = W_t @ term_t
        XFITS_pred_GBGD[:, t] = W_t @ term_L
        # GD-model fits
        RFITS_GD[:, t]      = Z_t @ (GD_GD @ FF[:, t])
        RFITS_pred_GD[:, t] = Z_t @ (GD_GD @ FF_L)
        XFITS_GD[:, t]      = W_t @ (GD_GD @ FF[:, t])
        XFITS_pred_GD[:, t] = W_t @ (GD_GD @ FF_L)

    # Q-model fits
    QFITS_GB    = GammaBeta_GB @ Factor_GB                    # (L, T)
    QFITS_pred_GB = np.tile(GammaBeta_GB @ Lambda_GB[:, None], (1, T))
    QFITS_GBGD    = GBGD_GB @ Factor_GBGD + GBGD_GD @ FF
    QFITS_pred_GBGD = np.tile(GBGD_GB @ Lambda_GBGD[:, None] + GBGD_GD @ FF_L[:, None], (1, T))
    QFITS_GD    = GD_GD @ FF
    QFITS_pred_GD = np.tile(GD_GD @ FF_L[:, None], (1, T))

    # Mask unavailable returns
    xret_masked = np.where(LOC, xret, np.nan)  # (N, T)
    totalsos = mySOS(xret_masked)              # scalar

    # RR2 for returns
    RR2_total_GB    = 1 - mySOS(xret_masked - RFITS_GB)      / totalsos
    RR2_pred_GB     = 1 - mySOS(xret_masked - RFITS_pred_GB) / totalsos
    RR2_total_GBGD  = 1 - mySOS(xret_masked - RFITS_GBGD)    / totalsos
    RR2_pred_GBGD   = 1 - mySOS(xret_masked - RFITS_pred_GBGD)/ totalsos
    RR2_total_GD    = 1 - mySOS(xret_masked - RFITS_GD)      / totalsos
    RR2_pred_GD     = 1 - mySOS(xret_masked - RFITS_pred_GD) / totalsos

    # QR2 for managed portfolios
    QR2_total_GB    = 1 - mySOS(Q - QFITS_GB)      / mySOS(Q)
    QR2_pred_GB     = 1 - mySOS(Q - QFITS_pred_GB) / mySOS(Q)
    QR2_total_GBGD  = 1 - mySOS(Q - QFITS_GBGD)    / mySOS(Q)
    QR2_pred_GBGD   = 1 - mySOS(Q - QFITS_pred_GBGD)/ mySOS(Q)
    QR2_total_GD    = 1 - mySOS(Q - QFITS_GD)      / mySOS(Q)
    QR2_pred_GD     = 1 - mySOS(Q - QFITS_pred_GD) / mySOS(Q)

    # XR2 for characteristics
    XR2_total_GB    = 1 - mySOS(X - XFITS_GB)      / mySOS(X)
    XR2_pred_GB     = 1 - mySOS(X - XFITS_pred_GB) / mySOS(X)
    XR2_total_GBGD  = 1 - mySOS(X - XFITS_GBGD)    / mySOS(X)
    XR2_pred_GBGD   = 1 - mySOS(X - XFITS_pred_GBGD)/ mySOS(X)
    XR2_total_GD    = 1 - mySOS(X - XFITS_GD)      / mySOS(X)
    XR2_pred_GD     = 1 - mySOS(X - XFITS_pred_GD) / mySOS(X)

    # Print R² summary for this FF model
    print(f"==== R² Summary for {FFchoice} ====")
    print(f"  Returns RR2_total: GB={RR2_total_GB:.4f}, GBGD={RR2_total_GBGD:.4f}, GD={RR2_total_GD:.4f}")
    print(f"  Returns RR2_pred:  GB={RR2_pred_GB:.4f}, GBGD={RR2_pred_GBGD:.4f}, GD={RR2_pred_GD:.4f}")
    print(f"  Q QR2_total:      GB={QR2_total_GB:.4f}, GBGD={QR2_total_GBGD:.4f}, GD={QR2_total_GD:.4f}")
    print(f"  Q QR2_pred:       GB={QR2_pred_GB:.4f}, GBGD={QR2_pred_GBGD:.4f}, GD={QR2_pred_GD:.4f}")
    print(f"  X XR2_total:      GB={XR2_total_GB:.4f}, GBGD={XR2_total_GBGD:.4f}, GD={XR2_total_GD:.4f}")
    print(f"  X XR2_pred:       GB={XR2_pred_GB:.4f}, GBGD={XR2_pred_GBGD:.4f}, GD={XR2_pred_GD:.4f}")

    # Store results in SummaryTable
    SummaryTable.append({
        'FFchoice': FFchoice,
        'RR2_total_GB': RR2_total_GB,
        'RR2_pred_GB': RR2_pred_GB,
        'RR2_total_GBGD': RR2_total_GBGD,
        'RR2_pred_GBGD': RR2_pred_GBGD,
        'RR2_total_GD': RR2_total_GD,
        'RR2_pred_GD': RR2_pred_GD,
        'QR2_total_GB': QR2_total_GB,
        'QR2_pred_GB': QR2_pred_GB,
        'QR2_total_GBGD': QR2_total_GBGD,
        'QR2_pred_GBGD': QR2_pred_GBGD,
        'QR2_total_GD': QR2_total_GD,
        'QR2_pred_GD': QR2_pred_GD,
        'XR2_total_GB': XR2_total_GB,
        'XR2_pred_GB': XR2_pred_GB,
        'XR2_total_GBGD': XR2_total_GBGD,
        'XR2_pred_GBGD': XR2_pred_GBGD,
        'XR2_total_GD': XR2_total_GD,
        'XR2_pred_GD': XR2_pred_GD
    })

    # %% 9. Save results for this FFchoice
    out_dir = '../IPCA_KELLY'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Results_GBGDFF_{dataname}_K{K}_{FFchoice}.npz")
    np.savez(
        out_file,
        xret=xret, Q=Q, X=X, W=W, Z=Z, LOC=LOC, Nts=Nts, date=date,
        GammaBeta_GB=GammaBeta_GB, Factor_GB=Factor_GB, Lambda_GB=Lambda_GB,
        GBGD_GB=GBGD_GB, GBGD_GD=GBGD_GD, Factor_GBGD=Factor_GBGD, Lambda_GBGD=Lambda_GBGD,
        GD_GD=GD_GD,
        RFITS_GB=RFITS_GB, RFITS_pred_GB=RFITS_pred_GB,
        RFITS_GBGD=RFITS_GBGD, RFITS_pred_GBGD=RFITS_pred_GBGD,
        RFITS_GD=RFITS_GD, RFITS_pred_GD=RFITS_pred_GD,
        XFITS_GB=XFITS_GB, XFITS_pred_GB=XFITS_pred_GB,
        XFITS_GBGD=XFITS_GBGD, XFITS_pred_GBGD=XFITS_pred_GBGD,
        XFITS_GD=XFITS_GD, XFITS_pred_GD=XFITS_pred_GD,
        QFITS_GB=QFITS_GB, QFITS_pred_GB=QFITS_pred_GB,
        QFITS_GBGD=QFITS_GBGD, QFITS_pred_GBGD=QFITS_pred_GBGD,
        QFITS_GD=QFITS_GD, QFITS_pred_GD=QFITS_pred_GD,
        RR2_total_GB=RR2_total_GB, RR2_pred_GB=RR2_pred_GB,
        RR2_total_GBGD=RR2_total_GBGD, RR2_pred_GBGD=RR2_pred_GBGD,
        RR2_total_GD=RR2_total_GD, RR2_pred_GD=RR2_pred_GD,
        QR2_total_GB=QR2_total_GB, QR2_pred_GB=QR2_pred_GB,
        QR2_total_GBGD=QR2_total_GBGD, QR2_pred_GBGD=QR2_pred_GBGD,
        QR2_total_GD=QR2_total_GD, QR2_pred_GD=QR2_pred_GD,
        XR2_total_GB=XR2_total_GB, XR2_pred_GB=XR2_pred_GB,
        XR2_total_GBGD=XR2_total_GBGD, XR2_pred_GBGD=XR2_pred_GBGD,
        XR2_total_GD=XR2_total_GD, XR2_pred_GD=XR2_pred_GD,
        timing_gb=timing_gb, timing_gbgd=timing_gbgd, timing_gd=timing_gd
    )
    print(f"[INFO] Saved results to '{out_file}'")

# %% 10. Print full summary table
import pandas as pd  # for tabular display

# Build DataFrame: index will be metrics, columns the FFchoices
df = pd.DataFrame(SummaryTable)            # each entry is one column’s results
df = df.set_index('FFchoice').T            # transpose so metrics are rows

# Print a clear header
print("\n============== ALL Fama-French R² SUMMARY ==============")

# Display the full table
# This will look like:
#                          FF1      FF3      FF4      FF5      FF6
# RR2_total_GB         0.1234   0.2345   0.3456   0.4567   0.5678
# RR2_pred_GB          0.0234   0.0345   0.0456   0.0567   0.0678
# RR2_total_GBGD       0.1334   0.2445   0.3556   0.4667   0.5778
# ...
print(df.to_string(float_format="%.4f"))
