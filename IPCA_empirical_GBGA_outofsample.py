# -----------------------------------------------------------------------------
# Script: IPCA_empirical_GBGA_outofsample.py
#
# Main purpose:
#   - Perform out-of-sample IPCA estimation for the GB (latent-only) and
#     GBGA (latent + intercept) models in a rolling-window fashion.
#   - For each date t ≥ startindex, fit models on data up to t, then:
#       • Forecast returns (OOSRFITS_pred_GB) and characteristics (OOSXFITS_pred_GB)
#       • Compute true factor exposures (OOSRealFact) via one-step ahead X and W
#       • Compute arbitrage portfolio returns (OOSARBPTF)
#       • Compute tangency portfolio returns (OOSRealTan)
#
# Key assumptions:
#   1. Preprocessed data (X, W, Nts, Z, xret, LOC, date, N, L, T) are stored
#      in 'IPCA_KELLY/IPCADATA_FNW36_RNKDMN_CON.npz'.
#   2. ALS converges within 5000 iterations at tolerance 1e-6.
#   3. Functions available for import:
#        - num_IPCA_estimate_ALS
#        - tanptf
#        - tanptfnext
#
# Inputs:
#   - dataname NPZ file in 'IPCA_KELLY/' directory.
#   - startindex: first index for out-of-sample forecasting.
#
# Outputs:
#   - Saves results to
#     'IPCA_KELLY/Results_GBGA_outofsample_<dataname>_K<K>.npz'
#   - Prints progress and key diagnostics to console.
# -----------------------------------------------------------------------------

# %% 0. Imports
import os                                                        # for file paths
import time                                                      # for timing
import numpy as np                                               # for numerical arrays
from scipy.sparse.linalg import svds                             # truncated SVD
from num_IPCA_estimate_ALS import num_IPCA_estimate_ALS          # ALS estimator
from tanptf import tanptf                                        # in-sample tangency
from tanptfnext import tanptfnext                                # out-of-sample tangency

# %% 1. Set estimation parameters and data choice
# Clear variables (not needed in Python, shown for completeness)
# dataname: base name of NPZ file (string)
dataname = 'IPCADATA_FNW36_RNKDMN_CON'  
# K: number of latent factors (scalar)
for K in range(1, 7):  
    # ALS options dictionary
    als_opt = {
        'MaxIterations': 5000,   # maximum ALS iterations
        'Tolerance': 1e-6        # convergence tolerance
    }
    # startindex: first time index (integer) for out-of-sample estimation
    startindex = 60  

    # %% 2. Start estimation printout
    print(f"[INFO] IPCA_empirical_GB starting at {time.ctime()}: K={K}, data={dataname}")

    # %% 3. Load full-sample data
    # Load NPZ file containing X, W, Nts, Z, xret, LOC, date, N, L, T
    data_path = os.path.join('../IPCA_KELLY', dataname + '.npz')
    data = np.load(data_path, allow_pickle=True)  

    # Extract arrays with sizes:
    X_full   = data['X']       # X_full: (L, T) moment matrix
    W_full   = data['W']       # W_full: (L, L, T) weight tensor
    Nts_full = data['Nts']     # Nts_full: (T,) valid obs count
    Z        = data['Z']       # Z: (N, L, T) instrument tensor
    xret     = data['xret']    # xret: (N, T) excess returns
    LOC      = data['LOC']     # LOC: (N, T) availability mask
    date     = data['dates']    # date: (T,) date vector

    # Dimensions
    L, T = X_full.shape        # L: # characteristics, T: # time periods
    N, _ = xret.shape          # N: # assets

    print(f"[INFO] Loaded data: X{X_full.shape}, W{W_full.shape}, Nts{Nts_full.shape}")

    # %% 4. Copy full-sample variables for rolling use
    bigX   = X_full.copy()     # bigX: (L, T)
    bigW   = W_full.copy()     # bigW: (L, L, T)
    bigNts = Nts_full.copy()   # bigNts: (T,)

    # %% 5. Preallocate out-of-sample result arrays
    OOSRFITS_pred_GB = np.full((N, T), np.nan)   # [N×T]
    OOSXFITS_pred_GB = np.full((L, T), np.nan)   # [L×T]
    OOSRealFact      = np.full((K, T), np.nan)   # [K×T]
    OOSReal_ArbTan   = np.full((T,), np.nan)     # [T,]
    OOSReal_Tan      = np.full((T,), np.nan)     # [T,]
    OOSRFITS_GB      = np.full((N, T), np.nan)   # [N×T]
    OOSXFITS_GB      = np.full((L, T), np.nan)   # [L×T]
    OOSARBPTF        = np.full((T,), np.nan)     # [T,]

    print("[INFO] Preallocated OOS result arrays")

    # %% 6. Rolling out-of-sample loop
    for t in range(startindex, T):  # t from startindex to T-1 inclusive
        # Subsample up to t (columns 0 to t-1 in Python)
        X   = bigX[:, :t]            # X: (L, t)
        W   = bigW[:, :, :t]         # W: (L, L, t)
        Nts = bigNts[:t]             # Nts: (t,)

        # Truncated SVD initialization: X ≈ U S V'
        U, s, v = svds(X, K)    # U:(L,K), s:(K,), v:(K,t)
        GammaBeta_XSVD = U            # (L, K)
        F_init = np.diag(s) @ v  # F_init: (K, t)

        # --- ALS for GB-model (latent only) ---
        tic = time.time()              # start timer
        if t == startindex:
            GB_Old = GammaBeta_XSVD.copy()   # GB_Old: (L, K)
            F_Old  = F_init.copy()           # F_Old: (K, t)
        else:
            GB_Old = GammaBeta.copy()        # GB_Old: (L, K) from previous step
            # Append last SVD-derived factor for initialization
            last_factor = (s * v[:, -1])[:, np.newaxis]  # (K,1)
            F_Old = np.hstack([FactorGB, last_factor])         # (K, t)
        tol = 1.0
        iter_count = 0
        # ALS iteration loop
        while iter_count <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
            GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts)  # ALS step
            # Compute max update magnitude
            tol = max(np.max(np.abs(GB_New - GB_Old)), np.max(np.abs(F_New - F_Old)))
            F_Old[:] = F_New
            GB_Old[:] = GB_New
            iter_count += 1
        # Assign GB-model results at time t
        FactorGB    = F_New.copy()               # FactorGB: (K, t)
        GammaBetaGB = GB_New.copy()              # GammaBetaGB: (L, K)
        LambdaGB    = np.mean(FactorGB, axis=1)  # LambdaGB: (K,)
        print(f"[INFO] NUM_ALS for GB-model at t={t} done in {iter_count} iters, {time.time()-tic:.2f}s")

        # --- ALS for GBGA-model (latent + intercept) ---
        tic = time.time()              # new timer
        if t == startindex:
            GB_Old = np.hstack([GammaBeta_XSVD, np.zeros((L,1))])  # (L, K+1)
            F_Old  = F_init.copy()                                  # (K, t)
        else:
            GB_Old = np.hstack([GammaBeta, GammaAlpha[:, np.newaxis]])  # (L, K+1)
            last_factor = (s * v[:, -1])[:, np.newaxis]            # (K,1)
            F_Old = np.hstack([Factor, last_factor])                    # (K, t)
        tol = 1.0
        iter_count = 0
        # ALS iteration with alpha intercept
        while iter_count <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
            GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts, np.ones((1, t)))  # ones(1,t)
            tol = max(np.max(np.abs(GB_New - GB_Old)), np.max(np.abs(F_New - F_Old)))
            F_Old[:] = F_New
            GB_Old[:] = GB_New
            iter_count += 1
        # Assign GBGA-model results
        GammaBeta  = GB_New[:, :-1].copy()      # GammaBeta: (L, K)
        GammaAlpha = GB_New[:, -1].copy()       # GammaAlpha: (L,)
        Factor     = F_New.copy()               # Factor: (K+1, t)
        Lambda     = np.mean(Factor, axis=1)    # Lambda: (K+1,)
        print(f"[INFO] NUM_ALS for GBGA-model at t={t} done in {iter_count} iters, {time.time()-tic:.2f}s")

        # --- Out-of-sample arbitrage portfolio return at t+1 ---
        # Z_next: (M, L), where M = number of valid obs at t+1
        loc_next = LOC[:, t]                   # loc_next: (N,)
        Z_next = Z[loc_next, :, t]             # (M, L)
        # Compute portfolio weights: alpha' * (Z'Z)^{-1} * Z'
        ZZ = Z_next.T @ Z_next                   # (L, L)
        w_alpha = GammaAlpha @ np.linalg.inv(ZZ) @ Z_next.T  # (L,) @ (L,L) @ (L,M) → (M,)
        r_next = xret[loc_next, t]             # (M,)
        OOSARBPTF[t] = w_alpha @ r_next        # scalar
        # Forecasted return for GB: Z_{t+1} GammaBetaGB LambdaGB
        OOSRFITS_pred_GB[:, t] = (Z[:, :, t] @ GammaBetaGB @ LambdaGB)  # vectorized

        # --- True factor exposure at t+1 (GB) via one-step ahead X, W ---
        x_next = bigX[:, t]                    # x_next: (L,)
        W_next = bigW[:, :, t]                 # W_next: (L, L)
        lhs = GammaBetaGB.T @ W_next @ GammaBetaGB  # (K, K)
        rhs = GammaBetaGB.T @ x_next                # (K,)
        OOSRealFact[:, t] = np.linalg.solve(lhs, rhs)  # (K,)

        # --- Tangency portfolio returns ---
        # Compute ts_arbptf: in-sample arb portfolio returns up to t
        ts_arbptf = np.full((t,), np.nan)         # (t,)
        for tt in range(t):
            loc_tt = LOC[:, tt]                   # (N,)
            Z_tt = Z[loc_tt, :, tt]               # (M_tt, L)
            ZZ_tt = Z_tt.T @ Z_tt                 # (L, L)
            w_a = GammaAlpha @ np.linalg.inv(ZZ_tt) @ Z_tt.T  # (M_tt,)
            ts_arbptf[tt] = w_a @ xret[loc_tt, tt]            # scalar
        # Compute tangency returns at t+1 using tanptfnext
        OOSReal_ArbTan[t], _ = tanptfnext(
            np.vstack([FactorGB, ts_arbptf]).T,                             # previous-factor & arb
            np.vstack([OOSRealFact[:, t].reshape((-1,1)), OOSARBPTF[t]]).T  # new exposures & arb
        )
        OOSReal_Tan[t], _ = tanptfnext(
            FactorGB.T,
            OOSRealFact[:, t].reshape((-1,1)).T
        )

    # %% 7. Restore full-sample variables
    X   = bigX.copy()      # X: (L, T)
    W   = bigW.copy()      # W: (L, L, T)
    Nts = bigNts.copy()    # Nts: (T,)

    print(f"[INFO] Out-of-sample estimation completed at {time.ctime()}")

    # %% 8. Save results
    out_dir = '../IPCA_KELLY'
    os.makedirs(out_dir, exist_ok=True)  
    out_file = os.path.join(out_dir, f"Results_GBGA_outofsample_{dataname}_K{K}.npz")
    np.savez(
        out_file,
        xret=xret, W=W, X=X, date=date, LOC=LOC, Nts=Nts,
        OOSRFITS_pred_GB=OOSRFITS_pred_GB,
        OOSXFITS_pred_GB=OOSXFITS_pred_GB,
        OOSRealFact=OOSRealFact,
        OOSReal_ArbTan=OOSReal_ArbTan,
        OOSReal_Tan=OOSReal_Tan,
        OOSRFITS_GB=OOSRFITS_GB,
        OOSXFITS_GB=OOSXFITS_GB,
        OOSARBPTF=OOSARBPTF
    )
    print(f"[INFO] Saved results to '{out_file}'")
