# -----------------------------------------------------------------------------
# Script: IPCA_empirical_GBGA_outofsample.py
#
# Main purpose:
#   - Perform out-of-sample IPCA estimation for the GB (latent-only) and
#     GBGA (latent + intercept) models.
#   - Rolling estimation: for each date t ≥ startindex, fit on data up to t,
#     then compute out-of-sample forecasts, arbitrage portfolio returns, and
#     true factor exposures and tangency portfolio returns at t+1.
#
# Key assumptions:
#   1. Preprocessed data (X, W, Z, xret, LOC, Nts, date, N, L, T) are stored
#      in 'IPCA_KELLY/IPCADATA_FNW36_RNKDMN_CON.npz'.
#   2. Functions available:
#        - num_IPCA_estimate_ALS(…)
#        - tanptf(…)
#        - tanptfnext(…)
#   3. ALS converges within 5000 iterations at tolerance 1e-6.
#
# Inputs:
#   - dataname NPZ file in 'IPCA_KELLY/'
#   - startindex: first date for out-of-sample forecasting
#
# Outputs:
#   - Forecasts and diagnostics saved to
#     'IPCA_KELLY/Results_GBGA_outofsample_<dataname>_K<K>.npz'
#   - Key progress printed to console.
# -----------------------------------------------------------------------------

# %% 0. Imports and setup
import os                                                      # file operations
import time                                                    # timing utilities
import numpy as np                                             # numerical arrays
from numpy.linalg import svd                                   # SVD
from num_IPCA_estimate_ALS import num_IPCA_estimate_ALS        # ALS estimator
from tanptf import tanptf                                      # in-sample tangency
from tanptfnext import tanptfnext                              # out-of-sample tangency

# %% 1. Parameters and data choice
# Base data filename (without extension)
dataname = 'IPCADATA_FNW36_RNKDMN_CON'  # dataname: string
# Number of latent factors
for K in range(3, 7):                                   # K: scalar
    # ALS options
    als_opt = {'MaxIterations': 5000,       # max iterations
            'Tolerance': 1e-6}            # convergence tol
    # Out-of-sample start index
    startindex = 60                         # integer index for first OOS

    # %% 2. Load full dataset
    data_path = os.path.join('../IPCA_KELLY', dataname + '.npz')
    data = np.load(data_path, allow_pickle=True)  # load NPZ

    # Extract full-sample arrays with shapes
    X_full    = data['X']    # X_full: (L, T)
    W_full    = data['W']    # W_full: (L, L, T)
    Nts_full  = data['Nts']  # Nts_full: (T,)
    Z         = data['Z']    # Z: (N, L, T)
    xret      = data['xret'] # xret: (N, T)
    LOC       = data['LOC']  # LOC: (N, T)
    date      = data['dates'] # date: (T,)

    # Derive dimensions
    L, T = X_full.shape       # L: # characteristics, T: # time
    N, _ = xret.shape         # N: # assets

    print(f"[INFO] Loaded full data X{X_full.shape}, W{W_full.shape}, Nts{Nts_full.shape}")

    # %% 3. Preallocate out-of-sample containers
    OOSRFITS_pred_GB   = np.full((N, T), np.nan)    # restricted forecasted returns (N, T)
    OOSXFITS_pred_GB   = np.full((L, T), np.nan)    # restricted forecasted X (L, T)
    OOSRealFact_GB     = np.full((K, T), np.nan)    # true GB factor exposures (K, T)
    OOSRealTan_GB      = np.full((T,), np.nan)      # in-sample tangency returns (T,)
    OOSRFITS_GB        = np.full((N, T), np.nan)    # restricted in-sample fits (N, T)
    OOSXFITS_GB        = np.full((L, T), np.nan)    # restricted in-sample X fits (L, T)
    OOSARBPTF         = np.full((T,), np.nan)       # out-of-sample arbitrage portfolio returns (T,)

    print("[INFO] Preallocated OOS arrays")

    # %% 4. Rolling out-of-sample loop
    for t in range(startindex, T-1):
        # Subsample up to t (inclusive)
        X  = X_full[:, :t+1]           # X: (L, t+1)
        W  = W_full[:, :, :t+1]        # W: (L, L, t+1)
        Nts = Nts_full[:t+1]           # Nts: (t+1,)

        # --- SVD initialization ---
        # Compute rank-K SVD of X: X ≈ U S V'
        GammaBeta_initial, s, v = svd(X)
        GammaBeta_initial = GammaBeta_initial[:, :K]
        s = s[:K]
        v = v[:K, :]
        F_init = np.diag(s) @ v            # (K, T)

        # --- ALS for GB-model ---
        start = time.time()
        if t == startindex:
            GB_Old = GammaBeta_initial.copy()             # (L, K)
            F_Old  = F_init.copy()                       # (K, t+1)
        else:
            GB_Old = GammaBeta.copy()                    # previous GB emd
            # append last time's SVD row as new factor
            last_svd = s[:K] * v[:K, -1]
            F_Old = np.hstack([FactorGB, last_svd[:, None]])
        tol, it = 1.0, 0
        # ALS loop (GB)
        while it <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
            GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts)
            tol = max(np.max(np.abs(GB_New - GB_Old)), np.max(np.abs(F_New - F_Old)))
            GB_Old[:] = GB_New
            F_Old[:]  = F_New
            it += 1
        FactorGB    = F_New.copy()                     # (K, t+1)
        GammaBetaGB = GB_New.copy()                    # (L, K)
        LambdaGB    = np.mean(FactorGB, axis=1)        # (K,)
        print(f"[INFO] GB ALS t={t} done in {it} iters, {time.time()-start:.2f}s")

        # --- ALS for GBGA-model ---
        start = time.time()
        if t == startindex:
            GB_Old = np.hstack([GammaBeta_initial, np.zeros((L,1))])  # (L, K+1)
            F_Old  = F_init.copy()                                 # (K, t+1)
        else:
            GB_Old = np.hstack([GammaBetaGB, GammaAlpha[:, None]]) # (L, K+1)
            last_svd = s[:K] * v[:K, -1]
            F_Old = np.hstack([Factor, last_svd[:, None]])                  # (K+1, t+1)
        tol, it = 1.0, 0
        # ALS loop (GBGA)
        alpha_factor = np.ones((1, t+1))
        while it <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
            GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W, X, Nts, alpha_factor)
            tol = max(np.max(np.abs(GB_New - GB_Old)), np.max(np.abs(F_New - F_Old)))
            GB_Old[:] = GB_New
            F_Old[:]  = F_New
            it += 1
        GammaBeta = GB_New[:, :K].copy()              # (L, K)
        GammaAlpha= GB_New[:, -1].copy()              # (L,)
        Factor    = F_New.copy()                      # (K+1, t+1)
        Lambda    = np.mean(F_New, axis=1)            # (K+1,)
        print(f"[INFO] GBGA ALS t={t} done in {it} iters, {time.time()-start:.2f}s")

        # --- Out-of-sample arbitrage portfolio return at t+1 ---
        Z_next = Z[:, :, t+1]                         # (N, L)
        loc_next = LOC[:, t+1]                        # (N,)
        # compute weight vector: alpha' * (Z'Z)⁻¹ Z'
        ZZ = Z_next[loc_next].T @ Z_next[loc_next]     # (L, L)
        w_alpha = GammaAlpha @ np.linalg.inv(ZZ) @ Z_next[loc_next].T
        r_next = xret[loc_next, t+1]                   # (sum(loc_next),)
        OOSARBPTF[t+1] = w_alpha @ r_next               # scalar

        # --- True factor exposure at t+1 using GB factors ---
        W_next = W_full[:, :, t+1]                     # (L, L)
        x_next = X_full[:, t+1]                        # (L,)
        # solve (Γ' W Γ) β = Γ' x
        lhs = GammaBetaGB.T @ W_next @ GammaBetaGB      # (K, K)
        rhs = GammaBetaGB.T @ x_next                    # (K,)
        OOSRealFact_GB[:, t+1] = np.linalg.solve(lhs, rhs)

        # --- Tangency portfolio returns ---
        # compute in-sample arb pts returns up to t
        ts_arb = np.full((1, t+1), np.nan)                # (1, t+1)
        for tt in range(t+1):
            Z_tt = Z[:, :, tt][LOC[:, tt]]
            ZZ_tt = Z_tt.T @ Z_tt
            w_a = GammaAlpha @ np.linalg.inv(ZZ_tt) @ Z_tt.T
            ts_arb[0, tt] = w_a @ xret[LOC[:, tt], tt]
        # tangency out-of-sample
        OOSRealTan_GB[t+1], _ = tanptfnext(
            np.vstack([FactorGB[:, :-1], ts_arb[:, :-1]]).T,
            np.vstack([OOSRealFact_GB[:, t+1].reshape((-1,1)), OOSARBPTF[t+1]]).T
        )

    # %% 5. Restore full-sample X, W, Nts
    X  = X_full.copy()
    W  = W_full.copy()
    Nts= Nts_full.copy()

    print("[INFO] Out-of-sample estimation completed")

    # %% 6. Save results
    out_dir = '../IPCA_KELLY'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Results_GBGA_outofsample_{dataname}_K{K}.npz")
    np.savez(
        out_file,
        xret=xret, W=W, X=X, date=date, LOC=LOC, Nts=Nts,
        OOSRFITS_pred_GB=OOSRFITS_pred_GB,
        OOSXFITS_pred_GB=OOSXFITS_pred_GB,
        OOSRealFact_GB=OOSRealFact_GB,
        OOSRealTan_GB=OOSRealTan_GB,
        OOSRFITS_GB=OOSRFITS_GB,
        OOSXFITS_GB=OOSXFITS_GB,
        OOSARBPTF=OOSARBPTF
    )
    print(f"[INFO] Saved out-of-sample results to '{out_file}'")
