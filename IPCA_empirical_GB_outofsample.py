# -----------------------------------------------------------------------------
# Script: IPCA_empirical_GB_outofsample.py
#
# Main purpose:
#   - Perform recursive out-of-sample estimation of the latent-only IPCA model (GB).
#   - For each period t ≥ startindex, fit the IPCA model on data up to t,
#     then predict returns and characteristics at t+1.
#   - Compute realized out-of-sample factors and tangency portfolio returns.
#   - Calculate out-of-sample R² for returns and characteristics, and Sharpe ratio.
#
# Key assumptions:
#   1. Preprocessed data arrays (X, W, Z, LOC, Nts, xret, date, N, L, T)
#      are stored in '../IPCA_KELLY/IPCADATA_FNW36_RNKDMN_CON.npz'.
#   2. ALS converges within 5000 iterations at tolerance 1e-6.
#   3. Utility functions available:
#        from num_IPCA_estimate_ALS import num_IPCA_estimate_ALS
#        from mySOS import mySOS
#        from tanptf import tanptf
#        from tanptfnext import tanptfnext
#
# Inputs:
#   - dataname NPZ file in directory '../IPCA_KELLY'
#   - startindex: first period for out-of-sample forecasting
#
# Outputs:
#   - NPZ file '../IPCA_KELLY/Results_GB_outofsample_<dataname>_K<K>.npz'
#     containing all OOS fits and diagnostics.
#   - Console prints of progress, diagnostics, and OOS performance summary.
# -----------------------------------------------------------------------------

# %% 0. Imports
import os                                                      # file/directory operations
import time                                                    # timing utilities
import numpy as np                                             # numerical arrays
from scipy.sparse.linalg import svds                             # truncated SVD
from num_IPCA_estimate_ALS import num_IPCA_estimate_ALS         # ALS estimator
from mySOS import mySOS                                         # sum-of-squares helper
from tanptf import tanptf                                       # in-sample tangency
from tanptfnext import tanptfnext                               # out-of-sample tangency

# %% 1. Set estimation parameters and data choice
# Clear workspace equivalent (no-op in Python)
# dataname: base filename for input NPZ (no extension)
dataname = 'IPCADATA_FNW36_RNKDMN_CON'  # string
# Model rank K (number of latent factors)
K = 6                                    # integer
# ALS options for convergence
als_opt = {
    'MaxIterations': 5000,               # scalar
    'Tolerance': 1e-6                    # scalar
}
# startindex: minimum training window before OOS begins
startindex = 60                         # integer

# %% 2. Print start message
print(f"[INFO] IPCA_empirical_GB starting at {time.ctime()}: K={K}, data={dataname}")

# %% 3. Load data
# Construct path to NPZ file
data_path = os.path.join('../IPCA_KELLY', dataname + '.npz')
# Load NPZ containing X, W, Z, LOC, Nts, xret, date
data = np.load(data_path, allow_pickle=True)

# Extract arrays with shapes
X   = data['X']        # X: [L x T]
W   = data['W']        # W: [L x L x T]
Z   = data['Z']        # Z: [N x L x T]
LOC = data['LOC']      # LOC: [N x T]
Nts = data['Nts']      # Nts: [T,]
xret= data['xret']     # xret: [N x T]
date= data['dates']     # date: [T,]

# Derive dimensions
L, T = X.shape          # L: # characteristics, T: # time periods
N, _ = xret.shape       # N: # assets

print(f"[INFO] Loaded data shapes -- X:{X.shape}, W:{W.shape}, Z:{Z.shape}, xret:{xret.shape}")

# %% 4. Save full-sample copies for rolling window
bigX   = X.copy()       # bigX: [L x T]
bigW   = W.copy()       # bigW: [L x L x T]
bigNts = Nts.copy()     # bigNts: [T,]

# %% 5. Preallocate out-of-sample result arrays
OOSRFITS_pred_GB = np.full((N, T), np.nan)   # [N x T]
OOSXFITS_pred_GB = np.full((L, T), np.nan)   # [L x T]
OOSRealFact      = np.full((K, T), np.nan)   # [K x T]
OOSRealTan       = np.full((T,), np.nan)     # [T,]
OOSRFITS_GB      = np.full((N, T), np.nan)   # [N x T]
OOSXFITS_GB      = np.full((L, T), np.nan)   # [L x T]

print("[INFO] Preallocated out-of-sample arrays")

# %% 6. Rolling out-of-sample estimation
for t in range(startindex, T):  # t = 60,...,T-1
    # Prepare training data up to time t (excludes index t)
    X_train   = bigX[:, :t]       # [L x t]
    W_train   = bigW[:, :, :t]    # [L x L x t]
    Nts_train = bigNts[:t]        # [t,]

    # SVD initialization for ALS
    U, s, v = svds(X_train, K)          # U:[L x K], s:[K], v:[K x t]
    GammaBeta_XSVD = U.copy()                 # [L x K]
    F_init = np.diag(s) @ v             # [K x t]

    # Print SVD diagnostic on first iteration
    if t == startindex:
        print(f"[DEBUG] SVD init at t={t}: GammaBeta_XSVD shape {GammaBeta_XSVD.shape}")

    # --- ALS for GB-model ---
    tic = time.time()                           # start timer
    if t == startindex:
        GB_Old = GammaBeta_XSVD.copy()          # [L x K]
        F_Old  = F_init.copy()                  # [K x t]
    else:
        GB_Old = GammaBetaGB.copy()               # [L x K]
        # append last singular factor for continuity
        last_factor = (s * v[:, -1])[:, np.newaxis]  # [K x 1]
        F_Old = np.hstack([FactorGB, last_factor])         # [K x t]
    tol = 1.0
    iter_count = 0

    # ALS iteration loop
    while iter_count <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
        GB_New, F_New = num_IPCA_estimate_ALS(GB_Old, W_train, X_train, Nts_train)
        # compute max change for convergence
        tol = max(np.max(np.abs(GB_New - GB_Old)), np.max(np.abs(F_New - F_Old)))
        # update state
        GB_Old[:] = GB_New
        F_Old[:]  = F_New
        iter_count += 1

    # Assign GB-model parameters
    FactorGB    = F_New.copy()               # [K x t]
    GammaBetaGB = GB_New.copy()              # [L x K]
    LambdaGB    = np.mean(FactorGB, axis=1)  # [K,]
    print(f"[INFO] GB ALS at t={t}: {iter_count} iter, {time.time()-tic:.2f}s")

    # --- Out-of-sample predictive fits at t+1 ---
    # Predict asset returns using mean factor LambdaGB
    OOSRFITS_pred_GB[:, t] = Z[:, :, t] @ (GammaBetaGB @ LambdaGB)  # [N,]
    # Predict characteristics portfolio using mean factor
    OOSXFITS_pred_GB[:, t] = bigW[:, :, t] @ (GammaBetaGB @ LambdaGB)  # [L,]

    # --- Realized OOS factor via X and W at t+1 ---
    X_next = bigX[:, t]                    # [L,]
    W_next = bigW[:, :, t]                 # [L x L]
    # solve (Γ'WΓ) β = Γ' X for β
    lhs = GammaBetaGB.T @ W_next @ GammaBetaGB  # [K x K]
    rhs = GammaBetaGB.T @ X_next                # [K,]
    OOSRealFact[:, t] = np.linalg.solve(lhs, rhs)  # [K,]

    # --- Out-of-sample tangency portfolio return at t+1 ---
    # Tangency on forecasted factor series (use tanptfnext)
    OOSRealTan[t], _ = tanptfnext(FactorGB.T, OOSRealFact[:, t][np.newaxis, :])

    # --- Fitted asset returns using realized factor (for comparison) ---
    OOSRFITS_GB[:, t] = Z[:, :, t] @ (GammaBetaGB @ OOSRealFact[:, t])  # [N,]
    OOSXFITS_GB[:, t] = bigW[:, :, t] @ (GammaBetaGB @ OOSRealFact[:, t])  # [L,]

    # Diagnostic print every 10 steps
    if (t - startindex) % 10 == 0:
        print(f"[DEBUG] t={t}: mean OOSRFITS_pred_GB={np.nanmean(OOSRFITS_pred_GB[:,t]):.4f}, "
              f"mean OOSRealFact={np.nanmean(OOSRealFact[:,t]):.4f}, "
              f"OOSRealTan={OOSRealTan[t]:.4f}")

# %% 7. Restore full-sample variables (optional)
X   = bigX.copy()        # [L x T]
W   = bigW.copy()        # [L x L x T]
Nts = bigNts.copy()      # [T,]

print("[INFO] Out-of-sample estimation completed")

# %% 8. Compute Out-of-Sample R² and Sharpe Ratio
# Define out-of-sample indices (t = startindex+1 to T-1)
oos_idx = np.arange(startindex, T)  # [indices]

# True vs predicted returns
y_true = xret[:, oos_idx]                 # [N x noOOS]
y_pred = OOSRFITS_pred_GB[:, oos_idx]     # [N x noOOS]
mask = ~np.isnan(y_true) & ~np.isnan(y_pred)  # boolean mask
# OOS R² for returns
num = mySOS((y_true - y_pred)[mask])         # SSE_pred
den = mySOS((y_true - np.nanmean(y_true))[mask])  # SST
oos_r2_asset = 1 - num/den                   # scalar

# True vs predicted characteristics portfolio
X_true = X[:, oos_idx]                      # [L x noOOS]
X_pred = OOSXFITS_pred_GB[:, oos_idx]       # [L x noOOS]
maskX = ~np.isnan(X_true) & ~np.isnan(X_pred)
numX = mySOS((X_true - X_pred)[maskX])
denX = mySOS((X_true - np.nanmean(X_true))[maskX])
oos_r2_char = 1 - numX/denX                  # scalar

# OOS tangency Sharpe ratio
tp = OOSRealTan[oos_idx]                     # [noOOS,]
oos_sr = np.nanmean(tp) / np.nanstd(tp)      # scalar

# %% 9. Print OOS performance summary
print("\n======================================")
print("  Out-of-Sample Performance Summary")
print("======================================")
print(f" OOS Predictive R² (Returns):      {oos_r2_asset:.4f}")
print(f" OOS Predictive R² (Characteristics): {oos_r2_char:.4f}")
print(f" OOS Tangency Portfolio Sharpe:   {oos_sr:.4f}")
print(f"  (Sample: t = {startindex+1} to {T})")
print("======================================")

# %% 10. Save results to NPZ
out_dir = '../IPCA_KELLY'                    
os.makedirs(out_dir, exist_ok=True)       # ensure directory exists
out_file = os.path.join(out_dir, f"Results_GB_outofsample_{dataname}_K{K}.npz")
np.savez(
    out_file,
    xret=xret, W=W, X=X, date=date, LOC=LOC, Nts=Nts,
    OOSRFITS_pred_GB=OOSRFITS_pred_GB,
    OOSXFITS_pred_GB=OOSXFITS_pred_GB,
    OOSRealFact=OOSRealFact,
    OOSRealTan=OOSRealTan,
    OOSRFITS_GB=OOSRFITS_GB,
    OOSXFITS_GB=OOSXFITS_GB
)
print(f"[INFO] Saved out-of-sample results to '{out_file}'")
