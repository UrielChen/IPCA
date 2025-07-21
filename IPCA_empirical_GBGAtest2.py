# -----------------------------------------------------------------------------
# Script: IPCA_empirical_GBGAtest.py
#
# Main purpose:
#   - Implements a bootstrap-based Wald-like test of the IPCA alpha intercept.
#   - Estimates the unrestricted IPCA model (with alpha) for a specified K.
#   - Performs wild-t block bootstrap for robust inference on the alpha vector.
#
# Key assumptions:
#   1. Core data arrays X, W, Z, xret, LOC, Nts, Q, date, N, L, T are stored
#      in NumPy .npz files under the "IPCA_KELLY" directory.
#   2. User specifies K, bootsims, and dataname in the script.
#   3. Bootstrap uses block wild-t resampling to preserve time dependence
#      and heteroskedasticity.
#
# Inputs:
#   - IPCA_KELLY/<dataname>.npz containing:
#       X    : (L, T) moment matrix
#       W    : (L, L, T) weight tensor
#       Z    : (N, L, T) instrument tensor
#       xret : (N, T) excess returns
#       LOC  : (N, T) availability mask
#       Nts  : (T,) count of valid observations per period
#       Q    : (L, T) managed portfolio returns
#       date : (T,) date vector
#       N, L, T : integers
#
# Outputs:
#   - Results_GBGAtest_<dataname>_K<K>_boot<bootsims><clustersuffix>.npz in
#     IPCA_KELLY/ containing estimated parameters, bootstrap alphas, and timing.
#   - Prints ALS diagnostics, fit/R² summaries, and bootstrap p-value.
#
# Requirements:
#   Python 3.x, numpy, scipy, num_IPCA_estimate_ALS, mySOS
# -----------------------------------------------------------------------------

# %% 0. Imports and environment setup
import os                                                      # file path operations
import time                                                    # timing utilities
import numpy as np                                             # numerical arrays
from scipy.sparse.linalg import svds                            # for truncated SVD
from num_IPCA_estimate_ALS import num_IPCA_estimate_ALS         # ALS estimator
from mySOS import mySOS                                         # sum-of-squares helper
from tqdm import tqdm

# %% 1. User-specified parameters
# Number of latent factors to estimate
K = 3                          # K: scalar
# Number of bootstrap simulations
bootsims = 500                 # bootsims: scalar (≥500 for basic inference)
# Base name of the input data file (without extension)
dataname = 'IPCADATA_FNW36_RNKDMN_CON'  # dataname: string

# Optional cluster suffix for file naming
clustersuffix = ''             # clustersuffix: string

# %% 2. Load data from NPZ
# Construct full path to input .npz file
data_path = os.path.join('../IPCA_KELLY', dataname + '.npz')  
# Load data
data = np.load(data_path, allow_pickle=True)  

# Extract arrays with shape annotations
# X: (L, T)
X = data['X']                  # X: moment matrix (L, T)
# W: (L, L, T)
W = data['W']                  # W: weight tensor (L, L, T)
# Z: (N, L, T)
Z = data['Z']                  # Z: instrument tensor (N, L, T)
# xret: (N, T)
xret = data['xret']            # xret: excess returns (N, T)
# LOC: (N, T)
LOC = data['LOC']              # LOC: availability mask (N, T)
# Nts: (T,)
Nts = data['Nts']              # Nts: valid obs count per period (T,)
# Q: (L, T)
Q = data['Q']                  # Q: managed portfolio returns (L, T)
# date: (T,)
date = data['dates']            # date: date vector (T,)

# Extract dimensions from loaded data
N, T = xret.shape              # N: number of observations, T: number of time periods
L, _ = X.shape                 # L: number of characteristics

# %% 3. Set ALS options
als_opt = {
    'MaxIterations': 5000,      # maximum ALS iterations
    'Tolerance': 1e-6           # convergence tolerance
}

# Print loaded data summary
print(f"[INFO] Loaded data from {data_path}")
print(f"[INFO] Shapes -- X:{X.shape}, W:{W.shape}, Z:{Z.shape}, xret:{xret.shape}, Q:{Q.shape}")

# %% 4. SVD initialization for ALS
print(f"[INFO] IPCA_empirical_GBGAtest starting at {time.ctime()}: K={K}, data={dataname}")
# Compute truncated SVD of X: X ≈ U S V'
U, s_vals, Vt = svds(X, k=K)    # U:(L,K), s_vals:(K,), Vt:(K,T)
# Form initial GammaBeta loadings (L, K)
GammaBeta_XSVD = U.copy()       # initial loadings
# Form initial factor series F_old = S * V' (K, T)
F_old = np.diag(s_vals) @ Vt    # initial factors

# %% 5. ALS for restricted model (no alpha)
print(f"[INFO] Restricted ALS started at {time.ctime()}")
start_time = time.time()        # start timer

# Initialize ALS state for restricted model
GB_old = GammaBeta_XSVD.copy()  # GB_old: (L, K)
tol = 1.0                       # tolerance tracker
iter_count = 0                  # iteration counter
tols = np.full((500,), np.nan)  # tol trace (500,)

# ALS loop (restricted)
while iter_count <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
    # One ALS update step
    GB_new, F_new = num_IPCA_estimate_ALS(GB_old, W, X, Nts)  # shapes: (L,K),(K,T)
    # Compute maximum change across GB and F
    tol_GB = np.max(np.abs(GB_new - GB_old))  # scalar
    tol_F  = np.max(np.abs(F_new - F_old))    # scalar
    tol = max(tol_GB, tol_F)                  # update tol
    # Update state
    GB_old[:] = GB_new
    F_old[:]  = F_new
    # Record tol history
    tols = np.roll(tols, -1)   # shift left
    tols[-1] = tol             # append current tol
    iter_count += 1            # increment iteration

# Assign restricted model results
GB_GB = GB_new.copy()                         # restricted loadings (L, K)
GB_F  = F_new.copy()                          # restricted factors (K, T)
GB_L  = np.mean(GB_F, axis=1)                 # factor means (K,)

# Record restricted ALS timing and diagnostics
timing_restricted = {
    'time': time.time() - start_time,         # elapsed seconds
    'iterations': iter_count,                 # iterations count
    'tols': tols.copy()                       # tol trace
}
print(f"[INFO] Restricted ALS done: {iter_count} iters, {timing_restricted['time']:.2f}s at {time.ctime()}")

# %% 6. ALS for unrestricted model (with alpha intercept)
print(f"[INFO] Unrestricted ALS (with alpha) started at {time.ctime()}")
start_time = time.time()

# Initialize with restricted solution + zero column for alpha
GB_old = np.hstack([GB_GB, np.zeros((L, 1))])  # GB_old: (L, K+1)
F_old = GB_F.copy()                            # F_old: (K, T)

tol = 1.0                                      # reset tol
iter_count = 0                                 # reset iteration counter
tols = np.full((500,), np.nan)                 # reset tol trace

# Pre-specified alpha factor of ones
alpha_factor = np.ones((1, T))  # shape (1, T)

# ALS loop (unrestricted)
while iter_count <= als_opt['MaxIterations'] and tol > als_opt['Tolerance']:
    # One ALS update with alpha_factor
    GB_new, F_new = num_IPCA_estimate_ALS(GB_old, W, X, Nts, alpha_factor)
    # Compute maximum change
    tol_GB = np.max(np.abs(GB_new - GB_old))
    tol_F  = np.max(np.abs(F_new - F_old))
    tol = max(tol_GB, tol_F)
    # Update state
    GB_old[:] = GB_new
    F_old[:]  = F_new
    # Record tol
    tols = np.roll(tols, -1)
    tols[-1] = tol
    iter_count += 1

# Assign unrestricted model results
GBGA_GB = GB_new[:, :-1].copy()  # unrestricted loadings (L, K)
GBGA_GA = GB_new[:, -1].copy()   # estimated alpha intercept (L,)
GBGA_F  = F_new.copy()           # unrestricted factors (K, T)
GBGA_L  = np.mean(GBGA_F, axis=1) # mean factors (K,)

# Record unrestricted ALS timing and diagnostics
timing_unrestricted = {
    'time': time.time() - start_time,
    'iterations': iter_count,
    'tols': tols.copy()
}
print(f"[INFO] Unrestricted ALS done: {iter_count} iters, {timing_unrestricted['time']:.2f}s at {time.ctime()}")

# %% 7. Compute fits and R² statistics
# Preallocate fit matrices with NaNs
RFITS_GB        = np.full((N, T), np.nan)  # restricted asset fits (N, T)
RFITS_pred_GB   = np.full((N, T), np.nan)  # restricted predicted fits (N, T)
RFITS_GBGA      = np.full((N, T), np.nan)  # unrestricted asset fits (N, T)
RFITS_pred_GBGA = np.full((N, T), np.nan)  # unrestricted predicted fits (N, T)
XFITS_GB        = np.full((L, T), np.nan)  # restricted managed portfolio fits (L, T)
XFITS_pred_GB   = np.full((L, T), np.nan)
XFITS_GBGA      = np.full((L, T), np.nan)  # unrestricted managed portfolio fits (L, T)
XFITS_pred_GBGA = np.full((L, T), np.nan)

# Loop over each time period
for t in range(T):
    Z_t = Z[:, :, t]                       # Z_t: (N, L)
    W_t = W[:, :, t]                       # W_t: (L, L)
    # Restricted model fits
    RFITS_GB[:, t]      = Z_t @ (GB_GB @ GB_F[:, t])
    RFITS_pred_GB[:, t] = Z_t @ (GB_GB @ GB_L)
    XFITS_GB[:, t]      = W_t @ (GB_GB @ GB_F[:, t])
    XFITS_pred_GB[:, t] = W_t @ (GB_GB @ GB_L)
    # Unrestricted model fits
    theta_t = GBGA_GA + GBGA_GB @ GBGA_F[:, t]  # (L,)
    theta_L = GBGA_GA + GBGA_GB @ GBGA_L        # (L,)
    RFITS_GBGA[:, t]      = Z_t @ theta_t
    RFITS_pred_GBGA[:, t] = Z_t @ theta_L
    XFITS_GBGA[:, t]      = W_t @ theta_t
    XFITS_pred_GBGA[:, t] = W_t @ theta_L

# Compute sum-of-squares total for returns
xret_masked = np.where(LOC, xret, np.nan)    # mask missing returns (N, T)
tot_sos = mySOS(xret_masked)                  # scalar

# Restricted R²
RR2_total_GB  = 1 - mySOS(xret_masked - RFITS_GB)      / tot_sos
RR2_pred_GB   = 1 - mySOS(xret_masked - RFITS_pred_GB) / tot_sos

# Unrestricted R²
RR2_total_GBGA  = 1 - mySOS(xret_masked - RFITS_GBGA)      / tot_sos
RR2_pred_GBGA   = 1 - mySOS(xret_masked - RFITS_pred_GBGA) / tot_sos

# Cross-sectional R² for X on managed fits
XR2_total_GB    = 1 - mySOS(X - XFITS_GB)      / mySOS(X)
XR2_pred_GB     = 1 - mySOS(X - XFITS_pred_GB) / mySOS(X)
XR2_total_GBGA  = 1 - mySOS(X - XFITS_GBGA)    / mySOS(X)
XR2_pred_GBGA   = 1 - mySOS(X - XFITS_pred_GBGA)/ mySOS(X)

print("[INFO] Completed fit and R² calculations")

# %% 8. Bootstrap for alpha Wald-like test
if bootsims > 0:
    print(f"[INFO] Starting bootstrap with {bootsims} simulations")
    boot_start = time.time()
    # Residuals from unrestricted fit: (L, T)
    RESID = X - XFITS_GBGA
    # Container for bootstrapped alphas: (bootsims, L)
    boot_GA = np.full((bootsims, L), np.nan)

    # Degrees of freedom for wild-t
    dof = 5
    tvar = dof / (dof - 2)  # variance adjustment

    # Perform bootstrap reps
    for b in tqdm(range(bootsims)):
        # Construct block bootstrap index array of length T
        btix = []
        tmp = 0
        block_size = 1
        while tmp < T:
            # Draw random starting point 0..T-1
            idx = np.random.randint(0, T)
            btix.append(idx)
            tmp += 1
        btix = np.array(btix[:T])  # ensure length T

        # Wild-t weights: (T,)
        t_samples = np.random.standard_t(dof, size=T)
        weights = t_samples / np.sqrt(tvar)

        # Bootstrapped X_b: (L, T)
        X_b = XFITS_GB + RESID[:, btix] * weights  # broadcast multiply

        # Re-estimate unrestricted model on X_b
        GB_b = np.hstack([GB_GB, np.zeros((L, 1))])  # (L, K+1)
        F_b = GB_F.copy()                            # (K, T)
        tol_b = 1.0
        iter_b = 0
        # ALS loop for bootstrap
        while iter_b <= als_opt['MaxIterations'] and tol_b > als_opt['Tolerance']:
            GB_new_b, F_new_b = num_IPCA_estimate_ALS(GB_b, W, X_b, Nts, alpha_factor)
            tol_GB_b = np.max(np.abs(GB_new_b - GB_b))
            tol_F_b  = np.max(np.abs(F_new_b - F_b))
            tol_b = max(tol_GB_b, tol_F_b)
            GB_b = GB_new_b
            F_b  = F_new_b
            iter_b += 1
        # todo: skip np.linalg.LinalgError: Singular matrix
        # Store bootstrapped alpha (last column)
        boot_GA[b, :] = GB_b[:, -1]

    # Record bootstrap timing
    timing_boot = {
        'simulations': bootsims,
        'time': time.time() - boot_start
    }
    print(f"[INFO] Bootstrap completed in {timing_boot['time']:.2f}s")

    # Print first 10 bootstrap alpha squared norms
    sq_norms = np.sum(boot_GA**2, axis=1)
    print("[INFO] First 10 bootstrap alpha squared norms:", sq_norms[:10])

    # Compute empirical p-value for Wald test
    observed_stat = mySOS(GBGA_GA)                 # observed squared norm
    p_val = np.mean(sq_norms > observed_stat)      # p-value
    print(f"[INFO] Bootstrap Wald p-value for alpha: {p_val:.4f}")

# %% 9. Save full results to NPZ
out_dir = '../IPCA_KELLY'
os.makedirs(out_dir, exist_ok=True)
suffix = f"_boot{bootsims}{clustersuffix}"
out_file = os.path.join(out_dir, f"Results_GBGAtest_{dataname}_K{K}{suffix}.npz")
np.savez(
    out_file,
    # Data
    X=X, W=W, Z=Z, xret=xret, LOC=LOC, Nts=Nts, Q=Q, date=date,
    # Restricted model
    GB_GB=GB_GB, GB_F=GB_F, GB_L=GB_L, timing_restricted=timing_restricted,
    # Unrestricted model
    GBGA_GB=GBGA_GB, GBGA_GA=GBGA_GA, GBGA_F=GBGA_F, GBGA_L=GBGA_L,
    timing_unrestricted=timing_unrestricted,
    # Fits
    RFITS_GB=RFITS_GB, RFITS_pred_GB=RFITS_pred_GB,
    RFITS_GBGA=RFITS_GBGA, RFITS_pred_GBGA=RFITS_pred_GBGA,
    XFITS_GB=XFITS_GB, XFITS_pred_GB=XFITS_pred_GB,
    XFITS_GBGA=XFITS_GBGA, XFITS_pred_GBGA=XFITS_pred_GBGA,
    # R²
    XR2_total_GB=XR2_total_GB, XR2_pred_GB=XR2_pred_GB,
    XR2_total_GBGA=XR2_total_GBGA, XR2_pred_GBGA=XR2_pred_GBGA,
    RR2_total_GB=RR2_total_GB, RR2_pred_GB=RR2_pred_GB,
    RR2_total_GBGA=RR2_total_GBGA, RR2_pred_GBGA=RR2_pred_GBGA,
    # Bootstrap
    bootsims=bootsims, boot_GA=boot_GA if bootsims>0 else None, timing_boot=locals().get('timing_boot', None)
)
print(f"[INFO] Results saved to '{out_file}'")