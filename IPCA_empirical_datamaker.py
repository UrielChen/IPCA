# -----------------------------------------------------------------------------
# IPCA_empirical_datamaker.py
# -----------------------------------------------------------------------------
# Purpose:
#   Prepare characteristic-based data for IPCA estimation: construct Z (exposures),
#   X (managed returns), W (covariance weights), and handle transformations
#   (e.g., ranks, means, deviations, subsampling).
#
# Assumptions:
#   - Loads preprocessed characteristic data from .npz:
#       xret (N,T): asset returns,
#       chars (N,T,L): characteristics,
#       charnames: list of characteristic names,
#       date (T,): time vector
#   - Data are lagged, ready for transformation.
#   - Minimum cross-section size filters applied.
#   - No look-ahead bias; no NaNs in final outputs.
#
# Inputs:
#   - Output of IPCA_empirical_datacall_FNW36 saved as .npz.
#
# Outputs:
#   - Z (N,L,T): processed characteristics (exposures)
#   - X (L,T): managed portfolio returns
#   - W (L,L,T): characteristic covariance matrices
#   - xret (N,T), LOC (N,T), Nts (T,), date (T,)
#   - Saves all above to .npz file for IPCA.
# -----------------------------------------------------------------------------

import numpy as np                     # For numerical array ops
import pandas as pd                    # For convenient DataFrame handling
from scipy.stats import rankdata       # For tiedrank equivalent
import os                              # For file operations

# -------------------------- 1. USER SETTINGS ---------------------------------

# Set base filename for output; suffixes will be added
mat_file_name = 'IPCADATA_FNW36'  # (str) base for output

# Set datacall script output location (already saved as .npz)
datacall_npz = '../IPCA_KELLY/IPCA_empirical_datacall_FNW36_out.npz'

# Transformation flags (mutually exclusive)
option_rnkdmn_beg     = 1   # 1=Rank-normalize chars first? Adds _RNKDMN
option_zscore_beg     = 0   # 1=Z-score chars first? Adds _ZSCORE

# Mean type (only one of tsmean or histmean)
option_tsmean         = 0   # 1=Time-series mean? Adds _TS
option_histmean       = 0   # 1=Historical (cumulative) mean? Adds _HIST

# Dataset type (only one: mean, dev, meandev)
option_mean_dataset   = 0   # 1=Only means? Adds _MEAN
option_dev_dataset    = 0   # 1=Only deviations? Adds _DEV
option_meandev_dataset= 0   # 1=Means + deviations? Adds _MEANDEV

# Additional options
option_constant       = 1   # 1=Add constant char? Adds _CON, L +=1
option_orthogonalize  = 0   # 1=Orthogonalize chars? Adds _ORTHO, sets W=eye(L)
option_keepthese      = []  # list of charnames to keep (empty = all)
option_keepbig        = 0   # 1=Keep top 1000 by size? Adds _KEEPBIG
option_keepsmall      = 0   # 1=Keep bottom 1000? Adds _KEEPSMALL
option_keepbighalf    = 0   # 1=Keep above median size? Adds _KEEPBIGHALF
option_keepsmallhalf  = 0   # 1=Keep below median? Adds _KEEPSMALLHALF
option_subsample      = []  # e.g., [yyyymm_start, yyyymm_end] or empty
option_keepfirsthalf  = 0   # 1=Keep first half of obs? Adds _KEEPFIRSTHALF
option_keepsecondhalf = 0   # 1=Keep second half? Adds _KEEPSECONDHALF
option_keeprandomhalf = 0   # 1=half, 2=complement; loads randomhalf_*.npz
randomhalf_file       = None # e.g. "../IPCA_KELLY/randomhalf_IPCADATA_FNW36.npz"

# -------------------- 2. LOAD RAW DATA FROM DATACALL -------------------------
print(f"IPCA_empirical_datamaker started.")
# Load output from datacall; expects keys: chars, xret, charnames, date
data = np.load(datacall_npz, allow_pickle=True)
chars = data['chars']           # [N,T,L]: characteristics
xret  = data['xret']            # [N,T]: returns
charnames = data['charnames']   # [L,] list of str (or object array)
date  = data['date']            # [T,] time points (int or float)
print("  Data read in.")

# Get shape information
N, T, L = chars.shape  # N: firms, T: months, L: characteristics
print(f"  chars shape: {chars.shape} (N,T,L)")
print(f"  xret shape:  {xret.shape} (N,T)")

# Compute LOC: mask for non-missing chars AND returns ([N,T])
LOC = (~np.isnan(chars)).all(axis=2) & ~np.isnan(xret)  # [N,T] boolean mask
print("  LOC mask created.")

# Ensure sufficient cross-section per t
minN = max(100, (option_meandev_dataset+1)*L + option_constant + 1)  # int
keepthese = LOC.sum(axis=0) >= minN                                  # [T,] bool
LOC = LOC[:, keepthese]                                              # [N, T']
chars = chars[:, keepthese, :]                                       # [N, T', L]
xret  = xret[:, keepthese]                                           # [N, T']
date  = date[keepthese]                                              # [T',]
N, T, L = chars.shape                                                # updated dims
print(f"  After filtering, N={N}, T={T}, L={L}")

# -------------------- 3. SUBSET CHARACTERISTICS IF SPECIFIED -----------------
if option_keepthese:
    # Find indices for specified charnames to keep
    keepidx = [i for i, name in enumerate(charnames) if name in option_keepthese]
    chars = chars[:,:,keepidx]           # [N,T,L']
    charnames = charnames[keepidx]       # [L',]
    N, T, L = chars.shape                # Update shape
    print(f"  Only keeping characteristics: {option_keepthese}")
    print(f"  Updated chars shape: {chars.shape} (N,T,L)")

# --------------------- 4. APPLY INITIAL TRANSFORMATIONS ----------------------
# Rank-normalize if flagged
if option_rnkdmn_beg:
    # Apply rank normalization to each [N,T] char independently
    chars_rnk = np.empty_like(chars)  # [N,T,L]
    for l in range(L):
        # For each characteristic l, rank over non-NaNs for each time t
        for t in range(T):
            col = chars[:,t,l]
            mask = ~np.isnan(col)
            # Only rank over non-missing
            if mask.sum() > 0:
                col_ranked = np.full_like(col, np.nan)
                ranks = rankdata(col[mask], method='average')  # Ranks: [1..sum(mask)]
                col_ranked[mask] = (ranks - 1) / (mask.sum() - 1) - 0.5
                chars_rnk[:,t,l] = col_ranked
            else:
                chars_rnk[:,t,l] = np.nan
    chars = chars_rnk
    print(f"  Rank normalization applied for all chars. chars.shape={chars.shape}")
    mat_file_name += '_RNKDMN'

if option_zscore_beg:
    # Apply z-score normalization to each [N,T] char independently
    chars_zs = np.empty_like(chars)  # [N,T,L]
    for l in range(L):
        for t in range(T):
            col = chars[:,t,l]
            mask = ~np.isnan(col)
            if mask.sum() > 0:
                mu = np.nanmean(col)
                std = np.nanstd(col)
                chars_zs[:,t,l] = (col - mu) / std if std > 0 else 0
            else:
                chars_zs[:,t,l] = np.nan
    chars = chars_zs
    print(f"  Z-score normalization applied for all chars. chars.shape={chars.shape}")
    mat_file_name += '_ZSCORE'

# ---------------------- 5. COMPUTE MEANS (TS OR HIST) ------------------------
if option_tsmean and option_histmean:
    raise ValueError('option_tsmean and option_histmean are both turned on')

if option_tsmean:
    # Time-series mean: mean over time for each [N,L]
    # Output tmp: [N,1,L] broadcasted to [N,T,L]
    tmp = np.nanmean(chars, axis=1, keepdims=True).repeat(T, axis=1)
    # Set missing entries to nan
    tmp[np.isnan(chars)] = np.nan
    print("  Time-series mean (TS) constructed.")
    meansuffix = '_TS'

if option_histmean:
    # Historical (cumulative) mean up to t for each [N,L]
    tmp = np.zeros_like(chars)            # [N,T,L]
    tmploc = np.broadcast_to(LOC[:,:,None], chars.shape)  # [N,T,L]
    tmp[tmploc] = chars[tmploc]
    # Compute cumulative mean for each firm/char
    tmp2 = np.full_like(chars, np.nan)    # [N,T,L]
    for n in range(N):
        for l in range(L):
            good = LOC[n,:]
            if np.any(good):
                cs = np.cumsum(tmp[n,good,l])
                denom = np.arange(1, good.sum()+1)
                tmp2[n,good,l] = cs / denom
    tmp = tmp2
    print("  Historical mean (HIST) constructed.")
    meansuffix = '_HIST'

# ------------- 6. CREATE DATASET TYPE (MEAN, DEV, MEANDEV) -------------------
if option_mean_dataset + option_dev_dataset + option_meandev_dataset > 1:
    raise ValueError('more than one of mean/dev/meandev are turned on')

if option_mean_dataset:
    chars = tmp
    print("  Kept only means.")
    mat_file_name += meansuffix + '_MEAN'

if option_dev_dataset:
    chars = chars - tmp
    print("  Kept only deviations (chars - mean).")
    mat_file_name += meansuffix + '_DEV'

if option_meandev_dataset:
    chars_new = np.full((N, T, 2*L), np.nan)
    chars_new[:,:,:L] = tmp
    chars_new[:,:,L:] = chars - tmp
    chars = chars_new
    L = 2*L
    print("  Kept means + deviations (MEANDEV).")
    mat_file_name += meansuffix + '_MEANDEV'

# ---------------- 7. ADD CONSTANT CHARACTERISTIC IF FLAGGED ------------------
if option_constant:
    chars = np.concatenate([chars, np.ones((N,T,1))], axis=2)  # [N,T,L+1]
    L = chars.shape[2]   # Update number of chars
    print("  Added constant characteristic to chars. chars.shape={}".format(chars.shape))
    mat_file_name += '_CON'

# ---------------- 8. Size-Based Filters (KEEPBIG, KEEPSMALL, KEEP(BIG/SMALL)HALF) ------------------
def _filter_by_size(chars, LOC, xret, date, L, N, T, keepbig, keepsmall, keepbighalf, keepsmallhalf):
    # All assume size is 8th characteristic
    chars_sz = chars.copy()
    chars_sz = np.transpose(chars_sz, (0,2,1))  # (N, L, T) for filtering
    # Initialize masks
    big1000 = np.copy(LOC)
    small1000 = np.copy(LOC)
    bighalf = np.copy(LOC)
    smallhalf = np.copy(LOC)
    for t in range(T):
        szvals = chars_sz[LOC[:,t], 7, t]   # characteristic 8, 0-based
        if szvals.size == 0:
            continue
        # -- Top/Bottom 1000 --
        if keepbig or keepsmall:
            idx_sort = np.argsort(szvals)
            # bottom 1000 mask
            thresh_small = szvals[idx_sort[min(999, len(szvals)-1)]]
            small_mask = chars_sz[:, 7, t] <= thresh_small
            # top 1000 mask
            thresh_big = szvals[idx_sort[max(-1000, -len(szvals))]]
            big_mask = chars_sz[:, 7, t] >= thresh_big
            small1000[:, t] &= small_mask
            big1000[:, t] &= big_mask
        # -- Half above/below median --
        if keepbighalf or keepsmallhalf:
            med = np.nanmedian(chars_sz[:, 7, t])
            smallhalf[:, t] &= chars_sz[:, 7, t] < med
            bighalf[:, t] &= chars_sz[:, 7, t] >= med
    chars_sz = np.transpose(chars_sz, (0,2,1))  # back to (N, T, L)
    # Apply selection
    if keepbig:
        print("You keep the big stocks.")
        return chars, big1000, xret, date
    if keepsmall:
        print("You keep the small stocks.")
        keepthese = np.sum(small1000, axis=0) >= max(100, L + 1)
        chars = chars[:, keepthese, :]
        LOC = small1000[:, keepthese]
        xret = xret[:, keepthese]
        date = date[keepthese]
        return chars, LOC, xret, date
    if keepbighalf:
        print("You keep the big half.")
        return chars, bighalf, xret, date
    if keepsmallhalf:
        print("You keep the small half.")
        keepthese = np.sum(smallhalf, axis=0) >= max(100, L + 1)
        chars = chars[:, keepthese, :]
        LOC = smallhalf[:, keepthese]
        xret = xret[:, keepthese]
        date = date[keepthese]
        return chars, LOC, xret, date
    return chars, LOC, xret, date

chars, LOC, xret, date = _filter_by_size(
    chars, LOC, xret, date, L, N, T,
    option_keepbig, option_keepsmall, option_keepbighalf, option_keepsmallhalf
)
N, T, L = chars.shape

# ---------------- 9. APPLY TIME SUBSAMPLE IF SPECIFIED -----------------------
if option_subsample:
    t1 = np.searchsorted(date, option_subsample[0], side='left')
    t2 = np.searchsorted(date, option_subsample[1], side='right')
    LOC = LOC[:, t1:t2]
    chars = chars[:, t1:t2, :]
    xret = xret[:, t1:t2]
    date = date[t1:t2]
    N, T, L = chars.shape
    mat_file_name += '_{}-{}'.format(option_subsample[0], option_subsample[1])

# --------------- 10. KEEP FIRST/SECOND HALF OF SAMPLE (SKIPPED) --------------
if option_keepfirsthalf or option_keepsecondhalf:
    half = T // 2
    if option_keepfirsthalf:
        chars = chars[:, :half, :]
        xret  = xret[:, :half]
        date  = date[:half]
        print("You keep the first half.")
        mat_file_name += '_KEEPFIRSTHALF'
    else:
        chars = chars[:, half:, :]
        xret  = xret[:, half:]
        date  = date[half:]
        print("You keep the second half.")
        mat_file_name += '_KEEPSECONDHALF'
    T = chars.shape[1]

# --------------- 11. KEEP RANDOM HALF IF FLAGGED (SKIPPED) -------------------
if option_keeprandomhalf:
    if not randomhalf_file:
        raise ValueError("randomhalf_file path required for random half operation.")
    rh = np.load(randomhalf_file)
    # Both arrays must be bool with shape (N, T)
    randomhalf = rh['randomhalf']
    randomhalfcomp = rh['randomhalfcomp']
    if option_keeprandomhalf == 1:
        LOC[randomhalf, :] = False
    elif option_keeprandomhalf == 2:
        LOC[randomhalfcomp, :] = False
    else:
        raise ValueError("Invalid value for option_keeprandomhalf")
    mat_file_name += f'_RANDOMHALF{option_keeprandomhalf}'

# ------------ 12. ORTHOGONALIZE OR CONSTRUCT X/W (NO ORTHO) ------------------
Nts = LOC.sum(axis=0)  # [T,] cross-sectional count per time
print(f"  Cross-sectional counts Nts: {Nts}")

if option_orthogonalize:
    # Orthogonalization code (not shown, as option_orthogonalize=0)
    pass
else:
    # chars: [N,T,L], permute to [N,L,T]
    Z = np.transpose(chars, (0,2,1))  # [N,L,T] characteristic exposures
    W = np.full((L, L, T), np.nan)    # [L,L,T] covariance weights
    X = np.full((L, T), np.nan)       # [L,T] managed returns
    Q = np.full((L, T), np.nan)       # [L,T] (used for completeness)
    for t in range(T):
        idx = LOC[:,t]                # [N,] bool
        if idx.sum() == 0:
            continue                  # skip if no valid obs
        Zt = Z[idx,:,t]               # [Nts(t),L]
        xt = xret[idx,t]              # [Nts(t),]
        W[:,:,t] = (Zt.T @ Zt) / Nts[t]     # [L,L]
        X[:,t]   = (Zt.T @ xt) / Nts[t]     # [L,]
        try:
            Q[:,t] = np.linalg.solve(W[:,:,t], X[:,t]) # [L,]
        except np.linalg.LinAlgError:
            Q[:,t] = np.nan
    print("  Constructed Z, X, W, Q for IPCA.")

# ------------- 13. SAVE PROCESSED DATA (TO .npz INSTEAD OF .mat) -------------
savefile = f'../IPCA_KELLY/{mat_file_name}.npz'
np.savez(
    savefile,
    Z=Z,        # [N,L,T] processed exposures
    X=X,        # [L,T] managed portfolio returns
    W=W,        # [L,L,T] char covariance matrices
    Q=Q,        # [L,T] managed char-sorted returns
    xret=xret,  # [N,T] returns
    LOC=LOC,    # [N,T] valid obs
    Nts=Nts,    # [T,] cross-sectional counts
    dates=date   # [T,] dates
)
print(f"  Saved processed IPCA data to {savefile}")
print("IPCA_empirical_datamaker done.")
