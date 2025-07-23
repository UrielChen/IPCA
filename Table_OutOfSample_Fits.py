# -----------------------------------------------------------------------------
# Script: Table_OutOfSample_Fits.py
#
# Main purpose:
#   - Summarize out-of-sample (OOS) performance metrics (R² and Sharpe ratios)
#     for IPCA (latent-only) and Fama–French (observable) models across
#     latent factor ranks K = 1…6.
#   - Results correspond to Table 5 and related tables in the paper.
#
# Key assumptions:
#   - OOS results are saved as NPZ files in 'IPCA_KELLY/' with names:
#       • Results_GB_outofsample_IPCADATA_FNW36_RNKDMN_CON_K{K}.npz
#       • Results_ObsFactRegROOS_..._K1_rec_60_60.npz (FF returns OOS)
#       • Results_ObsFactRegXOOS_..._K1_rec_60_60.npz (FF chars OOS)
#   - Each file contains the necessary arrays:
#       xret, X, LOC, OOSRFITS_pred_GB, OOSXFITS_pred_GB, OOSRFITS_GB, OOSXFITS_GB,
#       OOSRealFact, OOSRealTan for IPCA; FITS_FF{K}, FITS_cond_FF{K}, FITS_Factors
#       for FF results.
#   - Function utilities imported: mySOS, tanptf
#
# Inputs:
#   - NPZ result files in directory 'IPCA_KELLY'
#
# Outputs:
#   - Console printout of R² and Sharpe metrics for each K.
#   - 'report' matrix available in script for further use.
# -----------------------------------------------------------------------------

# %% 0. Imports
import os                                                         # path operations
import numpy as np                                                # array handling
from math import sqrt                                             # square root
from mySOS import mySOS                                           # sum-of-squares helper
from tanptf import tanptf                                         # tangency portfolio function

# %% 1. Input/output settings
dirname = '../IPCA_KELLY'                                             # directory containing NPZ files
# Base name for IPCA OOS results, K will be appended
name1  = 'Results_GB_outofsample_IPCADATA_FNW36_RNKDMN_CON_K'       # string
# Filenames for FF observable OOS results (consistent across K)
# nameFFR = 'Results_ObsFactRegROOS_Results_GB_IPCADATA_FNW36_RNKDMN_CON_K1_rec_60_60'  # returns
# nameFFX = 'Results_ObsFactRegXOOS_Results_GB_IPCADATA_FNW36_RNKDMN_CON_K1_rec_60_60'  # chars

Krange = range(1, 7)                                               # K = 1…6

# %% 2. Preallocate report matrix
si = 120                                                           # start index for OOS evaluation
report = np.full((30, 30), np.nan)                                 # [30×30], larger than needed

# %% 3. Loop over factor ranks K
for K in Krange:
    # Construct IPCA NPZ filename and load
    file_ipca = os.path.join(dirname, f"{name1}{K}.npz")           # IPCA OOS NPZ
    try:
        data_ipca = np.load(file_ipca, allow_pickle=True)          # load IPCA results
    except FileNotFoundError:
        continue                                                   # skip if missing

    # Load Fama–French observable OOS results (returns and chars)
    # data_ffr = np.load(os.path.join(dirname, nameFFR + '.npz'), allow_pickle=True)  # FF returns
    # data_ffx = np.load(os.path.join(dirname, nameFFX + '.npz'), allow_pickle=True)  # FF chars

    # Extract relevant arrays from IPCA results
    xret              = data_ipca['xret']                          # [N×T] asset returns
    X                 = data_ipca['X']                             # [L×T] char matrix
    LOC               = data_ipca['LOC']                           # [N×T] valid obs mask
    OOSRFITS_GB       = data_ipca['OOSRFITS_GB']                   # [N×T]
    OOSRFITS_pred_GB  = data_ipca['OOSRFITS_pred_GB']              # [N×T]
    OOSXFITS_GB       = data_ipca['OOSXFITS_GB']                   # [L×T]
    OOSXFITS_pred_GB  = data_ipca['OOSXFITS_pred_GB']              # [L×T]
    OOSRealFact       = data_ipca['OOSRealFact']                   # [K×T]
    OOSRealTan        = data_ipca['OOSRealTan']                    # [T,]

    # Extract FF arrays for returns and factors
    # FITS_FF        = data_ffr[f'FITS_FF{K}']                       # [N×T]
    # FITS_cond_FF   = data_ffr[f'FITS_cond_FF{K}']                  # [N×T]
    # FITS_Factors   = data_ffr['FITS_Factors']                      # [T×K]

    # --- Create mask of OOS evaluation dates/assets ---
    BIGLOC = LOC.astype(bool) # & FITS_FF.astype(bool)                            # [N×T] intersection mask
    BIGLOC[:, :si] = False                                         # mask out pre-OOS
    # Alternative mask override (as in MATLAB)
    BIGLOC = LOC.copy()                                            # reset to LOC
    BIGLOC[:, :si] = False                                         # mask pre-OOS again

    # Flatten mask for indexing
    mask_vec = BIGLOC.ravel()                                      # [N*T,]

    # --- Compute IPCA OOS R² metrics ---
    # Total R² on returns
    report[0, K-1] = 1 - mySOS(xret.ravel()[mask_vec] - OOSRFITS_GB.ravel()[mask_vec]) \
                         / mySOS(xret.ravel()[mask_vec])
    # Predicted R² on returns
    report[1, K-1] = 1 - mySOS(xret.ravel()[mask_vec] - OOSRFITS_pred_GB.ravel()[mask_vec]) \
                         / mySOS(xret.ravel()[mask_vec])
    # Total R² on characteristics
    report[2, K-1] = 1 - mySOS((X[:, si-1:] - OOSXFITS_GB[:, si-1:]).ravel()) \
                         / mySOS(X[:, si-1:].ravel())
    # Predicted R² on characteristics
    report[3, K-1] = 1 - mySOS((X[:, si-1:] - OOSXFITS_pred_GB[:, si-1:]).ravel()) \
                         / mySOS(X[:, si-1:].ravel())

    # --- Compute FF OOS R² metrics for returns ---
    # report[4, K-1] = 1 - mySOS(xret.ravel()[mask_vec] - FITS_FF.ravel()[mask_vec]) \
    #                      / mySOS(xret.ravel()[mask_vec])
    # report[5, K-1] = 1 - mySOS(xret.ravel()[mask_vec] - FITS_cond_FF.ravel()[mask_vec]) \
    #                      / mySOS(xret.ravel()[mask_vec])

    # --- Compute Sharpe ratios (annualized) ---
    # Univariate portfolio on last IPCA factor
    tmp_ipca = OOSRealFact[-1, si-1:]                                # [T_si,]
    report[4, K-1]  = sqrt(12) * np.nanmean(tmp_ipca) / np.nanstd(tmp_ipca)
    # Tangen2cy portfolio (IPCA)
    report[5, K-1]  = sqrt(12) * np.nanmean(OOSRealTan[si-1:]) / np.nanstd(OOSRealTan[si-1:])
    # Univariate portfolio on Kth FF factor
    # tmp_ff = FITS_Factors[si:, K-1]                                # [T_si,]
    # report[10, K-1] = sqrt(12) * np.nanmean(tmp_ff) / np.nanstd(tmp_ff)
    # # Tangency portfolio on all FF factors
    # tp_ff = tanptf(FITS_Factors[:, :K].T)                          # [T,]
    # report[11, K-1] = sqrt(12) * np.nanmean(tp_ff[si:]) / np.nanstd(tp_ff[si:])

# %% 4. Prepare row labels for printing
reportrows = [
    'R2_total_IPCA',     # 1
    'R2_pred_IPCA',      # 2
    'XR2_total_IPCA',    # 3
    'XR2_pred_IPCA',     # 4
    # 'R2_total_FF',       # 5
    # 'R2_pred_FF',        # 6
    # rows 7-8 for chars/FF omitted
    'Sharpe_univar_IPCA',# 9
    'Sharpe_tang_IPCA',  #10
    # 'Sharpe_univar_FF',  #11
    # 'Sharpe_tang_FF'     #12
]

# %% 5. Print summary table for LaTeX/paper
print("\nKs:  ", "   ".join(str(k) for k in Krange))
for idx, label in enumerate(reportrows):
    # Format values for each K
    vals = []
    for K in Krange:
        v = report[idx, K-1]
        # R² rows as percent, Sharpe rows as decimal
        if idx < 4:
            vals.append(f"{100*v:6.2f}%")
        else:
            vals.append(f"{v:6.3f}")
    # Print row
    print(f"{label.ljust(20)}:  " + "  ".join(vals))

# %% 6. Print formatted table
print("\n==============================================")
print(" Out-of-Sample R² and Sharpe Summary (K=1…6)")
print("==============================================")
# Header
header = "Metric".ljust(20) + "".join(f"K={k}".center(10) for k in Krange)
print(header)
print("-" * len(header))
# Rows
for idx, label in enumerate(reportrows):
    row = label.ljust(20)
    for K in Krange:
        v = report[idx, K-1]
        if idx < 4:
            row += f"{100*v:10.2f}%"
        else:
            row += f"{v:10.3f}"
    print(row)
print("==============================================")
