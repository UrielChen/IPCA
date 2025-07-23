# -----------------------------------------------------------------------------
# Script: Table_pure_alpha.py
#
# Main purpose:
#   - Summarize out-of-sample performance of arbitrage (alpha) portfolios
#     from extended IPCA models for latent ranks K = 1…6.
#   - Compute annualized Sharpe ratios (Table 7 in paper) using only the
#     second half of OOS observations to avoid startup noise.
#
# Assumptions:
#   - Out-of-sample NPZ files are in 'IPCA_KELLY/' named:
#       Results_GBGA_outofsample_IPCADATA_FNW36_RNKDMN_CON_K{K}.npz
#   - Each file contains:
#       OOSARBPTF: [T_oos,] array of OOS arbitrage portfolio returns.
#
# Inputs:
#   - NPZ result files for K = 1..6.
#
# Outputs:
#   - Console printouts of:
#       K value and annualized Sharpe ratio of OOS arbitrage portfolio.
# -----------------------------------------------------------------------------

# %% 0. Imports
import os                                                         # for path handling
import numpy as np                                                # numerical operations
from math import sqrt                                            # sqrt for annualization

# %% 1. Loop over latent factor ranks K = 1 to 6
for j in range(1, 7):  # j: latent rank K (scalar)
    # Construct filename for current K
    filename = f"Results_GBGA_outofsample_IPCADATA_FNW36_RNKDMN_CON_K{j}.npz"
    # Load .npz file from 'IPCA_KELLY' directory
    data = np.load(os.path.join('../IPCA_KELLY', filename))  # npz containing OOSARBPTF

    # Extract OOSARBPTF array
    OOSARBPTF = data['OOSARBPTF']  # [T_oos,] OOS arbitrage returns

    # Determine reporting sample: only use second half of observations
    n_obs = len(OOSARBPTF)                     # [scalar] total number of OOS returns
    start = n_obs // 2 - 1                        # integer index for second half start
    report_indices = np.arange(start, n_obs)   # [n_obs-start,] indices

    # Compute mean of OOS returns over reporting sample
    arbptf_mean = np.nanmean(OOSARBPTF[report_indices])  # [scalar] mean return
    # Compute standard deviation of OOS returns over reporting sample
    arbptf_std  = np.nanstd(OOSARBPTF[report_indices])   # [scalar] std deviation

    # Compute annualized Sharpe ratio (monthly data ×√12)
    arbptf_sr = arbptf_mean / arbptf_std * sqrt(12)      # [scalar] annualized Sharpe

    # Print key results for this K
    print(f"K = {j} | Ann. Sharpe (OOS Arb Ptf): {arbptf_sr:.4f}")  # display K and Sharpe

    # Optional diagnostics (commented out):
    # print(f"   [Diagnostics] Mean: {arbptf_mean:.4e}, Std: {arbptf_std:.4e}, Obs: {len(report_indices)}")
