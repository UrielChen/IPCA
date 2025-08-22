# Observable_Factor_Regressions.py
# -------------------------------------------------------------------------
# Main purpose:
#   Estimate time-series OLS betas and fitted returns for individual assets
#   or characteristic portfolios using observable factors (Fama-French, Stambaugh-Yuan,
#   Hou-Xue-Zhang, Barillas-Shanken). Compute Total R² (using realized factors) and
#   Predictive R² (using factor means).
#
# Assumptions:
#   - Data file (.npz) contains xret [N×T], Q [N×T], X [L×T], LOC [N×T], date [T×1].
#   - Factor data files (.npz) for FF, SY, HXZ, BS are pre-aligned by YYYYMM dates.
#   - LinearRegression from sklearn performs OLS (no intercept, consistent with MATLAB regress).
#   - tanptfnext() is a custom function computing tangency portfolio returns (placeholder).
#   - Out-of-sample (oosflag=True) uses recursive windows starting t=61; in-sample uses full sample.
#   - QXorR selects return matrix: 'R'=assets (xret), 'Q'=Q-portfolios, 'X'=managed portfolios.
#   - mySOS computes sum of squares for non-NaN elements.
#
# Outputs:
#   - FITS_* [N×T or L×T]: Realized fitted returns for each model.
#   - FITS_cond_* [N×T or L×T]: Predictive fitted returns using factor means.
#   - FITS_OOSTan_* [T×1]: Optional OOS tangency portfolio returns (computed but not saved).
#   - R² statistics (total and predictive) printed for each model.
#   - Saved .npz file: Results_ObsFactReg<Mode><OOS>_<dataname><suffix>.npz
# -------------------------------------------------------------------------

import numpy as np
from scipy import io
from sklearn.linear_model import LinearRegression
import datetime
import os
from mySOS import mySOS
from tanptfnext import tanptfnext

# 1. User settings
# Clear workspace (Python doesn't need this, but for clarity)
# No equivalent in Python; variables are scoped to script

# Base IPCA results file name (string scalar)
dataname = 'Results_GB_IPCADATA_FNW36_RNKDMN_CON_K1'

# Flag for out-of-sample (True) or in-sample (False) estimation; boolean scalar
oosflag = True

# Select return type: 'R' (assets), 'Q' (Q-portfolios), 'X' (managed portfolios); string scalar
QXorR = 'R'

# Flag to annualize factor returns (False for monthly); boolean scalar
annualize = False

# Choose factor model: 'FF', 'SY', 'HXZ', 'BS'; string scalar
ObsChoice = 'FF'

# 2. Load IPCA results (returns & mask)
# Load data from .npz file; assumes same structure as .mat
data = np.load(f'../IPCA_KELLY/{dataname}.npz')
# Expected: xret [N×T], LOC [N×T], date [T×1], possibly Q [N×T], X [L×T]
xret = data['xret']  # [N×T]
LOC = data['LOC']    # [N×T]
date = data['date']  # [T×1]
# Check for Q and X later in QXorR switch to avoid errors

# 3. Switch on factor family
if ObsChoice == 'FF':
    # 3.1 Load Fama-French (+MOM) data
    # Set suffix based on annualization flag; string scalar
    if annualize:
        ffsuffix = '_ANNRET'
    else:
        ffsuffix = ''
    
    # Load FF factor data from .npz
    ffdata = np.load(f'../IPCA_KELLY/F-F_Research_Data_5_Factors_2x3_plusMOM{ffsuffix}.npz')
    # Expected: ffdata['dates'] [T_FF×1], Mkt_RF, SMB, HML, MOM, RMW, CMA [T_FF×1]

    # 3.2 Align dates
    # Find intersection of factor and sample dates
    date, idx_ff, idx_ret = np.intersect1d(ffdata['dates'], date, return_indices=True)
    # date [T×1], idx_ff [T×1], idx_ret [T×1]

    # 3.3 Build factor arrays
    # Market factor; [T×1]
    FF1 = ffdata['Mkt_RF'][idx_ff]
    # FF3: Market, SMB, HML; [T×3]
    FF3 = np.column_stack((ffdata['Mkt_RF'][idx_ff], ffdata['SMB'][idx_ff], ffdata['HML'][idx_ff]))
    # FF4: FF3 + MOM; [T×4]
    FF4 = np.column_stack((FF3, ffdata['MOM'][idx_ff]))
    # FF5: FF3 + RMW, CMA; [T×5]
    FF5 = np.column_stack((FF3, ffdata['RMW'][idx_ff], ffdata['CMA'][idx_ff]))
    # FF6: FF5 + MOM; [T×6]
    FF6 = np.column_stack((FF5, ffdata['MOM'][idx_ff]))

    # 3.4 Select RET matrix
    if QXorR == 'R':
        # Use individual asset returns; [N×T]
        if 'xret' not in data:
            raise ValueError('xret not found in data for QXorR=R')
        RET = xret[:, idx_ret]
        Rlogic = True
    elif QXorR == 'Q':
        # Use Q-portfolio returns; [N×T]
        if 'Q' not in data:
            raise ValueError('Q not found in data for QXorR=Q')
        RET = data['Q'][:, idx_ret]
        Rlogic = False
    elif QXorR == 'X':
        # Use managed portfolio returns; [L×T]
        if 'X' not in data:
            raise ValueError('X not found in data for QXorR=X')
        RET = data['X'][:, idx_ret]
        Rlogic = False
    else:
        raise ValueError('Invalid QXorR value')

    # 3.5 Preallocate fit matrices
    # Realized fits for FF1; [N×T] or [L×T]
    FITS_FF1 = np.full_like(RET, np.nan)
    # Realized fits for FF3; [N×T] or [L×T]
    FITS_FF3 = np.full_like(RET, np.nan)
    # Realized fits for FF4; [N×T] or [L×T]
    FITS_FF4 = np.full_like(RET, np.nan)
    # Realized fits for FF5; [N×T] or [L×T]
    FITS_FF5 = np.full_like(RET, np.nan)
    # Realized fits for FF6; [N×T] or [L×T]
    FITS_FF6 = np.full_like(RET, np.nan)
    # Predictive fits for FF1; [N×T] or [L×T]
    FITS_cond_FF1 = np.full_like(RET, np.nan)
    # Predictive fits for FF3; [N×T] or [L×T]
    FITS_cond_FF3 = np.full_like(RET, np.nan)
    # Predictive fits for FF4; [N×T] or [L×T]
    FITS_cond_FF4 = np.full_like(RET, np.nan)
    # Predictive fits for FF5; [N×T] or [L×T]
    FITS_cond_FF5 = np.full_like(RET, np.nan)
    # Predictive fits for FF6; [N×T] or [L×T]
    FITS_cond_FF6 = np.full_like(RET, np.nan)

    # 3.6 Precompute factor means for Pred R²
    # Mean for FF1, repeated T times; [T×1]
    FF1_mean = np.repeat(np.mean(FF1, axis=0, keepdims=True), RET.shape[1], axis=0)
    # Mean for FF3, repeated T times; [T×3]
    FF3_mean = np.repeat(np.mean(FF3, axis=0, keepdims=True), RET.shape[1], axis=0)
    # Mean for FF4, repeated T times; [T×4]
    FF4_mean = np.repeat(np.mean(FF4, axis=0, keepdims=True), RET.shape[1], axis=0)
    # Mean for FF5, repeated T times; [T×5]
    FF5_mean = np.repeat(np.mean(FF5, axis=0, keepdims=True), RET.shape[1], axis=0)
    # Mean for FF6, repeated T times; [T×6]
    FF6_mean = np.repeat(np.mean(FF6, axis=0, keepdims=True), RET.shape[1], axis=0)

    # 3.7 (Optional) Tangency fits for OOS Sharpe
    # Tangency portfolio returns for FF1; [T×1]
    FITS_OOSTan_FF1 = np.full((RET.shape[1], 1), np.nan)
    # Tangency portfolio returns for FF2; [T×1]
    FITS_OOSTan_FF2 = np.full((RET.shape[1], 1), np.nan)
    # Tangency portfolio returns for FF3; [T×1]
    FITS_OOSTan_FF3 = np.full((RET.shape[1], 1), np.nan)
    # Tangency portfolio returns for FF4; [T×1]
    FITS_OOSTan_FF4 = np.full((RET.shape[1], 1), np.nan)
    # Tangency portfolio returns for FF5; [T×1]
    FITS_OOSTan_FF5 = np.full((RET.shape[1], 1), np.nan)
    # Tangency portfolio returns for FF6; [T×1]
    FITS_OOSTan_FF6 = np.full((RET.shape[1], 1), np.nan)

    # 4. Regressions & fits
    # Display start of FF model estimation
    print(f"Observable_Factor_Regressions (FF) on {dataname} | OOS={oosflag}")

    # Get dimensions of RET
    N, T = RET.shape  # N: assets/portfolios, T: time periods

    # Loop over each asset/portfolio
    for n in range(N):
        # Check if OOS estimation is enabled
        if oosflag:
            # Loop over time periods starting from t=61 (OOS)
            for t in range(60, T):  # 0-based indexing in Python
                # Skip if missing data and Rlogic is True
                if Rlogic and not LOC[n, t]:
                    continue
                # Set start index for recursive window
                st = 0  # recursive from start
                # Extract past returns for asset n; [(t-st)×1]
                y = RET[n, st:t].reshape(-1, 1)
                # Past factors for FF1; [(t-st)×1]
                f1 = FF1[st:t].reshape(-1, 1)
                # Past factors for FF3; [(t-st)×3]
                f3 = FF3[st:t]
                # Past factors for FF4; [(t-st)×4]
                f4 = FF4[st:t]
                # Past factors for FF5; [(t-st)×5]
                f5 = FF5[st:t]
                # Past factors for FF6; [(t-st)×6]
                f6 = FF6[st:t]
                # Check for sufficient non-NaN data
                if np.sum(~np.isnan(y)) < 60:
                    continue
                else:
                    y_loc = ~np.isnan(y).ravel()
                    y = y[y_loc, :]
                    f1_loc = f1[y_loc, :]
                    f3_loc = f3[y_loc, :]
                    f4_loc = f4[y_loc, :]
                    f5_loc = f5[y_loc, :]
                    f6_loc = f6[y_loc, :]
                    # Estimate betas using OLS (no intercept)
                    model1 = LinearRegression(fit_intercept=False).fit(f1_loc, y)
                    b1 = model1.coef_  # [1×1]
                    model3 = LinearRegression(fit_intercept=False).fit(f3_loc, y)
                    b3 = model3.coef_  # [3×1]
                    model4 = LinearRegression(fit_intercept=False).fit(f4_loc, y)
                    b4 = model4.coef_  # [4×1]
                    model5 = LinearRegression(fit_intercept=False).fit(f5_loc, y)
                    b5 = model5.coef_  # [5×1]
                    model6 = LinearRegression(fit_intercept=False).fit(f6_loc, y)
                    b6 = model6.coef_  # [6×1]

                    # Realized fits at time t
                    FITS_FF1[n, t] = FF1[t] @ b1.ravel()  # scalar
                    FITS_FF3[n, t] = FF3[t] @ b3.ravel()  # scalar
                    FITS_FF4[n, t] = FF4[t] @ b4.ravel()  # scalar
                    FITS_FF5[n, t] = FF5[t] @ b5.ravel()  # scalar
                    FITS_FF6[n, t] = FF6[t] @ b6.ravel()  # scalar

                    # Predictive fits using factor means
                    FITS_cond_FF1[n, t] = np.mean(f1, axis=0) @ b1.ravel()  # scalar
                    FITS_cond_FF3[n, t] = np.mean(f3, axis=0) @ b3.ravel()  # scalar
                    FITS_cond_FF4[n, t] = np.mean(f4, axis=0) @ b4.ravel()  # scalar
                    FITS_cond_FF5[n, t] = np.mean(f5, axis=0) @ b5.ravel()  # scalar
                    FITS_cond_FF6[n, t] = np.mean(f6, axis=0) @ b6.ravel()  # scalar

                    # OOS tangency returns (using FF6 as base, slicing columns)
                    FITS_OOSTan_FF1[t], _ = tanptfnext(f6[:, :1], FF6[t, :1].reshape(1, -1))  # scalar
                    FITS_OOSTan_FF2[t], _ = tanptfnext(f6[:, :2], FF6[t, :2].reshape(1, -1))  # scalar
                    FITS_OOSTan_FF3[t], _ = tanptfnext(f6[:, :3], FF6[t, :3].reshape(1, -1))  # scalar
                    FITS_OOSTan_FF4[t], _ = tanptfnext(f6[:, :4], FF6[t, :4].reshape(1, -1))  # scalar
                    FITS_OOSTan_FF5[t], _ = tanptfnext(f6[:, :5], FF6[t, :5].reshape(1, -1))  # scalar
                    FITS_OOSTan_FF6[t], _ = tanptfnext(f6[:, :6], FF6[t, :6].reshape(1, -1))  # scalar
        else:
            # Full-sample estimation
            # Skip if insufficient data and Rlogic is True
            if Rlogic and np.sum(LOC[n, :]) < 12:
                continue
            # Select valid time periods using LOC for asset n
            if Rlogic:
                valid_t = LOC[n, :]  # [T] boolean mask
            else:
                valid_t = np.full(LOC[n, :].shape, True)
            # Extract valid returns; [T_valid×1]
            y = RET[n, valid_t].reshape(-1, 1)
            # Extract valid factors
            f1 = FF1[valid_t].reshape(-1, 1)  # [T_valid×1]
            f3 = FF3[valid_t]                  # [T_valid×3]
            f4 = FF4[valid_t]                  # [T_valid×4]
            f5 = FF5[valid_t]                  # [T_valid×5]
            f6 = FF6[valid_t]                  # [T_valid×6]
            # Check if enough valid data points
            if np.sum(~np.isnan(y)) < 12:
                continue
            # Estimate betas using OLS on valid data
            model1 = LinearRegression(fit_intercept=False).fit(f1, y)
            b1 = model1.coef_  # [1×1]
            model3 = LinearRegression(fit_intercept=False).fit(f3, y)
            b3 = model3.coef_  # [3×1]
            model4 = LinearRegression(fit_intercept=False).fit(f4, y)
            b4 = model4.coef_  # [4×1]
            model5 = LinearRegression(fit_intercept=False).fit(f5, y)
            b5 = model5.coef_  # [5×1]
            model6 = LinearRegression(fit_intercept=False).fit(f6, y)
            b6 = model6.coef_  # [6×1]

            # Realized fits for valid time periods
            FITS_FF1[n, :] = (FF1 @ b1.T).flatten()
            FITS_FF3[n, :] = (FF3 @ b3.T).flatten()
            FITS_FF4[n, :] = (FF4 @ b4.T).flatten()
            FITS_FF5[n, :] = (FF5 @ b5.T).flatten()
            FITS_FF6[n, :] = (FF6 @ b6.T).flatten()

            # Predictive fits using factor means
            FITS_cond_FF1[n, :] = (FF1_mean @ b1.T).flatten()
            FITS_cond_FF3[n, :] = (FF3_mean @ b3.T).flatten()
            FITS_cond_FF4[n, :] = (FF4_mean @ b4.T).flatten()
            FITS_cond_FF5[n, :] = (FF5_mean @ b5.T).flatten()
            FITS_cond_FF6[n, :] = (FF6_mean @ b6.T).flatten()

        # Print progress every 100 assets
        if n % 100 == 0:
            print(f'  processed asset {n} of {N}')

    # 5. Build LOC & compute R²
    # Create logical mask for non-NaN fits; [N×T]
    if Rlogic:
        tmp = ~np.isnan(FITS_FF1 + FITS_FF3 + FITS_FF4 + FITS_FF5 + FITS_FF6)
        FITS_FFLOC = np.zeros_like(LOC, dtype=bool)
        FITS_FFLOC[:, idx_ret] = tmp
    else:
        FITS_FFLOC = ~np.isnan(FITS_FF1 + FITS_FF3 + FITS_FF4 + FITS_FF5 + FITS_FF6)

    # Store factors; [T×6]
    FITS_Factors = FF6

    # Mask missing returns for R² calculation
    xret[~LOC] = np.nan

    # Compute total sum of squares
    totSOS = mySOS(xret)  # scalar

    # Asset-level Total R²
    R2_tot_FF1 = 1 - mySOS(RET[FITS_FFLOC] - FITS_FF1[FITS_FFLOC]) / totSOS  # scalar
    R2_tot_FF3 = 1 - mySOS(RET[FITS_FFLOC] - FITS_FF3[FITS_FFLOC]) / totSOS  # scalar
    R2_tot_FF4 = 1 - mySOS(RET[FITS_FFLOC] - FITS_FF4[FITS_FFLOC]) / totSOS  # scalar
    R2_tot_FF5 = 1 - mySOS(RET[FITS_FFLOC] - FITS_FF5[FITS_FFLOC]) / totSOS  # scalar
    R2_tot_FF6 = 1 - mySOS(RET[FITS_FFLOC] - FITS_FF6[FITS_FFLOC]) / totSOS  # scalar

    # Asset-level Predictive R²
    R2_pre_FF1 = 1 - mySOS(RET[FITS_FFLOC] - FITS_cond_FF1[FITS_FFLOC]) / totSOS  # scalar
    R2_pre_FF3 = 1 - mySOS(RET[FITS_FFLOC] - FITS_cond_FF3[FITS_FFLOC]) / totSOS  # scalar
    R2_pre_FF4 = 1 - mySOS(RET[FITS_FFLOC] - FITS_cond_FF4[FITS_FFLOC]) / totSOS  # scalar
    R2_pre_FF5 = 1 - mySOS(RET[FITS_FFLOC] - FITS_cond_FF5[FITS_FFLOC]) / totSOS  # scalar
    R2_pre_FF6 = 1 - mySOS(RET[FITS_FFLOC] - FITS_cond_FF6[FITS_FFLOC]) / totSOS  # scalar

    # Display key R² results
    print('\nFF Models R² (Total | Pred):')
    print(f'  1-factor: {R2_tot_FF1:.3f} | {R2_pre_FF1:.3f}')
    print(f'  3-factor: {R2_tot_FF3:.3f} | {R2_pre_FF3:.3f}')
    print(f'  4-factor: {R2_tot_FF4:.3f} | {R2_pre_FF4:.3f}')
    print(f'  5-factor: {R2_tot_FF5:.3f} | {R2_pre_FF5:.3f}')
    print(f'  6-factor: {R2_tot_FF6:.3f} | {R2_pre_FF6:.3f}')

    # 6. Save results
    # Set OOS string for filename
    if oosflag:
        oosstr = 'OOS'
        suffix = '_rec_60_60'
    else:
        oosstr = ''
        # Suffix (empty for default)
        suffix = ''
    # Save results to .npz
    np.savez(f'../IPCA_KELLY/Results_ObsFactReg{QXorR}{oosstr}_{dataname}{suffix}.npz',
            FITS_FF1=FITS_FF1, FITS_FF3=FITS_FF3, FITS_FF4=FITS_FF4,
            FITS_FF5=FITS_FF5, FITS_FF6=FITS_FF6,
            FITS_cond_FF1=FITS_cond_FF1, FITS_cond_FF3=FITS_cond_FF3,
            FITS_cond_FF4=FITS_cond_FF4, FITS_cond_FF5=FITS_cond_FF5,
            FITS_cond_FF6=FITS_cond_FF6,
            FITS_OOSTan_FF1=FITS_OOSTan_FF1, FITS_OOSTan_FF2=FITS_OOSTan_FF2,
            FITS_OOSTan_FF3=FITS_OOSTan_FF3, FITS_OOSTan_FF4=FITS_OOSTan_FF4,
            FITS_OOSTan_FF5=FITS_OOSTan_FF5, FITS_OOSTan_FF6=FITS_OOSTan_FF6,
            FITS_Factors=FITS_Factors, FITS_FFLOC=FITS_FFLOC, date=date, RET=RET)

elif ObsChoice == 'SY':
    # 3.1 Load Stambaugh-Yuan data
    # Set suffix based on annualization flag
    if annualize:
        ffsuffix = '_ANNRET'
    else:
        ffsuffix = ''
    
    # Load SY factor data from .npz
    sydata = np.load(f'../IPCA_KELLY/M4{ffsuffix}.npz')
    # Expected: sydata['dates'] [T_SY×1], MKTRF, SMB, MGMT, PERF [T_SY×1]

    # 3.2 Align dates
    # Find intersection of factor and sample dates
    date, locSY, locRET = np.intersect1d(sydata['dates'], date, return_indices=True)
    # date [T×1], locSY [T×1], locRET [T×1]

    # 3.3 Build factor arrays
    # SY factors: MKTRF, SMB, MGMT, PERF; [T×4]
    SY = np.column_stack((sydata['MKTRF'].T[locSY], sydata['SMB'].T[locSY],
                          sydata['MGMT'].T[locSY], sydata['PERF'].T[locSY]))

    # 3.4 Select RET matrix
    if QXorR == 'R':
        # Use individual asset returns
        if 'xret' not in data:
            raise ValueError('xret not found in data for QXorR=R')
        RET = xret[:, locRET]
        Rlogic = True
    elif QXorR == 'Q':
        # Use Q-portfolio returns
        if 'Q' not in data:
            raise ValueError('Q not found in data for QXorR=Q')
        RET = data['Q'][:, locRET]
        Rlogic = False
    elif QXorR == 'X':
        # Use managed portfolio returns
        if 'X' not in data:
            raise ValueError('X not found in data for QXorR=X')
        RET = data['X'][:, locRET]
        Rlogic = False
    else:
        raise ValueError('Invalid QXorR value')

    # 3.5 Preallocate fit matrices
    # Realized fits; [N×T] or [L×T]
    FITS_SY = np.full_like(RET, np.nan)
    # Predictive fits; [N×T] or [L×T]
    FITS_cond_SY = np.full_like(RET, np.nan)

    # 3.6 Precompute factor means for Pred R²
    # Mean for SY, repeated T times; [T×4]
    SY_mean = np.repeat(np.mean(SY, axis=0, keepdims=True), RET.shape[1], axis=0)

    # 3.7 (Optional) Tangency fits for OOS Sharpe
    # Tangency portfolio returns; [T×1]
    FITS_OOSTan_SY = np.full((RET.shape[1], 1), np.nan)

    # 4. Regressions & fits
    # Display start of SY model estimation
    print(f"Observable_Factor_Regressions (SY) on {dataname} | OOS={oosflag}")

    # Get dimensions
    N, T = RET.shape

    # Loop over each asset/portfolio
    for n in range(N):
        # Check if OOS estimation is enabled
        if oosflag:
            # Loop over time periods starting from t=61
            for t in range(60, T):
                # Skip if missing data and Rlogic is True
                if Rlogic and not LOC[n, t]:
                    continue
                # Set start index for recursive window
                st = 0
                # Extract past returns
                y = RET[n, st:t].reshape(-1, 1)
                # Past factors for SY; [(t-st)×4]
                f = SY[st:t]
                # Check for sufficient non-NaN data
                if np.sum(~np.isnan(y)) < 60:
                    continue
                else:
                    y_loc = ~np.isnan(y).ravel()
                    y = y[y_loc, :]
                    f_loc = f[y_loc, :]
                    # Estimate betas
                    model = LinearRegression(fit_intercept=False).fit(f_loc, y)
                    b = model.coef_  # [4×1]

                    # Realized fit at time t
                    FITS_SY[n, t] = SY[t] @ b.ravel()  # scalar

                    # Predictive fit
                    FITS_cond_SY[n, t] = np.mean(f, axis=0) @ b.ravel()  # scalar

                    # OOS tangency return
                    FITS_OOSTan_SY[t], _ = tanptfnext(f[:, :1], SY[t, :1].reshape(-1, 1))  # scalar
        else:
            # Full-sample estimation
            # Skip if insufficient data and Rlogic is True
            if Rlogic and np.sum(LOC[n, :]) < 12:
                continue
            # Select valid time periods using LOC for asset n
            if Rlogic:
                valid_t = LOC[n, :]  # [T] boolean mask
            else:
                valid_t = np.full(LOC[n, :].shape, True)
            # Extract valid returns
            y = RET[n, valid_t].reshape(-1, 1)  # [T_valid×1]
            # Extract valid factors
            f = SY[valid_t]  # [T_valid×4]
            # Check if enough valid data points
            if np.sum(~np.isnan(y)) < 12:
                continue
            # Estimate betas
            model = LinearRegression(fit_intercept=False).fit(f, y)
            b = model.coef_  # [1x4]

            # Realized fits for valid time periods
            FITS_SY[n, :] = (SY @ b.T).flatten()
            # Predictive fits
            FITS_cond_SY[n, :] = (SY_mean @ b.T).flatten()

        # Print progress
        if n % 100 == 0:
            print(f'  processed asset {n} of {N}')

    # 5. Build LOC & compute R²
    # Create logical mask for non-NaN fits
    if Rlogic:
        tmp = ~np.isnan(FITS_SY)
        FITS_SYLOC = np.zeros_like(LOC, dtype=bool)
        FITS_SYLOC[:, locRET] = tmp
    else:
        FITS_SYLOC = ~np.isnan(FITS_SY)

    # Store factors
    FITS_Factors = SY  # [T×4]

    # Mask missing returns
    xret[~LOC] = np.nan

    # Compute total sum of squares
    totSOS = mySOS(xret)  # scalar

    # Asset-level Total R²
    R2_tot_SY = 1 - mySOS(xret[LOC] - FITS_SY[LOC]) / totSOS  # scalar

    # Asset-level Predictive R²
    R2_pre_SY = 1 - mySOS(xret[LOC] - FITS_cond_SY[LOC]) / totSOS  # scalar

    # Display key R² results
    print('\nSY Models R² (Total | Pred):')
    print(f'  4-factor: {R2_tot_SY:.3f} | {R2_pre_SY:.3f}')

    # 6. Save results
    # Set OOS string
    if oosflag:
        oosstr = 'OOS'
    else:
        oosstr = ''
    suffix = '' # adjust if OOS spec
    # Save to .npz
    np.savez(f'../IPCA_KELLY/Results_ObsFactReg_SY_{QXorR}{oosstr}_{dataname}{suffix}.npz',
            FITS_SY=FITS_SY, FITS_cond_SY=FITS_cond_SY,
            FITS_OOSTan_SY=FITS_OOSTan_SY, FITS_Factors=FITS_Factors,
            FITS_SYLOC=FITS_SYLOC, date=date, RET=RET)

elif ObsChoice == 'HXZ':
    # 3.1 Load Hou-Xue-Zhang data
    # Set suffix
    if annualize:
        ffsuffix = '_ANNRET'
    else:
        ffsuffix = ''
    
    # Load HXZ factor data
    hxzdata = np.load(f'../IPCA_KELLY/HXZ_q-Factors_monthly{ffsuffix}.npz')
    # Expected: hxzdata['yrmo'] [T_HXZ×1], Mkt_RF, ME, IA, ROE [T_HXZ×1]

    # 3.2 Align dates
    date, locHXZ, locRET = np.intersect1d(hxzdata['yrmo'], date, return_indices=True)
    # date [T×1], locHXZ [T×1], locRET [T×1]

    # 3.3 Build factor arrays
    # HXZ factors: Mkt_RF, ME, IA, ROE; [T×4]
    HXZ = np.column_stack((hxzdata['Mkt_RF'][locHXZ], hxzdata['ME'][locHXZ],
                        hxzdata['IA'][locHXZ], hxzdata['ROE'][locHXZ]))

    # 3.4 Select RET matrix
    if QXorR == 'R':
        if 'xret' not in data:
            raise ValueError('xret not found in data for QXorR=R')
        RET = xret[:, locRET]
        Rlogic = True
    elif QXorR == 'Q':
        if 'Q' not in data:
            raise ValueError('Q not found in data for QXorR=Q')
        RET = data['Q'][:, locRET]
        Rlogic = False
    elif QXorR == 'X':
        if 'X' not in data:
            raise ValueError('X not found in data for QXorR=X')
        RET = data['X'][:, locRET]
        Rlogic = False
    else:
        raise ValueError('Invalid QXorR value')

    # 3.5 Preallocate fit matrices
    FITS_HXZ = np.full_like(RET, np.nan)
    FITS_cond_HXZ = np.full_like(RET, np.nan)

    # 3.6 Precompute factor means
    HXZ_mean = np.repeat(np.mean(HXZ, axis=0, keepdims=True), RET.shape[1], axis=0)

    # 3.7 (Optional) Tangency fits
    FITS_OOSTan_HXZ = np.full((RET.shape[1], 1), np.nan)

    # 4. Regressions & fits
    print(f"Observable_Factor_Regressions (HXZ) on {dataname} | OOS={oosflag}")
    N, T = RET.shape

    for n in range(N):
        if oosflag:
            for t in range(60, T):
                if Rlogic and not LOC[n, t]:
                    continue
                st = 0
                y = RET[n, st:t].reshape(-1, 1)
                f = HXZ[st:t]
                if np.sum(~np.isnan(y)) < 60:
                    continue
                else:
                    y_loc = ~np.isnan(y).ravel()
                    y = y[y_loc, :]
                    f_loc = f[y_loc, :]
                    model = LinearRegression(fit_intercept=False).fit(f_loc, y)
                    b = model.coef_
                    FITS_HXZ[n, t] = HXZ[t] @ b.ravel()
                    FITS_cond_HXZ[n, t] = np.mean(f, axis=0) @ b.ravel()
                    FITS_OOSTan_HXZ[t], _ = tanptfnext(f[:, :1], HXZ[t, :1].reshape(-1, 1))
        else:
            # Full-sample estimation
            # Skip if insufficient data and Rlogic is True
            if Rlogic and np.sum(LOC[n, :]) < 12:
                continue
            # Select valid time periods using LOC for asset n
            LOC_REG = LOC[:, locRET]
            valid_t = LOC_REG[n, :]  # [T] boolean mask
            # Extract valid returns
            y = RET[n, valid_t].reshape(-1, 1)  # [T_valid×1]
            # Extract valid factors
            f = HXZ[valid_t]  # [T_valid×4]
            # Check if enough valid data points
            if np.sum(~np.isnan(y)) < 12:
                continue
            # Estimate betas
            model = LinearRegression(fit_intercept=False).fit(f, y)
            b = model.coef_  # [1x4]

            # Realized fits for valid time periods
            FITS_HXZ[n, :] = (HXZ @ b.T).flatten()
            # Predictive fits
            FITS_cond_HXZ[n, :] = (HXZ_mean @ b.T).flatten()

        if n % 100 == 0:
            print(f'  processed asset {n} of {N}')

    # 5. Build LOC & compute R²
    if Rlogic:
        tmp = ~np.isnan(FITS_HXZ)
        FITS_HXZLOC = np.zeros_like(LOC, dtype=bool)
        FITS_HXZLOC[:, locRET] = tmp
        tmp = np.full(LOC.shape, np.nan)
        tmp[:, locRET] = FITS_HXZ
        FITS_HXZ = tmp
        tmp = np.full(LOC.shape, np.nan)
        tmp[:, locRET] = FITS_cond_HXZ
        FITS_cond_HXZ = tmp
    else:
        FITS_HXZLOC = ~np.isnan(FITS_HXZ)

    FITS_Factors = HXZ

    xret[~LOC] = np.nan
    totSOS = mySOS(xret)
    R2_tot_HXZ = 1 - mySOS(xret[LOC] - FITS_HXZ[LOC]) / totSOS
    R2_pre_HXZ = 1 - mySOS(xret[LOC] - FITS_cond_HXZ[LOC]) / totSOS

    print('\nHXZ Models R² (Total | Pred):')
    print(f'  4-factor: {R2_tot_HXZ:.3f} | {R2_pre_HXZ:.3f}')

    if oosflag:
        oosstr = 'OOS'
    else:
        oosstr = ''
    suffix = '' # adjust if OOS spec
    np.savez(f'../IPCA_KELLY/Results_ObsFactReg_HXZ_{QXorR}{oosstr}_{dataname}{suffix}.npz',
            FITS_HXZ=FITS_HXZ, FITS_cond_HXZ=FITS_cond_HXZ,
            FITS_OOSTan_HXZ=FITS_OOSTan_HXZ, FITS_Factors=FITS_Factors,
            FITS_HXZLOC=FITS_HXZLOC, date=date, RET=RET)

elif ObsChoice == 'BS':
    # 3.1 Load Barillas-Shanken data
    if annualize:
        ffsuffix = '_ANNRET'
    else:
        ffsuffix = ''
    
    bsdata = np.load(f'../IPCA_KELLY/BarillasShanken{ffsuffix}.npz')
    # Expected: bsdata['dates'] [T_BS×1], Mkt_RF, SMB, UMD, HMLm, IA, ROE [T_BS×1]

    # 3.2 Align dates
    date, locBS, locRET = np.intersect1d(bsdata['dates'], date, return_indices=True)

    # 3.3 Build factor arrays
    BS = np.column_stack((bsdata['Mkt_RF'].T[locBS], bsdata['SMB'].T[locBS],
                        bsdata['UMD'].T[locBS], bsdata['HMLm'].T[locBS],
                        bsdata['IA'].T[locBS], bsdata['ROE'].T[locBS]))

    # 3.4 Select RET matrix
    if QXorR == 'R':
        if 'xret' not in data:
            raise ValueError('xret not found in data for QXorR=R')
        RET = xret[:, locRET]
        Rlogic = True
    elif QXorR == 'Q':
        if 'Q' not in data:
            raise ValueError('Q not found in data for QXorR=Q')
        RET = data['Q'][:, locRET]
        Rlogic = False
    elif QXorR == 'X':
        if 'X' not in data:
            raise ValueError('X not found in data for QXorR=X')
        RET = data['X'][:, locRET]
        Rlogic = False
    else:
        raise ValueError('Invalid QXorR value')

    # 3.5 Preallocate fit matrices
    FITS_BS = np.full_like(RET, np.nan)
    FITS_cond_BS = np.full_like(RET, np.nan)

    # 3.6 Precompute factor means
    BS_mean = np.repeat(np.mean(BS, axis=0, keepdims=True), RET.shape[1], axis=0)

    # 3.7 (Optional) Tangency fits
    FITS_OOSTan_BS = np.full((RET.shape[1], 1), np.nan)

    # 4. Regressions & fits
    print(f"Observable_Factor_Regressions (BS) on {dataname} | OOS={oosflag}")
    N, T = RET.shape

    for n in range(N):
        if oosflag:
            for t in range(60, T):
                if Rlogic and not LOC[n, t]:
                    continue
                st = 0
                y = RET[n, st:t].reshape(-1, 1)
                f = BS[st:t]
                if np.sum(~np.isnan(y)) < 60:
                    continue
                else:
                    y_loc = ~np.isnan(y).ravel()
                    y = y[y_loc, :]
                    f_loc = f[y_loc, :]
                    model = LinearRegression(fit_intercept=False).fit(f_loc, y)
                    b = model.coef_
                    FITS_BS[n, t] = BS[t] @ b.ravel()
                    FITS_cond_BS[n, t] = np.mean(f, axis=0) @ b.ravel()
                    FITS_OOSTan_BS[t], _ = tanptfnext(f[:, :1], BS[t, :1].reshape(-1, 1))
        else:
            # Full-sample estimation
            # Skip if insufficient data and Rlogic is True
            if Rlogic and np.sum(LOC[n, :]) < 12:
                continue
            LOC_REG = LOC[:, locRET]
            # Select valid time periods using LOC for asset n
            valid_t = LOC_REG[n, :]  # [T] boolean mask
            # Extract valid returns
            y = RET[n, valid_t].reshape(-1, 1)  # [T_valid×1]
            # Extract valid factors
            f = BS[valid_t]  # [T_valid×6]
            # Check if enough valid data points
            if np.sum(~np.isnan(y)) < 12:
                continue
            # Estimate betas
            model = LinearRegression(fit_intercept=False).fit(f, y)
            b = model.coef_  # [1x6]

            # Realized fits for valid time periods
            FITS_BS[n, :] = (BS @ b.T).flatten()
            # Predictive fits
            FITS_cond_BS[n, :] = (BS_mean @ b.T).flatten()

        if n % 1000 == 0:
            print(f'  processed asset {n} of {N}')

    # 5. Build LOC & compute R²
    if Rlogic:
        tmp = ~np.isnan(FITS_BS)
        FITS_BSLOC = np.zeros_like(LOC, dtype=bool)
        FITS_BSLOC[:, locRET] = tmp
        tmp = np.full(LOC.shape, np.nan)
        tmp[:, locRET] = FITS_BS
        FITS_BS = tmp
        tmp = np.full(LOC.shape, np.nan)
        tmp[:, locRET] = FITS_cond_BS
        FITS_cond_BS = tmp
    else:
        FITS_BSLOC = ~np.isnan(FITS_BS)

    FITS_Factors = BS

    xret[~LOC] = np.nan
    totSOS = mySOS(xret)
    R2_tot_BS = 1 - mySOS(xret[LOC] - FITS_BS[LOC]) / totSOS
    R2_pre_BS = 1 - mySOS(xret[LOC] - FITS_cond_BS[LOC]) / totSOS

    print('\nBS Models R² (Total | Pred):')
    print(f'  6-factor: {R2_tot_BS:.3f} | {R2_pre_BS:.3f}')

    if oosflag:
        oosstr = 'OOS'
    else:
        oosstr = ''
    suffix = '' # adjust if OOS spec
    np.savez(f'../IPCA_KELLY/Results_ObsFactReg_BS_{QXorR}{oosstr}_{dataname}{suffix}.npz',
            FITS_BS=FITS_BS, FITS_cond_BS=FITS_cond_BS,
            FITS_OOSTan_BS=FITS_OOSTan_BS, FITS_Factors=FITS_Factors,
            FITS_BSLOC=FITS_BSLOC, date=date, RET=RET)

else:
    # Raise error for invalid ObsChoice
    raise ValueError('Unknown ObsChoice: use FF, SY, HXZ, or BS')