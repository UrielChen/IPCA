import numpy as np

# tanptf.py
# -------------------------------------------------------------------------
# Main purpose:
#   Compute the realized returns of the tangency (maximum Sharpe) portfolio 
#   given a matrix of excess returns, using population mean and covariance.
#   This is used to evaluate the in-sample Sharpe ratio of a model-implied 
#   efficient portfolio, providing an upper bound on performance.
#
# Key Assumptions:
#   - Input X contains EXCESS returns (risk-free rate = 0).
#   - Time-series dimension is rows (T), assets are columns (N): X [T×N].
#   - Input matrix X has no missing values (NaNs should be handled upstream).
#   - The covariance matrix is invertible (non-singular).
#
# Inputs:
#   - X: [T×N] numpy array, each column is the time series of excess returns 
#        for one asset (T periods, N assets).
#
# Outputs:
#   - tp: [T×1] numpy array, tangency portfolio realized returns over time.
#
# Prints:
#   - Size of input matrix X and key statistics (means, covariances).
#   - First 5 tangency portfolio weights and their sum.
#   - First 5 realized returns and the Sharpe ratio of the portfolio.
# -------------------------------------------------------------------------

def tanptf(X):
    # Get dimensions of input matrix
    T, N = X.shape  # T: number of time periods, N: number of assets; [T×N]
    
    # Print input dimensions for monitoring
    print('--- tanptf.py ---')
    print(f'Input X: {T} periods (T), {N} assets (N)')
    
    # Set risk-free rate to zero (assumes X contains excess returns)
    rf = 0  # scalar
    
    # Create vector of ones for portfolio calculations
    iota = np.ones((N, 1))  # [N×1], vector of ones
    
    # Compute the sample covariance matrix of returns
    S = np.cov(X, rowvar=False)  # [N×N], sample covariance matrix
    S = np.atleast_2d(S)
    
    # Compute the sample mean return vector
    mu = np.mean(X, axis=0).reshape(-1, 1)  # [N×1], mean return for each asset
    
    # Compute the numerator of tangency portfolio weights
    numerator = np.linalg.solve(S, mu - rf * iota)  # [N×1], S \ (mu - rf * iota)
    
    # Compute the denominator of tangency portfolio weights
    denominator = (iota.T @ np.linalg.solve(S, mu - rf * iota))  # [1×1], (iota' / S) * (mu - rf * iota)
    
    # Compute tangency portfolio weights (maximum Sharpe, long-only unconstrained)
    tw = numerator / denominator  # [N×1], normalized weights
    
    # Print first 5 tangency weights and their sum for checking
    print('Tangency portfolio weights (first 5):')
    print(tw[:min(5, len(tw))].flatten())  # Show first 5 weights
    print(f'Sum of weights: {np.sum(tw):.4f}')
    
    # Compute tangency portfolio realized returns over time
    tp = X @ tw  # [T×1], portfolio returns
    
    # Print first 5 realized returns
    print(f'Output tp: [{T}×1] vector (first 5 returns):')
    print(tp[:min(5, len(tp))].flatten())
    
    # Compute and print realized Sharpe ratio for debugging
    realizedSR = np.mean(tp) / np.std(tp, ddof=1)  # scalar, mean/std
    print(f'Realized Sharpe Ratio of tangency portfolio: {realizedSR:.4f}')
    
    # Return tangency portfolio returns
    return tp  # [T×1]
