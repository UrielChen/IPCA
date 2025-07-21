import numpy as np

# tanptfnext.py
# -------------------------------------------------------------------------
# Main purpose:
#   Compute the realized out-of-sample return of a tangency (maximum Sharpe) portfolio
#   using historical excess returns to estimate weights and applying those weights to a
#   new (possibly future) return vector Xnext. Optionally rescales weights to match a
#   target volatility.
#
# Key Assumptions:
#   - Input X contains EXCESS returns (risk-free rate = 0).
#   - X: [T×N] numpy array, T periods, N assets.
#   - Xnext: [1×N] numpy array, excess returns for one period (test/future).
#   - If provided, targetvol is a positive scalar for volatility scaling.
#   - Input matrix X has no missing values (NaNs should be handled upstream).
#   - The covariance matrix is invertible (non-singular).
#
# Inputs:
#   - X: [T×N] numpy array, T periods of N-asset excess returns (historical data).
#   - Xnext: [1×N] numpy array, returns to apply weights to (future/test period).
#   - targetvol (optional): scalar, target volatility for scaling portfolio.
#
# Outputs:
#   - tpnext: [scalar] realized tangency portfolio return for Xnext.
#   - tw: [N×1] numpy array, tangency portfolio weights used.
#
# Prints:
#   - Sizes of input matrices X and Xnext.
#   - First 5 tangency weights (before/after scaling) and their sum.
#   - Intermediate statistics (means, realized volatility, scaling factor).
#   - Realized portfolio return (tpnext) and final portfolio volatility.
# -------------------------------------------------------------------------

def tanptfnext(X, Xnext, targetvol=None):
    # Get dimensions of historical returns matrix
    T, N = X.shape  # T: number of periods, N: number of assets; [T×N]
    
    # Print input dimensions for monitoring
    print('--- tanptfnext.py ---')
    print(f'Input X: {T} periods (T), {N} assets (N)')
    
    # Get dimensions of Xnext
    nrows_next, ncols_next = Xnext.shape  # Should be [1×N]
    
    # Print Xnext dimensions
    print(f'Input Xnext: {nrows_next} x {ncols_next}')
    
    # Set risk-free rate to zero (assumes excess returns)
    rf = 0  # scalar, risk-free rate assumed zero
    
    # Create vector of ones for portfolio calculations
    iota = np.ones((N, 1))  # [N×1], vector of ones
    
    # Compute sample covariance matrix of historical returns
    S = np.cov(X, rowvar=False)  # [N×N], sample covariance matrix
    
    # Compute sample mean vector of historical returns
    mu = np.mean(X, axis=0).reshape(-1, 1)  # [N×1], mean return for each asset
    
    # Compute numerator of tangency portfolio weights
    numerator = np.linalg.solve(S, mu - rf * iota)  # [N×1], S \ (mu - rf * iota)
    
    # Compute denominator of tangency portfolio weights
    denominator = (iota.T @ np.linalg.solve(S, mu - rf * iota))  # [1×1], (iota' / S) * (mu - rf * iota)
    
    # Compute tangency portfolio weights (maximum Sharpe, long-only unconstrained)
    tw = numerator / denominator  # [N×1], normalized weights
    
    # Check if weights produce negative expected return and flip if necessary
    if tw.T @ (mu - rf) < 0:
        tw = -tw  # [N×1], flip weights for positive expected return
        print('Weights flipped for positive expected return.')
    
    # Print first 5 tangency weights and their sum for monitoring
    print('Tangency portfolio weights (first 5):')
    print(tw[:min(5, len(tw))].flatten())  # Show first 5 weights
    print(f'Sum of tangency weights: {np.sum(tw):.4f}')
    
    # Check if target volatility is provided for rescaling
    if targetvol is not None:
        # Use provided target volatility; scalar
        targetvol = targetvol  # scalar, target volatility
        
        # Compute realized portfolio volatility
        realized_vol = np.sqrt(tw.T @ S @ tw)  # scalar, portfolio volatility
        
        # Compute scaling factor to match target volatility
        scale = targetvol / realized_vol  # scalar, scaling factor
        
        # Print rescaling information
        print(f'Rescaling portfolio: target vol {targetvol:.4f}, realized vol {realized_vol:.4f}, scale {scale:.4f}')
    else:
        # No scaling if targetvol not provided
        scale = 1  # scalar, no scaling
    
    # Apply scaling to weights
    tw = tw * scale  # [N×1], scaled tangency weights
    
    # Print first 5 scaled weights
    print('Scaled tangency weights (first 5):')
    print(tw[:min(5, len(tw))].flatten())
    
    # Compute tangency portfolio return for Xnext
    tpnext = Xnext @ tw  # [scalar], Xnext [1×N] @ tw [N×1]
    
    # Print realized tangency portfolio return
    print(f'tpnext (tangency portfolio return for Xnext): {tpnext[0,0]:.6f}')
    
    # Compute and print final realized portfolio volatility
    realized_vol_final = np.sqrt(tw.T @ S @ tw)  # scalar, final portfolio volatility
    print(f'Realized portfolio volatility after scaling: {realized_vol_final[0,0]:.4f}')
    
    # Return tangency portfolio return and weights
    return tpnext[0,0], tw  # tpnext: scalar, tw: [N×1]
