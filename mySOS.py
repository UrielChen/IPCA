# file: mySOS.py

# =========================================================================
# Purpose:
#   Compute the Sum‑Of‑Squares (SOS) of the entries of a vector or matrix,
#   ignoring any NaN entries.
#
# Main Assumptions:
#   - Input p may be a NumPy array (vector or matrix) containing NaNs.
#   - NaN values represent missing data and should be excluded from the SOS.
#   - The function returns a scalar: sum of squares over all non-NaN elements.
#
# Usage:
#   ret = mySOS(p)
#     p   = numeric NumPy array (any shape) containing real numbers and NaNs
#     ret = scalar sum-of-squares of p's non-NaN entries (float)
#
# Inputs:
#   p : numpy.ndarray
#       Vector or matrix (any size) containing real numbers and NaNs.
#
# Outputs:
#   ret : float
#       Scalar sum-of-squares of p's non-NaN entries.
# =========================================================================

import numpy as np

def mySOS(p):
    # --------------------------------------------------------------
    # Flatten input into a 1D array.
    # If p is shape (m, n), after flatten: (m*n,) 
    # This ensures we always sum over all elements regardless of original shape.
    flat_p = p.flatten()  # (m*n,)

    # --------------------------------------------------------------
    # Identify non-NaN entries using a boolean mask.
    # mask is (m*n,) boolean, True for valid (not NaN) entries.
    mask = ~np.isnan(flat_p)

    # --------------------------------------------------------------
    # Select valid entries (all non-NaN values).
    # valid_vals is 1D array of all non-NaN entries.
    valid_vals = flat_p[mask]  # shape: (num_non_nan,)

    # --------------------------------------------------------------
    # Compute the sum of squares of the non-NaN entries.
    # ret is a scalar (float)
    ret = np.dot(valid_vals, valid_vals)  # Equivalent to sum(valid_vals ** 2)

    # --------------------------------------------------------------
    # Return the result.
    return ret

# Example usage and test case
if __name__ == "__main__":
    # --------------------------------------------------------------
    # Create a test array with NaNs.
    # test_array is shape (2,3)
    test_array = np.array([[1.0, np.nan, 2.0],
                           [3.0, 4.0, np.nan]])
    # Print input for clarity.
    print("Input test array:")
    print(test_array)

    # --------------------------------------------------------------
    # Call mySOS on the test array and print the result.
    sos = mySOS(test_array)
    print(f"Sum of squares of non-NaN entries: {sos:.6f}")  # Should be 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30

# End of file
