# file: IPCA_empirical_datacall_FNW36.py
"""
Load, align, and preprocess characteristic and return data for IPCA estimation.
- Loads firm data and risk-free series from .npz files
- Applies log transform to 'lme' and 'at'
- Computes excess returns (raw - RF)
- Reorders characteristics according to literature publication order
- Outputs: xret (N,T), chars (N,T,L), charnames (L,), date (T,)
"""

import numpy as np

# --- 1. Load processed firm characteristics and returns ---
# Assumes 'characteristics_data_feb2017.npz' contains:
#   'char' (N,T,L), 'ret' (N,T), 'charnames' (L,), 'yrmo' (T,), 'permno' (N,)
data = np.load('characteristics_data_feb2017.npz', allow_pickle=True)
char = data['char']         # shape: (N, T, L)
ret = data['ret']           # shape: (N, T)
charnames = data['charnames'].tolist()  # list of strings, length L
yrmo = data['yrmo']         # shape: (T,)
permno = data['permno']     # shape: (N,)

# Rename for downstream scripts
chars = char.copy()         # shape: (N, T, L)
date = yrmo.copy()          # shape: (T,)

# --- 2. Log-transform selected characteristics ---
# Apply natural log to 'lme' and 'at'
for cname in ['lme', 'at']:
    if cname in charnames:
        idx = charnames.index(cname)
        chars[:, :, idx] = np.log(chars[:, :, idx])
    else:
        raise ValueError(f"Characteristic '{cname}' not found in charnames.")

# --- 3. Load risk-free rate series and align sample dates ---
# Fama-French RF file should contain:
#   'yrmolistFF' (P,), 'RF' (P,)
ffd = np.load('FF3F_RF_192607_201606.npz')
yrmolistFF = ffd['yrmolistFF']
RF = ffd['RF']              # in percent, shape: (P,)

# Align 'date' with 'yrmolistFF'
# Find indices in both arrays where dates match
common_dates, loc_date, loc_ff = np.intersect1d(date, yrmolistFF, return_indices=True)

if len(loc_date) != len(date):
    raise ValueError("RF series does not fully overlap the stock/chars sample period")

# --- 4. Compute excess returns ---
# Subtract aligned risk-free rate (convert to decimal)
RF_aligned = RF[loc_ff] / 100.0      # shape: (T,)
xret = ret - RF_aligned.T     # shape: (N, T)

# Optionally keep the aligned RF for reference
RF_used = RF_aligned

# --- 5. Reorder characteristics according to publication order ---
# List of (charName, publicationYear)
puborder = [
    ('a2me',              1988), 
    ('at',                2015), 
    ('ato',               2008), 
    ('beme',              1985), 
    ('beta',              1973),
    ('c',                 2012), 
    ('cto',               1996), 
    ('d2a',               2016), 
    ('dpi2a',             2008), 
    ('e2p',               1983),
    ('fc2y',              2016), 
    ('free_cf',           2011), 
    ('idio_vol',          2006), 
    ('investment',        2008), 
    ('lev',               2015),
    ('lme',               1992), 
    ('lturnover',         1998), 
    ('noa',               2004), 
    ('oa',                1996), 
    ('ol',                2011), 
    ('pcm',               2016),
    ('pm',                2008), 
    ('prof',              2015), 
    ('q',                 1985), 
    ('rel_to_high_price', 2004), 
    ('rna',               2008),
    ('roa',               2010), 
    ('roe',               1996), 
    ('cum_return_12_2',   1996), 
    ('cum_return_12_7',   2012),
    ('cum_return_1_0',    1990), 
    ('cum_return_36_13',  1985), 
    ('s2p',               2015), 
    ('sga2m',             2015),
    ('spread_mean',       2014), 
    ('suv',               2009),
]
# Sort by publication year
puborder_sorted = sorted(puborder, key=lambda x: x[1])
# Find the index order for charnames
char_order = [charnames.index(name) for name, _ in puborder_sorted]
chars = chars[:, :, char_order]
charnames = [charnames[i] for i in char_order]

# --- 6. Clean up (optional in Python) ---
# At this stage, you have:
#   chars: (N, T, L) reordered and log-transformed
#   xret:  (N, T) excess returns
#   charnames: reordered list of length L
#   date: (T,) vector of year-month codes

# --- 7. Example: Save the results for downstream use ---
np.savez('IPCA_empirical_datacall_FNW36_out.npz',
         xret=xret, chars=chars, charnames=np.array(charnames, dtype='object'), date=date)

# Example: Usage in next step
if __name__ == "__main__":
    print(f"xret shape: {xret.shape}")
    print(f"chars shape: {chars.shape}")
    print(f"First 5 characteristic names: {charnames[:5]}")
    print(f"First 5 dates: {date[:5]}")
