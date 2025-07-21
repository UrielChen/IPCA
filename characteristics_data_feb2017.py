# file: process_characteristics.py

"""
Fast conversion of characteristics CSV to compressed HDF5 and NPZ.
- Loads 'characteristics_data_feb2017.csv'.
- Produces: ret (N x T), char (N x T x K), charnames, yrmo, permno.
- Stores data as .h5 (HDF5) and .npz (NumPy compressed) for fast Python I/O.
"""

import pandas as pd
import numpy as np

# --- Config ---
csv_path = 'characteristics_data_feb2017.csv'
h5_path = 'characteristics_data_feb2017.h5'
npz_path = 'characteristics_data_feb2017.npz'

neuhierlchoices = [
    'lme', 'lturnover', 'spread_mean',
    'cum_return_1_0', 'cum_return_12_2', 'cum_return_12_7',
    'cum_return_36_13', 'beta', 'idio_vol', 'beme',
    'at', 'ato', 'c', 'cto', 'd2a', 'dpi2a', 'e2p', 'fc2y', 'free_cf',
    'investment', 'lev', 'noa', 'oa', 'ol', 'pcm', 'pm', 'prof',
    'q', 'rel_to_high_price', 'rna', 'roa', 'roe', 's2p', 'sga2m', 'suv',
    'a2me'
]
RETNAME = 'ret'

# --- Step 1: Load and clean ---
df = pd.read_csv(csv_path)
df.columns = [c.replace('"', '') for c in df.columns]

# --- Step 2: Parse date, id, mapping ---
df['yr'] = df.iloc[:, 1].astype(int)
df['mo'] = df.iloc[:, 2].astype(int)
df['bigyrmo'] = df['yr'] * 100 + df['mo']
df['permno'] = df.iloc[:, 4].astype(int)

yrmo = np.sort(df['bigyrmo'].unique())
permno = np.sort(df['permno'].unique())
yrmo_idx = {v: i for i, v in enumerate(yrmo)}
permno_idx = {v: i for i, v in enumerate(permno)}

# --- Step 3: Identify columns for characteristics and returns ---
charcols = [c for c in neuhierlchoices if c in df.columns]
charloc = [df.columns.get_loc(c) for c in charcols]
retloc = df.columns.get_loc(RETNAME)

K = len(charcols)
N = len(permno)
T = len(yrmo)

# --- Step 4: Map all panel rows to (firm, time) indices efficiently ---
df['n'] = df['permno'].map(permno_idx)
df['t'] = df['bigyrmo'].map(yrmo_idx)

# --- Step 5: Create arrays with vectorized assignment ---
char = np.full((N, T, K), np.nan)
ret = np.full((N, T), np.nan)

# Assign returns
ret[df['n'], df['t']] = df.iloc[:, retloc].astype(float).values

# Assign characteristics: each is a column, assign all at once
for k, col in enumerate(charcols):
    char[df['n'], df['t'], k] = df[col].astype(float).values

charnames = np.array(charcols)
# (Optional: for pandas-friendly tabular output, could also flatten as DataFrame)

# --- Step 6: Store in HDF5 and NPZ ---
# HDF5 (pandas): good for huge data, random access
# with pd.HDFStore(h5_path, 'w') as store:
#     store.put('ret', pd.DataFrame(ret, index=permno, columns=yrmo))
#     for i, cname in enumerate(charcols):
#         store.put(f'char_{cname}', pd.DataFrame(char[:, :, i], index=permno, columns=yrmo))
#     store.put('permno', pd.Series(permno))
#     store.put('yrmo', pd.Series(yrmo))
#     store.put('charnames', pd.Series(charnames))

# NPZ: easy, native for NumPy use
np.savez_compressed(
    npz_path,
    ret=ret,
    char=char,
    charnames=charnames,
    yrmo=yrmo,
    permno=permno
)

print(f"NPZ to {npz_path}. Shapes: ret {ret.shape}, char {char.shape}")
