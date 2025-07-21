import numpy as np
from scipy.io import loadmat
import h5py

# data = loadmat("/Users/chenyurui/Desktop/SIF/Data/BarillasShanken.mat")
# yrmo = data['yrmo']
# Mkt_RF = data['Mkt_RF']
# ME=data['ME']
# IA=data['IA']
# ROE=data['ROE']

data = h5py.File("/Users/chenyurui/Desktop/SIF/Data/M4.mat")

dates=np.array(data['dates'])
MKTRF=np.array(data['MKTRF'])
SMB=np.array(data['SMB'])
MGMT=np.array(data['MGMT'])
PERF=np.array(data['PERF'])
RF=np.array(data['RF'])


np.savez_compressed(
    'M4.npz',
    dates=dates,
    MKTRF=MKTRF,
    SMB=SMB,
    MGMT=MGMT,
    PERF=PERF,
    RF=RF
)