print('cnv_load.py: Beginning execution')

import numpy as np

def labels(filename):
    return np.loadtxt(filename, skiprows=1)

def tracking(filename):
    return np.loadtxt(filename, skiprows=13)

print('cnv_load.py: Completed execution')