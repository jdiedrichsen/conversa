import numpy as np

print('cnv_load.py: Beginning execution')

# Constants
frame_rate = 30


def tracking(filename):
    return np.genfromtxt(filename, dtype=float, skip_header=12, names=True)


def labels(filename, include_timestamps=False):
    pre_labels = np.genfromtxt(filename, dtype=float, names=True)
    length = pre_labels.shape[1]
    return(pre_labels)

# ADD: Can use approach in https://stackoverflow.com/questions/9171157/#9174892 to import into dict


def time_to_frame(min, sec, frame):
    return (min*60 + sec)*frame_rate + frame


print('cnv_load.py: Completed execution')
