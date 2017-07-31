import numpy as np

print('cnv_load.py: Beginning execution')

# Constants
_FIELD_NAME_MIN = 'min'
_FIELD_NAME_SEC = 'sec'
_FIELD_NAME_FRAME = 'frame'

def data(tracking_filename, label_filename):
    return apply_labels(label_filename, tracking(tracking_filename))


def tracking(filename):
    try:
        return np.genfromtxt(filename, dtype=float, skip_header=12, names=True)
    except IOError:
        print('Failed to open tracking file at ' + filename)


def apply_labels(filename, tracking_data):
    # Alternatively, can import into dict via approach at https://stackoverflow.com/questions/9171157/#9174892
    try:
        labels = np.genfromtxt(filename, dtype=float, names=True)
    except IOError:
        print('Failed to open label file at ' + filename)
    mins = labels[_FIELD_NAME_MIN]
    secs = labels[_FIELD_NAME_SEC]
    frames = labels[_FIELD_NAME_FRAME]
    n_names = len(labels.dtype.names)
    # n_names = labels.dtype.names.shape[0]
    print (n_names)
    # for name_i in range(0, n_names):
    #     print(name_i)
    # return labels


def time_to_frame(min, sec, frame, frame_rate=30):
    return (min*60 + sec)*frame_rate + frame


print('cnv_load.py: Completed execution')
