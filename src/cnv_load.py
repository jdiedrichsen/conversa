import numpy as np

print('cnv_load.py: Beginning execution')

# Constants
_FIELD_NAME_MIN = 'min'
_FIELD_NAME_SEC = 'sec'
_FIELD_NAME_FRAME = 'frame'
_LABEL_NON_BEHAV_FIELDS = {'pid', 'cam', _FIELD_NAME_MIN, _FIELD_NAME_SEC, _FIELD_NAME_FRAME, 'ms', 'absoluteframe'}


def data(tracking_filename, label_filename):
    return labels(label_filename, tracking(tracking_filename))


def tracking(filename):
    try:
        return np.genfromtxt(filename, dtype=float, skip_header=12, names=True)
    except IOError:
        print('Failed to open tracking file at ' + filename)



def labels(label_file, tracking_data):

    # Load tracking data

    # Load label data
    try:
        label_data = np.genfromtxt(label_file, dtype=float, names=True)
    except IOError:
        print('Failed to open label file data at ' + label_file)

    # Get behaviour fields
    label_fields = label_data.dtype.names
    behav_names = []
    for label in label_fields:
        if not(label in _LABEL_NON_BEHAV_FIELDS):
            behav_names.append(label)
    n_samples = tracking_data.shape[0]
    n_behavs = len(behav_names)

    # Set labels
    behav_labels = np.zeros(n_samples, dtype=[tuple((behav_names[i], '<f8')) for i in range(0, n_behavs)])
    label_length = label_data.shape[0]
    for behav_i in range(0, n_behavs):
        behav_name = behav_names[behav_i]
        # print('Applying behaviour label: ' + behav_name)
        behav_data = label_data[behav_name]
        curr_i = 0
        next_i = curr_i + 1
        curr_state = behav_data[curr_i]
        next_state = behav_data[next_i]
        while next_i + 1 < label_length:
            # Go to point of change and fill data from current point to point of change
            while (next_state == curr_state) and (next_i + 1 < label_length):
                next_i = next_i + 1
                next_state = behav_data[next_i]
            # Set data from beginning to point of change
            start_i = index_to_frame(label_data, curr_i)
            end_i = index_to_frame(label_data, next_i)
            # print('Setting ' + behav_name + ' label to ' + str(curr_state) + ' from indices ' + str(start_i) + ' to ' + str(end_i))
            for j in range(start_i, end_i):
                behav_labels[j][behav_i] = curr_state
            # Reset states
            # print('Resetting states')
            curr_i = next_i
            curr_state = behav_data[curr_i]

    # Return
    # print('Returning labels')
    return tracking_data, behav_labels


def time_to_frame(minute, second, frame, frame_rate=30):
    return int((minute * 60 + second) * frame_rate + frame)


def index_to_frame(data, index):
    return time_to_frame(
        data[_FIELD_NAME_MIN][index],
        data[_FIELD_NAME_SEC][index],
        data[_FIELD_NAME_FRAME][index])


print('cnv_load.py: Completed execution')
