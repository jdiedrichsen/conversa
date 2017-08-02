import numpy as np

print('cnv_load.py: Beginning execution')

# Constants
_FIELD_NAME_MIN = 'min'
_FIELD_NAME_SEC = 'sec'
_FIELD_NAME_FRAME = 'frame'
_LABEL_NON_BEHAV_FIELDS = {'pid', 'cam', _FIELD_NAME_MIN, _FIELD_NAME_SEC, _FIELD_NAME_FRAME, 'ms', 'absoluteframe'}


def data(tracking_filename, label_filename):
    return apply_labels(label_filename, tracking(tracking_filename))


def tracking(filename):
    try:
        # # Get file data
        # tr = np.genfromtxt(filename, dtype=float, skip_header=12, names=True)
        # # Get header
        # fh = open(filename, 'r')
        # for i in range(0, 12):
        #     fh.readline()
        # header = fh.readline().rstrip('\n').split('\t')
        # fh.close()
        # tr_2 = np.recarray(tr.shape, dtype=float, names=tuple(header))
        # n_fields = len(header)
        # for i in range(0, n_fields):
        #     print(tr_2)
        #     tr_2[i] = tr[header[i]];
        return np.genfromtxt(filename, dtype=float, skip_header=12, names=True)
    except IOError:
        print('Failed to open tracking file at ' + filename)


def apply_labels(filename, tracking_data):
    # Alternatively, can import into dict via approach at https://stackoverflow.com/questions/9171157/#9174892
    try:
        label_data = np.genfromtxt(filename, dtype=float, names=True)
    except IOError:
        print('Failed to open label file at ' + filename)
    # mins = label_data[_FIELD_NAME_MIN]
    # secs = label_data[_FIELD_NAME_SEC]
    # frames = label_data[_FIELD_NAME_FRAME]
    # Get behaviour fields
    label_fields = label_data.dtype.names
    n_label_fields = len(label_fields)
    behaviour_names = []
    for name_i in range(0, n_label_fields):
        label = label_fields[name_i]
        if not(label in _LABEL_NON_BEHAV_FIELDS):
            behaviour_names.append(label)
    n_samples = tracking_data.shape[0]
    n_behavs = len(behaviour_names)
    behav_labels = np.recarray((n_samples, n_behavs), dtype=float, names=tuple(behaviour_names))
    print(tracking_data.shape)
    print(behav_labels.shape)
    # Fill labels
    label_length = label_data.shape[0]
    for behav_i in range(0, n_behavs):
        behav_name = behaviour_names[behav_i]
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
            start_i = time_to_frame(
                label_data[_FIELD_NAME_MIN][curr_i],
                label_data[_FIELD_NAME_SEC][curr_i],
                label_data[_FIELD_NAME_FRAME][curr_i])
            end_i = time_to_frame(
                label_data[_FIELD_NAME_MIN][next_i],
                label_data[_FIELD_NAME_SEC][next_i],
                label_data[_FIELD_NAME_FRAME][next_i])
            # print('Setting ' + behav_name + ' label to ' + str(curr_state) + ' from indices ' + str(start_i) + ' to ' + str(end_i))
            for j in range(start_i, end_i):
                behav_labels[j][behav_i] = curr_state
            # Reset states
            # print('Resetting states')
            curr_i = next_i
            curr_state = behav_data[curr_i]
    print('Applied labels')
    return np.hstack((tracking_data, behav_labels))


def time_to_frame(min, sec, frame, frame_rate=30):
    return int((min*60 + sec)*frame_rate + frame)


print('cnv_load.py: Completed execution')
