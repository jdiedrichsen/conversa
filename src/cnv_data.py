'''
cnv_data - Deals with data loading for Conversa
'''

import numpy as np

# Constants
_FIELD_NAME_MIN = 'min'
_FIELD_NAME_SEC = 'sec'
_FIELD_NAME_FRAME = 'frame'
_LABEL_NON_BEHAV_FIELDS = {'pid', 'cam', _FIELD_NAME_MIN, _FIELD_NAME_SEC, _FIELD_NAME_FRAME, 'ms', 'absoluteframe'}
_FRAME_SHIFT = 0  # Describes the amount to label frames forward by - compensates for misalignments


def load_tracking(tracking_file):
    '''
    Loads tracking file data into a structured array
    :param tracking_file: 
    :return: 
    '''
    try:
        return np.genfromtxt(tracking_file, dtype=float, skip_header=12, names=True)
    except IOError:
        print('Failed to open tracking file at ' + tracking_file)


# def load_labels(label_file):
#     '''
#     TODO: Implement
#     :param label_file:
#     :return:
#     '''
#     return


def load(tracking_file, label_file, join=False):
    '''
    Loads a tracking and label file
    :param tracking_file: The tracking file address
    :param label_file: The label file address
    :param join: Whether to join the data into one structured array
    :return: A structured array of tracking data and label data
    '''

    tracking_data, label_data = None, None  # Initialize before loading files

    # Load tracking data
    try:
        tracking_data = np.genfromtxt(tracking_file, dtype=float, skip_header=12, names=True)
    except IOError:
        print('Failed to open tracking file at ' + tracking_file)

    # Load label data
    try:
        label_data = np.genfromtxt(label_file, dtype=float, names=True)
    except IOError:
        print('Failed to open label file at ' + label_file)

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
            # print('Setting ' + behav_name + ' label to ' + str(curr_state) + ' from indices ' + str(start_i)
            #       + ' to ' + str(end_i))
            behav_labels[start_i:end_i][behav_i] = curr_state
            # Reset states
            # print('Resetting states')
            curr_i = next_i
            curr_state = behav_data[curr_i]

    # TODO: Add behaviour for join=True

    # Return data
    # print('Returning labels')
    return tracking_data, behav_labels


def time_to_frame(minute, second, frame, frame_rate=30):
    return _FRAME_SHIFT + int((minute * 60 + second) * frame_rate + frame)


def index_to_frame(data, index):
    return time_to_frame(
        data[_FIELD_NAME_MIN][index],
        data[_FIELD_NAME_SEC][index],
        data[_FIELD_NAME_FRAME][index])


print('Imported cnv_data')
