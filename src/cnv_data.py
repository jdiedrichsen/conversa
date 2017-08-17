''' cnv_data - Deals with data loading for Conversa '''

# TODO: File documentation tags
# TODO: Add functionalities in cnv_test_lstm.py

import numpy as np

# Constants

# Tracking constants
_TR_N_SKIP_LINES = 12

# Label constants
_LA_FIELD_NAME_MIN = 'min'
_LA_FIELD_NAME_SEC = 'sec'
_LA_FIELD_NAME_FRAME = 'frame'
_LA_NON_BEHAV_FIELDS = {'pid', 'cam', _LA_FIELD_NAME_MIN, _LA_FIELD_NAME_SEC, _LA_FIELD_NAME_FRAME, 'ms', 'absoluteframe'}
_LA_FRAME_SHIFT = 0  # Describes the amount to label frames forward by - compensates for misalignments


def load_tracking(tracking_file):
    '''
    Loads tracking file data into a structured array
    :param tracking_file: The address of the tracking file
    :return: A structured array containing the information in the tracking file with field names
    '''
    try:
        return np.genfromtxt(tracking_file, dtype=float, skip_header=12, names=True)
    except IOError:
        print('Failed to open tracking file at ' + tracking_file)


# def load_labels(LABEL_FILE):
#     '''
#     TODO: Implement
#     :param LABEL_FILE:
#     :return:
#     '''
#     return


def load(tracking_file, label_file, behaviour_fields=None, structured=True):
    # TODO
    '''
    Loads data from a tracking file and a label file into structured arrays with corresponding entries 
    :param tracking_file: The address of the tracking file, see File Format Examples for an example of a tracking file
    :param label_file: The address of the label file, see File Format Examples  for an example of a label file
    :param behaviour_fields: A list of behaviours to include from the label file, leave as None if you want all
    behaviours included
    :param structured: Whether the returned numpy arrays are structured numpy array with fieldnames  
    :return: A 2 element tuple containing a strucutred array of the predictors and labels, as in (predictors, labels
    '''

    tracking_data, label_data = None, None  # Initialize before loading files

    # Load tracking data
    try:
        tracking_data = np.genfromtxt(tracking_file, dtype=float, skip_header=_TR_N_SKIP_LINES, names=True)
    except IOError:
        print('Failed to open tracking file at ' + tracking_file)

    # Load label data
    try:
        label_data = np.genfromtxt(label_file, dtype=float, names=True)
    except IOError:
        print('Failed to open label file at ' + label_file)

    if behaviour_fields is not None:  # Check if the label fields have been set
        keep_fields = _LA_NON_BEHAV_FIELDS | behaviour_fields  # Fields to keep
        for name in label_data.dtype.names:
            if name not in keep_fields:
                label_data = rm_field(label_data, name)

    # Get behaviour fields
    behaviour_fields = label_data.dtype.names
    behav_names = []
    for label in behaviour_fields:
        if not(label in _LA_NON_BEHAV_FIELDS):
            behav_names.append(label)
    n_samples = tracking_data.shape[0]
    n_behavs = len(behav_names)

    # Set labels
    behav_data = np.zeros(n_samples, dtype=[tuple((behav_names[i], '<f8')) for i in range(0, n_behavs)])
    label_length = label_data.shape[0]
    for behav_i in range(0, n_behavs):
        behav_name = behav_names[behav_i]
        # print('Applying behaviour label: ' + behav_name)
        behav_column = label_data[behav_name]
        curr_i = 0
        next_i = curr_i + 1
        curr_state = behav_column[curr_i]
        next_state = behav_column[next_i]
        while next_i + 1 < label_length:
            # Go to point of change and fill data from current point to point of change
            while (next_state == curr_state) and (next_i + 1 < label_length):
                next_i = next_i + 1
                next_state = behav_column[next_i]
            # Set data from beginning to point of change
            start_i = _LA_FRAME_SHIFT + index_to_frame(label_data, curr_i)
            end_i = _LA_FRAME_SHIFT + index_to_frame(label_data, next_i)
            # print('Setting ' + behav_name + ' label to ' + str(curr_state) + ' from indices ' + str(start_i)
            #       + ' to ' + str(end_i))
            behav_data[start_i:end_i][behav_i] = curr_state
            # Reset states
            # print('Resetting states')
            curr_i = next_i
            curr_state = behav_column[curr_i]

    # Add dimension to get 2D (required for Keras)
    tracking_data = add_dim(tracking_data)
    behav_data = add_dim(behav_data)

    # Destructure if needed
    if not structured:
        tracking_data = destructure(tracking_data)
        behav_data = destructure(behav_data)

    # Return data
    return tracking_data, behav_data


def time_to_frame(minute, second, frame, frame_rate=30):
    return int((minute * 60 + second) * frame_rate + frame)


def index_to_frame(data, index):
    return time_to_frame(
        data[_LA_FIELD_NAME_MIN][index],
        data[_LA_FIELD_NAME_SEC][index],
        data[_LA_FIELD_NAME_FRAME][index])


def destructure(data):
    '''
    Converts a structured array to a standard numpy array
    :param data: A structured array
    :return: A view of the ndarray with no fieldnames
    '''
    return data.view((data.dtype[0], len(data.dtype.names)))  # Assumes data type of data is homogeneous


def add_dim(data, n_dims=1):
    '''
    Adds a given number of dimensions to an ndarray
    :param data: The ndarray
    :param n_dims: The number of dimensions to add
    :return: The ndarray with added dimensions
    '''
    for i in range(0, n_dims):
        data = data[..., np.newaxis]
    return data


def to_seqs(data, seq_len, n_dims):
    '''
    Divides a numpy array into a series of sequences
    :param data: The numpy array to be divided into sequences
    :param seq_len: The length of sequences to produce
    :param n_dims: The number of dimensions for each member of each sequence to have
    :return: A numpy array which contains the original data divided into sequences of length seq_len. Data in the last 
    few rows may be cut off if it does not fill an entire sequence
    '''
    # seq_len = data.shape[0]
    # n_seqs = int(seq_len / seq_len)
    # n_dims = data.shape[1]
    # # Trim before reshaping
    # new_len = n_seqs * seq_len
    # data = data[:new_len]
    # # # Reshape into batches
    # data = np.reshape(data, (n_seqs, seq_len, n_dims))
    data_len = data.shape[0]
    n_seqs = int(data_len/seq_len)
    # data_dim = data.shape[1]
    return np.reshape(data[:(n_seqs*seq_len)], (n_seqs, seq_len, n_dims))


def rm_field(data, field_name):
    '''
    Removes a field from structutrecd
    :param data: 
    :param field_name: 
    :return: 
    '''
    names = list(data.dtype.names)
    if field_name in names:
        names.remove(field_name)
    return data[names]


print('Imported cnv_data')
