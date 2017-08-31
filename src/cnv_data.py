''' cnv_data - Deals with data loading for Conversa '''

import numpy as np
import pandas as pd

__author__ = 'Shayaan Syed Ali'
# __copyright__ = ''
__credits__ = ['Shayaan Syed Ali']
__maintainer__ = 'Shayaan Syed Ali'
__email__ = 'shayaan.syed.ali@gmail.com'
__status__ = 'Development'
# __license__ = ''
# __version__ = ''

# TODO: Add functions to write to file
# TODO: Add vprint and verbose flags

# Constants

# Tracking constants
_TR_N_SKIP_LINES = 12

# Label constants
_LA_FIELD_NAME_MIN = 'min'
_LA_FIELD_NAME_SEC = 'sec'
_LA_FIELD_NAME_FRAME = 'frame'
_LA_NON_BEHAV_FIELDS = ['pid', 'cam', _LA_FIELD_NAME_MIN, _LA_FIELD_NAME_SEC, _LA_FIELD_NAME_FRAME, 'ms', 'absoluteframe']
_LA_FRAME_SHIFT = 0  # Describes the amount to label frames forward by - compensates for misalignments


def load(tracking_file, label_file, behaviour_fields=None):
    '''
    Loads data from a tracking file and a label file into either a pandas DataFrame or numpy structured array
    :param tracking_file: The address of the tracking file, see File Format Examples for an example of a tracking file
    :param label_file: The address of the label file, see File Format Examples  for an example of a label file
    :param behaviour_fields: A list of behaviours to include from the label file, leave as None if you want al
    behaviours included
    :return: A 2 element tuple containing a pandas DataFrameof the predictors and labels, as in (predictors, labels) =
    cnv_data.load(...)
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
        keep_fields = _LA_NON_BEHAV_FIELDS + behaviour_fields  # Fields to keep
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
            behav_data[behav_name][start_i:end_i] = curr_state

            # Reset states
            curr_i = next_i
            curr_state = behav_column[curr_i]

    tracking_data = pd.DataFrame(tracking_data)
    behav_data = pd.DataFrame(behav_data)

    # Return data
    return tracking_data, behav_data

# TODO: Documentation
def load_subject(pid, cam,
                 tracking_dir='..\\data\\tracking\\',
                 tracking_file_suffix='.txt',
                 label_dir='..\\data\\labels\\',
                 label_file_suffix='.dat'):

    par_cam_str = ''.join(['par', str(pid), 'cam', str(cam)])
    cam_par_str = ''.join(['cam', str(cam), 'par', str(pid)])
    p_cam_str = ''.join(['p', str(pid), 'cam', str(cam)])

    # Note that the hierarchy of files is different between tracking and label files, this is hardcoded below
    # FUTURE_TODO: Find way to deal with different file hierarchy and implement

    tracking_file = ''.join([tracking_dir, par_cam_str, '\\', cam_par_str, tracking_file_suffix])
    label_file = ''.join([label_dir, p_cam_str, label_file_suffix])

    return load(tracking_file, label_file)


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


def to_subseqs(data, subseq_len):
    '''
    Divides a numpy array into a series of subsequences
    Data in the last few rows may be cut off if it does not fill an entire subsequence
    :param data: The numpy array to be divided into subsequences
    :param subseq_len: The length of sequences to produce
    :return: A numpy array which contains the original data divided into subsequences of length seq_len
    '''
    seq_len = data.shape[0]
    n_seqs = int(seq_len / subseq_len)
    data_dim = data.shape[-1]
    # Trim before reshaping into sequences
    new_len = n_seqs * subseq_len
    data = data[:new_len] # TODO: Add partial sequence at end instead of cutting, also add param partial_seqs=True for this behaviour
    # Reshape into sequences and return
    return np.reshape(data, (n_seqs, subseq_len, data_dim))


def rm_field(data, field_name):
    '''
    Removes a field from structured numpy array
    If the field is not in the array, the original array is returned
    :param data: The structured numpy array 
    :param field_name: A string of the field name to remove
    :return: The numpy array without the given field or the original array if the field is not found
    '''
    names = list(data.dtype.names)
    if field_name in names:
        names.remove(field_name)
    return data[names]


print('Imported cnv_data')
