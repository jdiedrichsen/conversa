# Conversa

## Data Loading and Handling - cnv_data

### Functions

**```load(tracking_file, label_file, behaviour_fields=None, structured=True)```**  
    Loads data from a tracking file and a label file into structured arrays with corresponding entries  
    Parameters:  
        ```tracking_file```: The address of the tracking file, see File Format Examples for an example of a tracking file  
        ```label_file```: The address of the label file, see File Format Examples  for an example of a label file  
        ```behaviour_fields```: A list of behaviours to include from the label file, leave as None if you want all behaviours included  
        ```structured```: Whether the returned numpy arrays are structured numpy array with fieldnames, see cnv_data.destructure for converting a structured array to a standard ndarray  
    Returns a 2 element tuple containing numpy arrays of the predictors and labels, as in ```(predictors, labels) = cnv_data.load(...)```  

**```load_tracking(tracking_file)```**  
    Loads tracking file data into a structured array  
    Parameters:  
        ```tracking_file```: The address of the tracking file  
    Returns structured array containing the information in the tracking file with field names  
    
**```destructure(data)```**  
    Converts a structured array to a standard numpy array  
    Parameters:  
        ```data```: A structured array  
    Returns a view of the ndarray with no fieldnames  
    
**```add_dim(data, n_dims=1)```**
    Adds a given number of dimensions to an ndarray, useful when a neural network layer requires higher dimensional input  
    Dimensions are added such that an ndarray of shape (10, 3) would be returned with shape (10, 3, 1) if ```n_dims=1```  
    Parameters:  
        ```data```: The ndarray  
        ```n_dims```: The number of dimensions to add  
    Returns the ndarray with added dimensions  
    
**```to_seqs(data, seq_len, n_dims)```***  
    ***Work in progress, not fully functional***  
    Divides a numpy array into a series of sequences  
    Data in the last sequence may be cut off if the ndarray ```data``` does not have a number of rows which is divisile by ```seq_len```  
    Parameters:  
        ```data```: The numpy array to be divided into sequences  
        ```seq_len```: The length of sequences to produce  
        ```n_dims```: The number of dimensions for each member of each sequence to have  
    Returns a numpy array which contains the original data divided into sequences of length seq_len  
    
**```rm_field(data, field_name)```**
    Removes a field from structured numpy array
    If the field is not in the array, the original array is returned
    Parameters:
        ```data```: The structured numpy array 
        ```field_name```: A string of the field name to remove
    Returns the numpy array without the given field or the original array if the field is not found

### Usage

Loading data into ```predictors``` and ```labels```:  
``` python
try:
    (predictors, labels) = (cnv_data.load(
        tracking_file,  # A string containing the location of the kinematic tracking data
        label_file,  # A string containing the location of the label data
        behaviours))
except IOError:
    print('Failed to open files')
```

For examples of what tracking and label data should look like, see the File Format Examples section.

## Evaluation - cnv_eval

### Functions

**```k_fold(predictors, labels, n_folds)```**  
    Splits predictors and labels into a number of testing groups  
    Parameters:
        ```predictors```: All of the predictors data to be split  
        ```labels```: All of the label data to be split  
        ```n_folds```: The number of folds to split the data into  
    Returns an array of fold where each fold is a nested tuple, of ```(train_data, test_data)``` where ```train_data = (train_predictors, train_labels) and test_data = (test_predictors, test_labels)```  

### Usage

Using k_fold to partition the data into exclusive folds:
``` python
folds = cnv_eval.k_fold(predictors, labels, n_folds=5)  # This splits the data into 5 folds
for fold in folds:
    (train_data, test_data) = fold
    # Do something with the training and testing data
```
When using ```k_fold```, keep in mind that each elements in each fold may not keep their ordering. In order to use this function for sequence data, be sure to set each element of the predictor and label data to a sequence using ```cnv_data.to_seqs``` or ```numpy.reshape```.

## Example Programs

Load some data into numpy arrays:
``` python
# Load data and evaluation modules
try:
    import cnv_data, cnv_eval
except ImportError:
    print('Unable to import cnv_data, cnv_eval')

# Load predictor and label data
# Predictors are kinematic data, labels are behaviour data
tracking_file = '..\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
label_file = '..\\data\\labels\\p2024cam1.dat'

predictors, labels = None, None
behaviours = {'smile'}  # We'll only be looking at the smile behaviour
try:
    (predictors, labels) = (cnv_data.load(tracking_file, label_file, behaviours))
except IOError:
    print('Failed to open files')
```

Partition existing data in numpy arrays ```predictors``` and ```labels``` using ```k_fold``` and printing some of the data:
``` python
n_folds = 5
folds = cnv_eval.k_fold(predictors, labels, n_folds=5)
for (train_data, test_data) in folds:
    (train_predictors, train_labels) = train_data
    (test_predictors, test_labels) = test_data
    for i in range(0, 3):
        print('Test data:\n' + str(test_predictors[i]))
        for j in range(0, n_folds - 1):
            print('Train data:\n' + str((train_predictors[i * (n_folds - 1) + j])))
```

## File Format Examples

These files are given as tab-delimited files.  Note that the behaviour files typically have a header containing metadata which is ignored by ```cnv_data```.

### Behaviour file

| timestamp | isTracked | bodyId | neckPosX | neckPosY | neckPosZ | ... | Jaw_Open |
|-----------|-----------|--------|----------|----------|----------|-----|----------|
| 0         | 1         | 2      | 1.59916  | -1.63241 | -79.9777 | ... | 0        |
| 0.03333   | 1         | 2      | 1.59916  | -1.63241 | -79.9777 | ... | 0        |
| 0.06667   | 1         | 2      | 1.59622  | -1.63241 | -79.9777 | ... | 0        |
| 0.1       | 1         | 2      | 1.58414  | -1.63241 | -79.9777 | ... | 0        |
| 0.13333   | 1         | 2      | 1.5711   | -1.63712 | -79.9777 | ... | 0        |
| ...       | ...       | ...    | ...      | ...      | ...      | ... | ...      |
| 328.2     | 1         | 2      | 2.26969  | -2.58357 | -77.6746 | ... | 1.77907  |

### Label file

| pid  | cam | min | sec | frame | absoluteframe | smile | talk | laugh |
|------|-----|-----|-----|-------|---------------|-------|------|-------|
| 2024 | 1   | 0   | 0   | 0     | 1             | 0     | 0    | 0     |
| 2024 | 1   | 0   | 1   | 19    | 50            | 1     | 0    | 0     |
| 2024 | 1   | 0   | 2   | 6     | 67            | 0     | 0    | 0     |
| 2024 | 1   | 0   | 5   | 13    | 164           | 0     | 1    | 0     |
| 2024 | 1   | 0   | 5   | 18    | 169           | 1     | 1    | 0     |
| ...  | ... | ... | ... | ...   | ...           | ...   | ...  | ...   |
| 2024 | 1   | 5   | 27  | 0     | 9811          | 0     | 0    | 0     |
