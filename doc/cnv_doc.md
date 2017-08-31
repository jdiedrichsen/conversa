# Conversa Documentation

## Data Loading and Handling - cnv_data

### Functions

**```load(tracking_file, label_file, behaviour_fields=None)```**  
    Loads data from a tracking file and a label file into structured arrays with corresponding entries  
    Parameters:  
        ```tracking_file```: The address of the tracking file, see File Format Examples for an example of a tracking file  
        ```label_file```: The address of the label file, see File Format Examples  for an example of a label file  
        ```behaviour_fields```: A list of behaviours to include from the label file, leave as None if you want all behaviours included  
    Returns a 2 element tuple containing numpy arrays of the predictors and labels, as in ```(predictors, labels) = cnv_data.load(...)```  
    
**```destructure(data)```**  
    Converts a structured array to a standard numpy array  
    Parameters:  
        ```data```: A structured array  
    Returns a view of the ndarray with no fieldnames  
    
**```add_dim(data, n_dims=1)```**
    Adds a given number of dimensions to an ndarray, useful when a model requires higher dimensional input  
    Dimensions are added such that an ndarray of shape (10, 3) would be returned with shape (10, 3, 1) if ```n_dims=1```  
    Also see [```numpy.reshape```](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)  
    Parameters:  
        ```data```: The ndarray  
        ```n_dims```: The number of dimensions to add  
    Returns the ndarray with added dimensions  
    
**```to_subseqs(data, seq_len, n_dims)```**  
    Divides a numpy array into a series of subsequences  
    Data in the last few rows may be cut off if it does not fill an entire subsequence  
    Parameters:  
        ```data```: The numpy array to be divided into subsequences  
        ```seq_len```: The length of sequences to produce  
        ```n_dims```: The number of dimensions for each member of each sequence to have  
    Returns a numpy array which contains the original data divided into subsequences of length seq_len  
    
**```rm_field(data, field_name)```**  
    Removes a field from structured numpy array  
    If the field is not in the array, the original array is returned  
    Parameters:  
        ```data```: The structured numpy array  
        ```field_name```: A string of the field name to remove  
    Returns the numpy array without the given field or the original array if the field is not found  

### Usage

Loading all predictor and label behaviours into DataFrames:  
``` python
tracking_file = '..\\data\\tracking\\par2024Cam1\\cam1par2024.txt'
label_file = '..\\data\\labels\\p2024cam1.dat'
try:
    cnv_data.load(tracking_file, label_file)
except IOError:
    print('Failed to open tracking and label files')
```

Loading data into ```predictors``` and *some* labels into ```labels```, then removing a label:  
``` python
behaviours = ['smile', 'talk']  # We'll only load the smile and talk behaviours
try:
    (predictors, labels) = cnv_data.load(
        tracking_file,  # A string containing the location of the kinematic tracking data
        label_file,  # A string containing the location of the label data
        behaviours)
except IOError:
    print('Failed to open files')

# Remove the smile label
labels = cnv_data.rm_field(labels, 'smile')
```

For examples of what tracking and label data should look like, see the File Format Examples section.

## Evaluation - cnv_eval

### Functions  

**```accuracy(predicted, true, rounding=True)```**  
    Determines the accuracy of a predicted value against an actual value for values in the range \[0, 1]  
    Requires that the predicted and true values are numpy arrays (or of classes that work with numpy functions) and that they are of the same shape  
    Parameters:  
        ```predicted```: The predicted value(s) as a numpy array, same shape as true  
        ```true```: The actual value(s) as a numpy array, same shape as predicted  
        ```rounding```: Whether to round predicted values or not, defaults to True  
        Returns the accuracy of the prediction against the true value  
    
**```eval_models(models, predictors, labels, verbose=0)```**  
    Evaluates models given predictor and label data to train and test the models on  
    Parameters:  
        ```models```: The models to evaluate  
        ```predictors```: Predictors to test the models on  
        ```labels```: Labels to test the models on   
        ```verbose```: The verbosity level of model training and testing - note that model console output often conflicts with outputs from cnv_eval - defaults to 0 (not verbose)  
    Returns a pandas DataFrame with columns fold_no, model_no, and accuracy  
    
**```order_fields(df, priority_fields)```**  
    Re-orders the columns of a pandas DataFrame according to column_names  
    Parameters:  
        ```df```: The DataFrame whose columns are to be reordered  
        ```priority_fields```: The fields to bring to the left in order, does not need to include all columns - others will be added at the back  
    Returns the DataFrame with reordered columns  
    
**```eval_models_on_subjects(models, subjects, behaviours=None, n_folds=5, verbose=1)```**  
    Runs evaluation for a list of models on a list of subjects  
    Parameters:  
        ```models```: Model objects, should implement Model abstract base class from cnv_model  
        ```subjects```: A tuple of the form (pid, cam), where pid and cam denote the pid number and cameras number respectively, like (2024, 2)  
        ```behaviours```: Behaviours to train on, leave as None for training on all behaviour separately  
        ```n_folds```: The number of folds for the k-folds cross validation algorithm  
        ```verbose```: How much debugging information is given, higher numbers giv more info, zero is the minimum and gives only errors
    Returns a pandas DataFrame summarizing all the results  
    
**```summary(eval_results):```**  
    Returns a summarized version of model evaluations which averages the accuracy of models across folds  
    Parameters:  
        ```eval_results```: The DataFrame to summarize  
    Returns a summary DataFrame  

### Usage



## Example Programs

Load some data into pandas DataFrames:

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
behaviours = ['talk']  # We'll only be looking at the talk behaviour
try:
    (predictors, labels) = (cnv_data.load(tracking_file, label_file, behaviours))
except IOError:
    print('Failed to open files')

# Run an SVM on predictors and labels, printing some of the data:

# Load the SVM model
try:
    from cnv_model import SVMModel
except ImportError:
    print('Unable to import from cnv_model')

results = cnv_eval.eval_models([SVMModel()], predictors, labels, 
                               verbose=1)  # If we want to suppress output we can set this to 0

# Now our SVM will train on the data

# We have the evaluation results in the results DataFrame, we can print these out in a table
print(cnv_eval.summary())

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
