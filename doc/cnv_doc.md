# Conversa

## Data Loading and Handling - cnv_data

### Functions

**```load(tracking_file, label_file, label_fields=None, structured=True)```**  
    Loads data from a tracking file and a label file into structured arrays with corresponding entries  
    ```tracking_file```: The address of the tracking file, see File Format Examples for an example of a tracking file
    ```label_file```: The address of the label file, see File Format Examples  for an example of a label file
    ```behaviour_fields```: A list of behaviours to include from the label file, leave as None if you want all behaviour included  
    Returns a 2 element tuple containing a strucutred array of the predictors and labels, as in (predictors, labels)   

**```load_tracking(tracking_file)```**  
    

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

**```k_fold(predictors, labels, n_folds```**  
    Splits predictors and labels into a number of testing groups  
    ```predictors```: All of the predictors data to be split  
    ```labels```: All of the label data to be split  
    ```n_folds```: The number of folds to split the data into
    Each fold is a nested tuple, of ```(train_data, test_data)``` where ```train_data = (train_predictors, train_labels) and test_data = (test_predictors, test_labels)```  

### Usage

Using k_fold to partition the data into exclusive folds:
``` python
folds = cnv_eval.k_fold(predictors, labels, n_folds=5)  # This splits the data into 5 folds
for fold in folds:
    (train_data, test_data) = fold
    # Do something with the training and testing data
```
When using ```k_fold```, keep in mind that each elements in each fold may not keep their ordering. In order to use this function for sequence data, be sure to set each element of the predictor and label data to a sequence.

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
