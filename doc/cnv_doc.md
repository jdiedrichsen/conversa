# Conversa

Example program
```python
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

## Data Loading - cnv_data

Loading data into ```predictors``` and ```labels```.
```
try:
    (predictors, labels) = (cnv_data.load(
        tracking_file, 
        label_file,
        behaviours))
except IOError:
    print('Failed to open files')
```

## Evaluation - cnv_eval

