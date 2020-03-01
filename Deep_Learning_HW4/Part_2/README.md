# Documentation for HW4 TASK 2

## Instructions to Run

For Linux systems:
```
python3 hw4_task2.py
```
For Windows systems:
```
python hw4_task2.py
```


## README


For `mode = 1`, which is training from scratch the test accuracy is observed to be lower than `mode = 2` and `mode = 3`, which uses the pretrained weights.

The comparison for mode 1
```V
Validation Accuracy Mean: 0.5587286063569682`
Test Accuracy: 0.6656497864551556
```

The comparison for mode 2
```V
Validation Accuracy Mean: 0.8956295843520783
Test Accuracy: 0.9243441122635754
```

The comparison for mode 3
```V
Validation Accuracy Mean: 0.874541564792176
Test Accuracy: 0.9212934716290421
```


As seen, training from scratch gives a significantly smaller accuracy than using the pretrained model. Training the last 2 layers and using the pretrained weights give similar test and validation accuracy. As such, rather than training from scratch, it is generally better to do transfer learning since the model is already trained with thousands/millions of parameters.

To see the accuracy, please refer to the files accuracy{mode number}.md

