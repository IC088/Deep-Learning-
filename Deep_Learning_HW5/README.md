# Documentation for HW5 TASK 1 & 2

## File Structure
```
├───{extracted folder name}
│    ├───data
│    │    ├───names
│    │    └───eng-fra.txt
│    ├───prep
│    │    ├───custom_lstm.py
│    │    ├───custom_gru.py
│    │    └───data_preprocessing.py
│    ├───vis
│    │    └───vis.py
│    ├───results
│    │    ├───lstm_1 (task 1)
│    │    ├───gru_1 (task 1)
│    │    ├───test_confusion_matrix (task 1 and 2)
│    │    └───lstm_2 (task 2)
│    ├───dl_hw5.py
```
## Instructions to Run

For Linux systems:
```
python3 dl_hw5.py
```
For Windows systems:
```
python dl_hw5.py
```
## Results
Note: Accuracy is already percentage
### TASK 1A results
- LSTM module for task 1
- learning rate = 0.1
- number of epochs = 5

```
n_layer = 1 
hidden size = 200

Train Loss     : [1.227332592010498, 0.923230767250061, 0.7824930548667908, 0.6861002445220947, 0.609287679195404]
Val Loss       : [0.9993002414703369, 0.8412607312202454, 0.7841787934303284, 0.7109963297843933, 0.6849240660667419]
Val Accuracy   : [68.32669322709162, 73.30677290836654, 75.54780876494024, 77.93824701195219, 78.98406374501991]
Test Loss      : 0.6599024534225464
Test Accuracy  : 79.7808764940239
```

```
n_layer = 1 
hidden size = 500

Train Loss     : [1.246099829673767, 0.969415009021759, 0.8317446708679199, 0.7318781018257141, 0.6526939272880554]
Val Loss       : [1.0254466533660889, 0.8806233406066895, 0.7979020476341248, 0.7279176115989685, 0.6880820989608765]
Val Accuracy   : [67.2808764940239, 73.2569721115538, 74.9003984063745, 77.4402390438247, 78.93426294820716]
Test Loss      : 0.6775903105735779
Test Accuracy  : 78.88446215139442
```
```
n_layer = 2 
hidden size = 200

Train Loss     : [1.3845361471176147, 1.000089406967163, 0.8405395150184631, 0.7273303866386414, 0.6444365382194519]
Val Loss       : [1.0865888595581055, 0.9140353798866272, 0.8374977707862854, 0.7060388922691345, 0.6851726174354553]
Val Accuracy   : [66.93227091633466, 72.16135458167331, 74.35258964143426, 78.78486055776892, 78.98406374501991]
Test Loss      : 0.6850999593734741
Test Accuracy  : 79.38247011952191
```

```
n_layer = 2 
hidden size = 500

Train Loss     : [1.3988966941833496, 1.0480680465698242, 0.9085752964019775, 0.7946827411651611, 0.7043721079826355]
Val Loss       : [1.0859586000442505, 0.9680438041687012, 0.8731743693351746, 0.786329448223114, 0.7188457250595093]
Val Accuracy   : [67.43027888446214, 69.97011952191235, 73.35657370517929, 76.09561752988047, 77.49003984063745]
Test Loss      : 0.6975340843200684
Test Accuracy  : 78.23705179282868
```

### TASK 1B results
- GRU module for task 1
- learning rate = 0.1
- number of epochs = 5
```
n_layer = 1 
hidden size = 200


Train Loss      : [1.163597583770752, 0.8449446558952332, 0.7082198262214661, 0.627559244632721, 0.5676727294921875]
Val Loss        : [0.9191424250602722, 0.7596294283866882, 0.6706835031509399, 0.6754388213157654, 0.7014129161834717]
Val Accuracy    : [71.21513944223108, 75.54780876494024, 78.98406374501991, 79.2828685258964, 79.33266932270917]
Test Loss       : 0.6982545852661133
Test Accuracy   : 79.8804780876494
```


### TASK 2 results
The defaulted setting:
- number of layers  = 2 
- hidden size = 500
- learning rate = 0.1
- number of epochs = 5
- Network used = LSTM

```
Batch size = 1

Train Loss     : [1.3952969312667847, 1.0465350151062012, 0.9101250767707825, 0.8020501136779785, 0.7149381637573242]
Val Loss       : [1.0843290090560913, 0.9646908044815063, 0.8927043676376343, 0.8026851415634155, 0.6923583745956421]
Val Accuracy   : [68.57569721115537, 70.2191235059761, 73.40637450199203, 76.09561752988047, 78.83466135458167]
Test Loss      : 0.6777627468109131
Test Accuracy  : 78.48605577689243
```

```
Batch size = 10

Train Loss     : [1.7424702644348145, 1.4949227571487427, 1.3174715042114258, 1.2133417129516602, 1.1044772863388062]
Val Loss       : [1.6378519535064697, 1.3401843309402466, 1.2433338165283203, 1.153752088546753, 1.0485551357269287]
Val Accuracy   : [51.55, 60.6, 63.6, 66.85, 69.19999999999999]
Test Loss      : 1.053547978401184
Test Accuracy  : 68.05
```

```
Batch size = 50

Train Loss     : [1.889815092086792, 1.7371599674224854, 1.7028112411499023, 1.6639549732208252, 1.6086490154266357]
Val Loss       : [1.7740343809127808, 1.7239850759506226, 1.6979023218154907, 1.6451936960220337, 1.578213095664978]
Val Accuracy   : [46.9, 49.9, 49.2, 51.2, 53.300000000000004]
Test Loss      : 1.5704189538955688
Test Accuracy  : 54.949999999999996
```


## vis folder

`vis.py` contains the visualisation tools for visualisation purposes, mainly using matplotlib

## prep folder

The `prep` folder is the folder containing hte utilities regarding the model architecture and preprocessing procedures.

`prep` folder contains the data preprocessing code in the `data_preprocessing.py` 

`prep` folder also contains the custom made GRU and LSTM modules in `custom_gru.py` and `custom_lstm.py` respectively. 

## results folder

`results` folder contain the figures as well as the text file containing the scores from the tasks.

For the file naming format, the figures are named in the following format:

- {n_type}_{name}_graph.png where `n_type` is either `lstm` or `gru` and `name`is the name of either test loss/train loss/ train accuracy. This is for the graph of train loss, test loss, and test accuracy for task 2.


- Format for the confusion matrix figure is {n_type}_{n_layers}_{h_size}_{batch_size}_{task}_test_confusion_matrix.png where `n_type` is either `lstm` or `gru`, `n_layers` is the number of layers, `h_size` is the hidden size, `batch_size` is the batch size, and `task` is either 1 or 2 corresponding to the task.

- results_{n_type}_{n_layers}_{h_size}_{batch_size}.txt where `n_type` is either `lstm` or `gru`, `n_layers` is the number of layers, `h_size` is the hidden size, and `batch_size` is the batch size. This is to save the accuracy and loss for train, val, test for both tasks. This can be found in the respective `{n_type}_{task}` folder where `task` is either 1 or 2 and `n_type` is either `lstm` or `gru`.