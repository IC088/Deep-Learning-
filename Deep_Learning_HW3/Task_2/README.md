# Documentation for HW3 TASK 2

## File Structure
```
├───{extracted folder name}
│    ├───data
│    │    └───FASHIONMNIST
│    ├───models
│    │    └───model.py
│    ├───utils
│    │    ├───data_loaders.py
│    │    └───vis.py
│    ├───Acc-graph.png
│    ├───loss-graph.png
│    ├───task_2.py
│    ├───hw2-task2.py
```
## Question
When you train a deep neural net, then you get after every epoch one model (actually after every minibatch). Why you should not select the best model over all epochs on the test dataset?


## Answer

To answer the question, the reason why is to prevent overfitting and doing very well only on the test set


## Instructions to Run

For Linux systems:
```
python3 task_2.py
```
For Windows systems:
```
python task_2.py
```


## utils folder

utils folder contain `data_loaders.py`. This contains the initialisation of the dataset and the train-val splitting as well as the test set initialisation

`data_loaders.py` also contains `test_accuracy` where it is used to check the model performance after training.


`vis.py` contains the visualisation tools for visualisation purposes, mainly using matplotlib

## models folder

models folder contains `model.py` which contains the initialisation for the models. For this homework, 2 models are initialised, `NN` and `Net`. NN is the fully-connected layer model with 300, 100, 10 neuron for the layers respectively. Net is the model using `Conv2D`