# Documentation for HW2

## File Structure
```
├───{extracted folder name}
│    ├───data.npy
│    ├───hw2.py
│    ├───hw2-task1.py
│    ├───hw2-task2.py
│    ├───figure-2.png
│    ├───figure-5.png
│    └───figure-8.png
```
To run Task 1 only, run `hw2-task1.py`, this would print out the timing, the results and the shape of the results that results for the results. If you have CUDA enabled, the timing for the results will be shown.

To run Task 2 only, run `hw2-task2.py`, this would plot out the cluster based on the saved `data.npy`. Figure-k where k = 2,5,or 8, is the saved plot of the plot.To change the values of k, please change teh variable `k` in this file.

To run both Task 1 and Task 2, run `hw2.py`, this would give the combination of the results from both `hw2-task1.py`, `hw2-task2.py`.


## How to run
To run the homework simply run the hw1.py file in the folder.


For Linux systems (example):
```
python3 hw2.py
```
For Windows systems (example):
```
python hw2.py
```