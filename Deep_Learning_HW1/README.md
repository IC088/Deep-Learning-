# Documentation for HW1


## File Structure
structure of the project after the zip file is extracted should look something like the following:
```
├───{extracted folder name}
│    ├───dataset
│    │    └───dataset provided
│    ├───splits
│    │    ├───x_train.npy
│    │    ├───x_test.npy
│    │    ├───x_val.npy
│    │    ├───y_train.npy
│    │    ├───y_test.npy
│    │    └───y_val.npy
│    ├───accuracy
│    │    └───accFiles.txt
│    ├───2020_codinghw_w1.pdf
│    ├───Deep Learning HW1.pdf
│    ├───README.MD
│    └───hw1.py
```

The last result calculated will be in `accFile.txt`. Running `hw1.py` will overwrite the files in `splits` and `accuracy`
## How to run
To run the homework simply run the hw1.py file in the folder.


For Linux systems:
```
python3 hw.py
```
For Windows systems:
```
python hw.py
```

This will create the `accuracy` and `splits` folder which will contain the necessary files that is graded which will create the following folder structure:

```
├───{extracted folder name}
│    ├───dataset
│    │    └───dataset provided
│    ├───splits
│    │    ├───x_train.npy
│    │    ├───x_test.npy
│    │    ├───x_val.npy
│    │    ├───y_train.npy
│    │    ├───y_test.npy
│    │    └───y_val.npy
│    ├───accuracy
│    │    └───accFiles.txt
│    ├───2020_codinghw_w1.pdf
│    ├───Deep Learning HW1.pdf
│    ├───README.MD
│    └───hw1.py
```

## Question

What is the difference to a random 60−15−25 split of the whole data as compared to split class-wise? Why I asked you to split class-wise ? 
Explain in at most 5 sentences.

    Answer:
    Random splitting of the whole data will mix out the whole categories and allow for false positives,which are the correctly classified wrong seasons (the 0-0 pair in the prediction), to be undetected. 
    Class-wise splitting allows for the 'true' accuracy for each class to be taken into account without taking into account the false positives.
    
## Short Analysis

As of the last run of the program, the results are the following:


```
best c for spring = 0.01 with score: 0.19061707523245985
best c for summer = 0.01 with score: 0.8709932422966852
best c for autumn = 0.01 with score: 0.3924469195962409
best c for winter = 0.1 with score: 0.6543297678652708

Best c = 0.01
Average class wise accuracy tested against validation set = 0.5270967512476642
Overall average class wise accuracy against test set = 0.41015653233473326
vanilla accuracy against test set = 0.8866995073891626
```
It can be seen that the overall vanilla accuracy is higher than that of the average class wise accuracy. As the training dataset is arranged according to the y dataset stratification, the resulting accuracy will differ from iteration to iteration because of the difference in the randomisation of the y dataset stratification. This causes the vanilla accuracy to fluctuate in the 86-90% and the average class accuracy to fluctuate in the 36-42% range.