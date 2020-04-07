# Documentation for HW6

## File Structure
```
├───{extracted folder name}
│    ├───data
│    │    ├───star_trek_transcripts_all_episodes.csv
│    │    └───star_trek_transcripts_all_episodes_f.csv
│    ├───loader
│    │    ├───custom_lstm.py
│    │    └───prep.py
│    ├───utils
│    │    └───vis.py
│    ├───results
│    │    ├───sample_text
│    │    │    ├───sample-text-output-1.txt
│    │    │    ├───...
│    │    │    └───sample-text-output-35.txt
│    │    ├───graph
│    │    │    ├───TestAccuracy.png
│    │    │    ├───TestLoss.png
│    │    │    └───TrainLoss.png
│    ├───main.py
```
## Instructions to Run

For Linux systems:
```
python3 main.py
```
For Windows systems:
```
python main.py
```

## Data Processing

The given function is used to processed the filtered text with some adjustments to cleaning:

- Instead of using ``","``, which is a common preposition used in text, as the delimiter, the text is processed to have `\t` as its delimiter. After some preprocessing (see `prep.py`), the text is ensured to not be missing in any alphabets/ numbers/ allowed prepositions. 
- An addition of the string "|" is given as the EOS sentence.
## Results

Accuracy per epoch
```
Starting Epoch : 1
Test Accuracy : 0.39210940578203957

Starting Epoch : 2
Test Accuracy : 0.49036605745595924

Starting Epoch : 3
Test Accuracy : 0.4996506063926332

Starting Epoch : 4
Test Accuracy : 0.5195196631030252

Starting Epoch : 5
Test Accuracy : 0.5086650242576285

Starting Epoch : 6
Test Accuracy : 0.524587667817992

Starting Epoch : 7
Test Accuracy : 0.5489366897066278

Starting Epoch : 8
Test Accuracy : 0.5575256378932081

Starting Epoch : 9
Test Accuracy : 0.5714175647626262

Starting Epoch : 10
Test Accuracy : 0.5580377993417815

Starting Epoch : 11
Test Accuracy : 0.5782556883720014

Starting Epoch : 12
Test Accuracy : 0.5778761028292185

Starting Epoch : 13
Test Accuracy : 0.5786950885147714

Starting Epoch : 14
Test Accuracy : 0.5876321630237409

Starting Epoch : 15
Test Accuracy : 0.5675378931688537

Starting Epoch : 16
Test Accuracy : 0.6073840229495528

Starting Epoch : 17
Test Accuracy : 0.5878082625092417

Starting Epoch : 18
Test Accuracy : 0.5838952797156993

Starting Epoch : 19
Test Accuracy : 0.5805670354758267

Starting Epoch : 20
Test Accuracy : 0.5812426485664658

Starting Epoch : 21
Test Accuracy : 0.5813332617877072

Starting Epoch : 22
Test Accuracy : 0.5977881045881934

Starting Epoch : 23
Test Accuracy : 0.5812383386151454

Starting Epoch : 24
Test Accuracy : 0.5862659332363951

Starting Epoch : 25
Test Accuracy : 0.5949206157271729

Starting Epoch : 26
Test Accuracy : 0.5875192603984405

Starting Epoch : 27
Test Accuracy : 0.5769227114991772

Starting Epoch : 28
Test Accuracy : 0.5885013246000939

Starting Epoch : 29
Test Accuracy : 0.5969872223233776

Starting Epoch : 30
Test Accuracy : 0.5976645959985232

Starting Epoch : 31
Test Accuracy : 0.6002652433491003

Starting Epoch : 32
Test Accuracy : 0.5921358504383203

Starting Epoch : 33
Test Accuracy : 0.590928831753664

Starting Epoch : 34
Test Accuracy : 0.6029589650046756

Starting Epoch : 35
Test Accuracy : 0.5984938458292246

Starting Epoch : 36
Test Accuracy : 0.6063665718081358

Starting Epoch : 37
Test Accuracy : 0.6039227308497033

Starting Epoch : 38
Test Accuracy : 0.6061413681374184

Starting epoch 39
Test Accuracy : 0.5985873135876567

Starting epoch 40
Test Accuracy : 0.5994089315883989
```


Sample Text Generated at epoch = 1
```
SULU: No the gist. 
SULU: Ome. 
SULU: Soelm. 
KIRK: Nirg the se tho croe. 
APOOY: I poart, Castor. 
```

Sample Text Generated at epoch = 40

```
SPOCK: You report and coming and we think it is the Captain. 
SPOCK: I cannot give the sealilate, you're be aster them. 
SPOCK: You want to get them. 
SPOCK: All right, I'm see. What is it? 
KIRK: Anything to be all probact. What is it? 
```


Top-5 (Weird) Quotes generated:

```
SPOCK: You want to get them. (found in sample-output-40.txt)
KIRK: Oh, Captain, Captain. (found in sample-output-39.txt)
MCCOY: I'm something. (found in sample-output-37.txt)
MCCOY: Mister Spock? (found in sample-output-37.txt)
ELDER: Now same me. (found in sample-output-31.txt)
```


