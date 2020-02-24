'''
Ivan Christian

Homework 4 Problem 1

'''

'''
For testing purposes
'''
from utils.getimagenetclasses import test_parsesyn, testparse, test_parseclasslabel


'''
Importing the classes from the custom and starting the task
'''



from utils.name_pairs import extract_filename






if __name__=='__main__':
	filenames = extract_filename('imagenet2500', 'output.txt', '.JPEG')
	print(filenames)