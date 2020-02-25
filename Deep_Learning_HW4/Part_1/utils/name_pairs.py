'''
Create list of filenames of the images and save it to a .csv file
'''
'''
Utilities function
'''

import os
import pandas as pd
from utils.getimagenetclasses import parsesynsetwords, parseclasslabel

'''
_get_label is a helper function to get labels based on the file names

Input :
- image_path : string (filename of the xml file)
- output_folder : string (folder in which the xml files are contained in)

Output :
label : string (label of the image)
'''



def _get_label(image_path, output_folder):
	image_path = image_path.split('.')[0]
	filen = 'synset_words.txt'

	nm = os.path.join(output_folder, image_path + '.xml')
	
	indicestosynsets,synsetstoindices,synsetstoclassdescr = parsesynsetwords(filen)
	
	label = parseclasslabel(nm, synsetstoindices)

	return label
'''
Function extract_filename is a helperr function to extract the 

Input:
- input_folder : string (.txt file containing the list of all the xml file names)
- output_folder : string (folder in which the xml files are contained in)
- output : string ( output file name for the dataframe)

OUtput:
- String confirmation of the data 

'''
def extract_filename(input_file, output_folder, output):
	xml_paths = []
	pairs = dict()
	with open(input_file) as f:
		for line in f.readlines():
			xml_paths.append(line.strip('\n'))
	for val in xml_paths:
		labels = _get_label(val, output_folder)
		pairs[val] = labels
	df = pd.DataFrame.from_dict(pairs, orient='index')
	# new_header = df.iloc[0]
	# df = df[1:]
	# df.columns = new_header
	df.to_csv(output)

	return f'Files saved in {output}'