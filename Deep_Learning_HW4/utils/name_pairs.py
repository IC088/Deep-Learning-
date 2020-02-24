'''
Create list of filenames of the images and save it to a .csv file
'''

import os
import csv


'''
Function extract_filename is used to save the list of filenames to a csv file.
Input: 
- String (folder name containing the files that will be used for the training) [ folder ]
- String (Filename of the ) [ output ]
- String (File extension of the data) [ ext ]
Output: String (string confirmation that the file has been saved in the output)

'''

def extract_filename(folder, output, ext):

	image_names = []
	for file in os.listdir(folder):
		if file.endswith(ext):
			image_names.append(file)

	print(type(image_names))
	with open(output, "w") as outfile:
		outfile.write("\n".join(str(item) for item in image_names))

	return f'Files saved in {output}'