import os


'''
Function extract_name is a Helper Function for general extraction of filenames


inputs:
- path : string (path that would like to be searched)
- ext : string (extension of the files)
- output : string (filename of the saved file)
Outputs:

None

Saves file in the current diectory

'''

def extract_name(path, ext, output):
	ls = []
	for file in os.listdir(path):
		if file.endswith(ext):
			ls.append(file)
	with open(output, 'w') as f:
		for file in ls:
			f.write(str(file) + '\n')
	return f'Saved file in {output} successfully'