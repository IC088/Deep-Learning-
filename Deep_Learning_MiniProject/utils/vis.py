'''
Visualisation Utils function
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle



def visualise_from_pickle(score_file, path, dataset, Pascal, mode = 1):
	'''
	Function to visualise the images from pickle
	Input:
	- score_file : pickle ( data for the scores )
	- path : string (root directory)
	- dataset : Dataset (Dataset used)
	- Pascal : Class ( class to help visualise )
	- mode : int ( 1 for top 50, 0 for bot 50 )

	Output:
	- images50 : list (List of PIL Images)
	- 
	'''

	print('Showing top/bot 5 classes')
	root_dir = path
	with open(score_file, 'rb') as s:
		scores = pickle.load(s)
	for index,c in enumerate(dataset.classes[:5]): # top 5 classes
		img_names = Pascal.imgs_from_category_as_list(c,'val')

		all_scores_c = [x for x in scores if x[0][:-4] in img_names]

		all_scores_c.sort(key=lambda x: x[1][index], reverse=True)
		
		
		if mode == 1:
			print(f'Showing top 50 {c} images')

			image_scores_50 = all_scores_c[:50]
			images50 = [Image.open(os.path.join(path, "JPEGImages", f'{image[0]}')) for image in image_scores_50]

			scores50 = [str(float(image[1][index]))[:7] for image in image_scores_50]

			name = f'top50_{c}.png'
			directory = os.path.join('results','top')
			if not os.path.exists(directory):
				os.makedirs(directory)

			show_images(images50,scores50, directory ,name)

		elif mode == 0:
			print(f'Showing bottom 50 {c} images')
			image_scores_50 = all_scores_c[-50:]
			images50 = [Image.open(os.path.join(path, "JPEGImages", f'{image[0]}')) for image in image_scores_50]
			scores50 = [str(float(image[1][index]))[:7] for image in image_scores_50]

			name = f'bot50_{c}.png'
			directory = os.path.join('results','bot')
			if not os.path.exists(directory):
				os.makedirs(directory)

			show_images(images50,scores50, directory ,name)
		else:
			return 'NOT VALID'
	print(f'Image saved in {directory}')
	return images50, scores50

def show_images(images, scores50, directory, name):
	'''
	Function to show the images

	Input:
	- image : list (list of PIL Images)
	- score : list (list of scores)

	Output:
	- None (but will show plt figure)
	'''

	n_images = len(images)
	fig = plt.figure(figsize=(15,9))

	imscore = list(zip(images,scores50))
	for n, (image, score) in enumerate(imscore):
		a = fig.add_subplot(5, np.ceil(n_images/float(5)), n + 1)
		plt.axis('off')
		plt.imshow(image)
		a.set_title(f'{name}\n{score}',fontsize= 5)
	plt.savefig(os.path.join(directory,name))
	# plt.show()


def tailAccuracy(score_file, path, dataset, Pascal):
	'''
	Function calculates the tail accuracy ( precision with a threshold )

	Input: 
	- score_file : string (file that stores scores in it )
	- path : string
	- dataset : Dataset
	- Pascal : class object (Pascal Class object )

	Output:
	- None (but will display the graphs)

	Beacuse the activation function for the model is signmoid(), t-value starts from 0.5

	'''
	
	with open(score_file, 'rb') as s:
		scores = pickle.load(s)

	t_vals = np.linspace(0.5,0.99, 20)
	tacc_all_classes = []
	for index,c in enumerate(dataset.classes):
		print(c)
		ground_truth_classification = Pascal.imgs_from_category_as_list(c,'val')
		true_positive = [x[1][index] for x in scores if x[0][:-4] in ground_truth_classification]
		false_positive = [x[1][index] for x in scores if x[0][:-4] not in ground_truth_classification]
		tail_acc_list = []
		for t in t_vals:
			print(t)
			tp = len([x for x in true_positive if x > t])
			fp = len([x for x in false_positive if x > t])
			tail_acc_list.append(tp/(tp+fp))


		fig = plt.figure(figsize=(10,6))
		plt.title(c)
		plt.plot(t_vals, tail_acc_list)
		plt.axis([0.45, 1, 0, 1])

		tacc_all_classes.append(tail_acc_list)

		directory = os.path.join('results','tail_acc')
		if not os.path.exists(directory):
			os.makedirs(directory)

		plt.savefig(os.path.join(directory,f'{c}_tail_acc.png'))
		# plt.show()

	avg = np.mean(tacc_all_classes, axis = 0)

	fig = plt.figure(figsize=(10,6))
	plt.title('Tail Accuracy Average over 20 classes vs t')
	plt.plot(t_vals, avg)
	plt.axis([0.45, 1, 0, 1])


	directory = os.path.join('results','tail_acc_avg')
	if not os.path.exists(directory):
		os.makedirs(directory)

	plt.savefig(os.path.join(directory, 'Average_tail_acc.png'))
	# plt.show()
