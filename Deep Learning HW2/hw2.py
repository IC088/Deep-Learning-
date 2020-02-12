'''
Homework 2

Ivan Christian
'''

import torch

from random import randrange
import numpy as np
import matplotlib.pyplot as plt

from time import time
'''
let the tensors be defined as the following for hte calculations
'''

N = randrange(1,5000)
D = randrange(1,300)
P = randrange(1,500)
'''
TASK 1
'''
#
# print('TASK 1 \n')
#
# X = torch.randn(N,D)
# T = torch.randn(P,D)
#
#
# print(f'X = {X.size()}')
# print(f'T = {T.size()}')
#
# '''
# 2 loops over i,j appraoch
#
# '''
# a = time()
# dist = np.zeros(N)
# for i in range(N):
# 	for j in range(P):
# 		dist[i] += torch.sum(X[i,:] - T[j,:])
# time_elapsed=float(time()) - float(a)
# print(dist.shape)
#
# print(f'Time elapsed for "For loop" : {time_elapsed}')
# '''
# numpy broadcasting
# '''
#
# x = X.data.numpy()
# t = T.data.numpy()
#
# # dist[i,j] = ||x[i,:]-y[j,:]||
# b = time()
# d = np.linalg.norm(x[:, np.newaxis, :] - t, axis = 2)
# time_elapsed=float(time()) - float(b)
# print((d**2).shape)
# print(f'Time elapsed for "Numpy broadcasting" : {time_elapsed}')
#
# '''
# pytorch cpu
#
# torch broadcasting
# '''
#
def pytorch_eucledian_distances(x, y,device=torch.device('cpu')):
    '''
    Input: x is an Nxd matrix
           y is an Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances
# c = time()
# print('Using CPU')
# torch.device("cpu")
# l = pytorch_eucledian_distances(X,T)
# time_elapsed=float(time()) - float(c)
# print(l.size())
# print(f'Time elapsed for "CPU" : {time_elapsed}')
#
#
#
#
#
# device = torch.device( cuda if torch.cuda.is_available() else 'cpu' )
# if device.type == 'cuda':
# 	X = torch.randn(N,D)
# 	T = torch.randn(P,D)
# 	X.to(device)
# 	T.to(device)
# 	d = time()
# 	l = pytorch_eucledian_distances(X,T, device)
# 	time_elapsed=float(time()) - float(d)
# 	print(l.size())
# 	print(f'Time elapsed for "GPU" : {time_elapsed}')





'''
K means implementation

function Kmeans is used to create teh k-means result

input:
x : data X[i,:] 5 blobs from 5 different gaussian means
k : Desired number of clusters P
m : max number of iterations
device : choose which device to use, defaulteed to cpu

output:
returns
choice_clusters : the classification cluster for the points
init_centroid_i : final centroids that classifies the data points
'''



def KMeans(x, k, m, device=torch.device('cpu')):
	'''
	TODO: initialise cluster centers T[j,:], select 5 from x fixed
	'''
	init_centroid = initialise(x,k)
	x_i = torch.from_numpy(x).float().to(device)
	init_centroid_i = torch.from_numpy(init_centroid).float().to(device)
	iterations = 0


	while iterations < m:
		dist = pytorch_eucledian_distances(x_i, init_centroid_i,device=device)
		choice_cluster = torch.argmin(dist, dim=1)
		# print(choice_cluster)

		initial_state_pre = init_centroid_i.clone()
		for index in range(k):
			selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
			selected = torch.index_select(x_i, 0, selected)
			init_centroid_i[index] = selected.mean(dim=0)
		print(init_centroid_i - initial_state_pre)
		# center_shift = torch.sum(torch.sqrt(torch.sum((init_centroid_i - initial_state_pre) ** 2, dim=1)))
		# center_shift = pytorch_eucledian_distances(init_centroid_i, initial_state_pre,device=device)

		iterations+=1
	# print(center_shift)
	return choice_cluster, init_centroid_i

'''
Function initialise is a helper function in doing the k-means.
The use of this function is to choose the initial k centroids for the k-means process.
The first k elements of x will be used as the initial centroids for the process

input :
x : numpy array of the dataset. This will be used to determine the initial centroids that is chosenself.
k : integer input to determine how many elements are going to become the initial centroidsself.
k is defaulted to 5, but Homework mentions that k can be 2,5,8
output:

init_centroid : array of initial centroids.

'''

def initialise(x,k=5):
	num_samples = len(x)
	'''
	TODO: fixed choices for X
	'''
	idx = [i for i in range(k)]
	indices = tuple(idx)
	init_centroid = x[indices,:]

	return init_centroid

print('TASK 2')
dataset = []
for i in range(5):
	mean = randrange(-10,10)
	sigma = randrange(1,2)
	dataset1 = np.random.normal(mean,sigma,(50,2))
	dataset.append(dataset1)

X = np.vstack(dataset)
X = np.take(X,np.random.permutation(X.shape[0]),axis=0,out=X)

choice_cluster, centroid= KMeans(X, 8, 500)
# print(choice_cluster, centroid)
plt.scatter(X[:, 0], X[:, 1], c=choice_cluster)
plt.scatter(centroid[:,0], centroid[:,1], c ='black')
plt.show()
