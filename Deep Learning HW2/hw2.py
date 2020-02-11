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

print('TASK 1 \n')

X = torch.randn(N,D)
T = torch.randn(P,D)


print(f'X = {X.size()}')
print(f'T = {T.size()}')

'''
2 loops over i,j appraoch

'''
a = time()
dist = np.zeros(N)
for i in range(N):
	for j in range(P):
		dist[i] += torch.sum(X[i,:] - T[j,:])
time_elapsed=float(time()) - float(a)
print(dist.shape)

print(f'Time elapsed for "For loop" : {time_elapsed}')
'''
numpy broadcasting
'''

x = X.data.numpy()
t = T.data.numpy()

# dist[i,j] = ||x[i,:]-y[j,:]||
b = time()
d = np.linalg.norm(x[:, np.newaxis, :] - t, axis = 2)
time_elapsed=float(time()) - float(b)
print((d**2).shape)
print(f'Time elapsed for "Numpy broadcasting" : {time_elapsed}')

'''
pytorch cpu

torch broadcasting
'''

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
c = time()
print('Using CPU')
torch.device("cpu")
l = pytorch_eucledian_distances(X,T)
time_elapsed=float(time()) - float(c)
print(l.size())
print(f'Time elapsed for "CPU" : {time_elapsed}')





device = torch.device('cuda')
print('Using device:', torch.cuda.get_device_name(0))
if device.type == 'cuda':
	X = torch.randn(N,D)
	T = torch.randn(P,D)
	X.to(device)
	T.to(device)
	d = time()
	l = pytorch_eucledian_distances(X,T, device)
	time_elapsed=float(time()) - float(d)
	print(l.size())
	print(f'Time elapsed for "GPU" : {time_elapsed}')





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
'''



def KMeans(x, k, m, device=torch.device('cpu')):
	
	x = x.float()
	x = x.to(device)
	iterations = 0
	'''
	TODO: initialise cluster centers T[j,:], select 5 from x fixed
	'''

	init_centroid = initialise(x,k)
	while iterations < m:
		dist = pytorch_eucledian_distances(X, init_centroid,device=device)
		choice_cluster = torch.argmin(dist, dim=1)

		initial_state_pre = init_centroid.clone()
		for index in range(k):
			selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
			selected = torch.index_select(X, 0, selected)
			init_centroid[index] = selected.mean(dim=0)
		center_shift = torch.sum(torch.sqrt(torch.sum((init_centroid - initial_state_pre) ** 2, dim=1)))
		iterations+=1

	return choice_cluster, init_centroid
def initialise(x,k):
	num_samples = len(x)
	'''
	TODO: fixed choices for X (Need to ask around)
	'''
	indices = np.random.choice(num_samples, k, replace=False)
	init_centroid = x[indices,:]
	return init_centroid

print('TASK 2')
choice_cluster, init_centroid= KMeans(X, 5, 1000)
print(choice_cluster, init_centroid)