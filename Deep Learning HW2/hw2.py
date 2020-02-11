'''
Homework 2 
'''

import torch

from random import randrange
import numpy as np


'''
let the tensors be defined as the following for hte calculations
'''

N = randrange(1,100)
D = randrange(1,100)
P = randrange(1,100)
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
d = 0
for i in range(N):
	for j in range(P):
		d += (X[i,:] - T[j,:])**2
# print(d)

'''
numpy broadcasting
'''

x = X.data.numpy()
t = T.data.numpy()

# dist[i,j] = ||x[i,:]-y[j,:]||

d = np.linalg.norm(x[:, np.newaxis, :] - t, axis = 2)
# print(d**2)

'''
pytorch cpu

torch broadcasting
'''

def pytorch_eucledian_distances(x, y):
    '''
    Input: x is an Nxd matrix
           y is an Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances

print('Using CPU')
torch.device("cpu")
l = pytorch_eucledian_distances(X,T)
# print(l)




device = torch.device('cuda')
print('Using device:', torch.cuda.get_device_name(0))
if device.type == 'cuda':
	N = randrange(1,100)
	D = randrange(1,100)
	P = randrange(1,100)
	X = torch.randn(N,D)
	T = torch.randn(P,D)

	X.to(device)
	T.to(device)
	l = pytorch_eucledian_distances(X,T)
	# print(l)






'''
K means implementation

function Kmeans is used to create teh k-means result

input:
x : data X[i,:] 5 blobs from 5 different gaussian means 
k : Desired cluster P
m : max number of iterations

output:
returns the 
'''

def KMeans(x, k, m):
	'''
	TODO: initialise cluster centers T[j,:], select 5 from x fixed
	'''
	

	x_i = x.unsqueeze(1)
	c = torch.gather(x, 5)
	return c
	'''
	TODO: iterate the loop at most m times, stop when the 
	'''
	# for i in range(m):
	# 	c_j = c.unsqueeze(0)
	# 	d_ij = pytorch_eucledian_distances(x_i,c_j)

	# 	cl = D_ij.argmin(dim=1).long().view(-1)

	# 	Ncl = torch.bincount(cl).type(torchtype[dtype]) 
 #        for d in range(D): 
 #            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl
	# return 



results = KMeans(X, 5, 5)
print(results)
