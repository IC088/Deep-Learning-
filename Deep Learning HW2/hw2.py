'''
Homework 2 
'''

import torch

from random import randrange
import numpy as np

'''
let the tensors be defined as the following for hte calculations
'''

N = randrange(1,10)
D = randrange(1,10)
P = randrange(1,10)


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
print(d)
	# 	print(f'T[{j}] = {T[j,:]}')
	# print(f'X[{i}] = {X[i,:]}')


'''
numpy broadcasting
'''

x = X.data.numpy()
t = T.data.numpy()

'''
dist[i,j] = ||x[i,:]-y[j,:]||
'''

d = np.linalg.norm(x[:, np.newaxis, :] - t, axis = 2)
print(d**2)

'''
pytorch cpu

torch broadcasting
'''

def expanded_pairwise_distances(x, y):
    '''
    Input: x is an Nxd matrix
           y is an Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances

l = expanded_pairwise_distances(X,T)
print(l)


