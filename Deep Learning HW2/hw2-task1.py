'''
Homework 2

Ivan Christian
'''

import torch

from random import randrange
import numpy as np
import matplotlib.pyplot as plt


import os
from time import time
'''
let the tensors be defined as the following for hte calculations
'''
'''
Function pytorch_eucledian_distances is a helper function to calculate the eucledian distances 
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

N = randrange(1,500)
D = randrange(1,200)
P = randrange(1,300)

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
dist = np.zeros((N,P))
for i in range(N):
	for j in range(P):
		y = torch.sum((X[i,:] - T[j,:])**2,dim=0)
		dist[i][j] = y
print(dist)
print(dist.shape)
time_elapsed=float(time()) - float(a)

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
print((d**2))
print((d**2).shape)
print(f'Time elapsed for "Numpy broadcasting" : {time_elapsed}')

'''
pytorch cpu

torch broadcasting
'''

c = time()
print('Using CPU')
torch.device("cpu")
l = pytorch_eucledian_distances(X,T)
time_elapsed=float(time()) - float(c)
print(l.size())
print(f'Time elapsed for "CPU" : {time_elapsed}')



device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
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
