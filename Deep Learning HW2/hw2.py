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
y = 0
for i in range(N):
	for j in range(P):
		y += X[i,:] - T[j,:]
print(y)


'''
numpy broadcasting
'''

x = X.data.numpy()
t = T.data.numpy()

print(x,t)

'''
pytorch cpu

torch broadcasting
'''
