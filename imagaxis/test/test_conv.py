# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:29:32 2020

@author: benno
"""

import numpy as np
import src
from convolution import basic_conv


n = 21
A = np.random.randn(n)
B = np.random.randn(n)


print('testing circular convolutions')
print('')

# convolve
C0 = basic_conv(A, B, ['x,y-x'], [0], [True], izeros=[n//2])[:n]

# explicit convolve (assuming index 0 is at n//2)
C1 = np.zeros(n)

for iy in range(n):
    for ix in range(n):
        # when iy = n//2
        # and  ix = n//2   
        # C[iy] = A[ix] B[iy - ix + n//2]
    
        C1[iy] += A[ix] * B[(iy - ix + n//2)%n]

print(np.mean((C0-C1)).real)


# convolve
C0 = basic_conv(A, B, ['x,x+y'], [0], [True], izeros=[n//2])[:n]

# explicit convolve (assuming index 0 is at n//2)
C1 = np.zeros(n)

for iy in range(n):
    for ix in range(n):
        # when iy = n//2
        # and  ix = n//2  
        # C[iy] = A[ix] B[ix + iy - n//2]
    
        C1[iy] += A[ix] * B[(ix + iy - n//2)%n]

print(np.mean((C0-C1)).real)




print('testing non-circular convolutions')
print('')

# convolve
C0 = basic_conv(A, B, ['x,y-x'], [0], [False], izeros=[n//2])[:n]
#print(C0.real)

# explicit convolve (assuming index 0 is at n//2)
C1 = np.zeros(n)

for iy in range(n):
    for ix in range(n):
        # when iy = n//2
        # and  ix = n//2   
        # C[iy] = A[ix] B[iy - ix + n//2]
    
        if iy - ix + n//2 >=0 and iy - ix + n//2 < n:   
            C1[iy] += A[ix] * B[iy - ix + n//2]

print(np.mean((C0-C1)).real)




# convolve
C0 = basic_conv(A, B, ['x,x+y'], [0], [False], izeros=[n//2])[:n]
#print(C0.real)

# explicit convolve (assuming index 0 is at n//2)
C1 = np.zeros(n)

for iy in range(n):
    for ix in range(n):
        # when iy = n//2
        # and  ix = n//2  
        # C[iy] = A[ix] B[ix + iy - n//2]
    
        if ix + iy - n//2 >=0 and ix + iy - n//2 < n:
            C1[iy] += A[ix] * B[ix + iy - n//2]

print(np.mean((C0-C1)).real)





# convolve
C0 = basic_conv(A, B, ['x,y-x'], [0], [False], izeros=[4])[:n]
#print(C0.real)


# explicit convolve (where index 0 is at iz)
C1 = np.zeros(n)

iz = 4
for iy in range(n):
    for ix in range(n):
        # when iy = iz
        # and  ix = iz   
        # C[iy] = A[ix] B[iy - ix + iz]
    
        if iy - ix + iz >=0 and iy - ix + iz < n:   
            C1[iy] += A[ix] * B[iy - ix + iz]

print(np.mean((C0-C1)).real)


# convolve
C0 = basic_conv(A, B, ['x,x+y'], [0], [False], izeros=[4])[:n]
#print(C0.real)


# explicit convolve (where index 0 is at iz)
C1 = np.zeros(n)

iz = 4
for iy in range(n):
    for ix in range(n):
        # when iy = n//2
        # and  ix = n//2  
        # C[iy] = A[ix] B[ix + iy - n//2]
    
        if ix + iy - iz >=0 and ix + iy - iz < n:
            C1[iy] += A[ix] * B[ix + iy - iz]

print(np.mean((C0-C1)).real)


