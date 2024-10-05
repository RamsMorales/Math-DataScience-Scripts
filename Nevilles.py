#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:19:56 2024

@author: ramsonmunoz
"""
import numpy as np
import math

def NevillesMethod(approxiationPoint: float, data: np.ndarray) -> float:
    n = int(data.shape[0])
    
    
    Q = np.zeros((n,n))
    
    Q[:,0] = data[:,1]
    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = ((approxiationPoint - data[i + j,0]) * Q[i, j - 1] + 
                        (data[i,0] - approxiationPoint) * Q[i + 1, j - 1]) / (data[i,0] - data[i + j,0])
    print(Q,end="\n\n")      
    return Q[0,n-1]
            
def main():
    
    data = np.array([[0,0],
                     [1,1],
                     [2,math.sqrt(2)],
                     [4,2],
                     [5,math.sqrt(5)]])
    
    
    
    #print(data.shape)
    
    approxiationPoint = 3
    
    print("The approximation of sqrt(3) using the given points with nevilles method is: %.9f\n" % (NevillesMethod(approxiationPoint, data)))

if __name__ == "__main__":
    main()           