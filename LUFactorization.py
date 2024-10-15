#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:20:37 2024

@author: ramsonmunoz
"""
import numpy as np
import scipy as sp
import sys

def computeLU(matrix: np.ndarray) -> (np.ndarray,np.ndarray):
    
    '''
    We assume no permutation necessary
    '''
    
    nCols = nRows = matrix.shape[0] # we know its square from calling function
    
    l = np.eye(nCols)
    
    for i in range(nCols):
        for j in range(nRows-1,i,-1):
            #guass elimination scheme
            
            scaleFactor = float(matrix[j,i]/matrix[i,i])
            
            matrix[j,:] = matrix[j,:] - (scaleFactor * matrix[i,:]) 
            
            #performing the same operations on the identity matrix
            l[j,:] = l[j,:] - (scaleFactor * l[i,:]) 
            
    #lA = U which I call matrix in this 
    
    #Lower = inverise of l
    Lower = np.linalg.inv(l)
     
    
    return Lower, matrix
    
            
    
    
    
    

def solveLinearSystemLU(matrix: np.ndarray, vector: np.ndarray) ->np.ndarray:
    
    try:
        
        #This block is used to verify the inputs from the problem statement. end will be signified by ****** 
        
        '''Here we ensure a square n x n matrix that is invertible and our solution vector is of dimension n'''
        
        #*****************************************************************************************************************
        if(len(matrix.shape) != 2):
            raise Exception("Invalid dimension. Expected dim 2 but got %d" %(len(matrix.shape)))
            
        if(matrix.shape[0] != matrix.shape[1]):
            raise Exception("Please ensure a square matrix. Current matrix is %d by %d" %(matrix.shape[0],matrix.shape[1]))
        #*****************************************************************************************************************
        
        L,U = computeLU(matrix)
        
        #Ly = b forward substitution so that we can back substitute to find x
        
        y = np.zeros(vector.shape)
        
        for k in range(y.shape[0]):
            if k == 0:
                y[k,:] = vector[k,:]
            else:    
                #implementing 2.1
                y[k,:] = vector[k,:] - (L[k,:k] @y[:k,:])
            
        #now we back substitute
        x = np.zeros(y.shape) # initialize x
        
        #Back substitution algorithm
        for i in range(x.shape[0]-1,-1,-1):
            if i+1 == x.shape[0]:
                x[i] = y[i] / U[i][i]
            elif i < 0:
                break
            else:
                x[i] = (y[i] - U[i,i+1:] @ x[i+1:]) / U[i][i]
                
        return x
        
            
    except Exception as e:
        
        print(e)
        sys.exit
    
    
    
def main():
    
    A = np.array([[25,5,1],[64,8,1],[144,12,1]],dtype=float)
    
    #print(A)
    
    b = np.array([[106.8],[177.2],[279.2]])
    
    print("The solution for exercise 1 is: ")
    print(solveLinearSystemLU(A, b))
    
if __name__ == "__main__":
    main()