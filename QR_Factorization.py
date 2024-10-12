#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:20:26 2024

@author: ramsonmunoz
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
 
def factorizationQR(matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    
    numVectors= matrix.shape[0]
    dimensions = matrix.shape[1]
    
    L = np.zeros(numVectors)
    for i in range(numVectors):
        L[i] = np.sqrt(matrix[i].T @ matrix[i])
    
    V = matrix.copy() / L
    Q = V.copy()
    
    for j in range(0,numVectors):
        Q[j] = V[j]/np.sqrt(V[j].T @ V[j])
        for k in range(j, numVectors):
            V[k] = V[k] - (Q[j].T @ V[k] * Q[j])
            
    R = Q.T @ matrix
            
    return Q, R

def solveLinearSystemQR(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    
    try:
        
        #This block is used to verify the inputs from the problem statement. end will be signified by ****** 
        
        '''Here we ensure a square n x n matrix that is invertible and our solution vector is of dimension n'''
        
        #*****************************************************************************************************************
        if(len(matrix.shape) != 2):
            raise Exception("Invalid dimension. Expected dim 2 but got %d" %(len(matrix.shape)))
            
        if(matrix.shape[0] != matrix.shape[1]):
            raise Exception("Please ensure a square matrix. Current matrix is %d by %d" %(matrix.shape[0],matrix.shape[1]))
            
        det_matrix = np.linalg.det(matrix)
        
        if np.isclose(det_matrix, 0):
            raise Exception("Matrix is not invertible. Please ensure invertible matrix")
            
        if (vector.shape[0] != matrix.shape[1]):
            raise Exception("Vector has invalid dimension. Expeced %d but got %d" %(matrix.shape[1], vector.shape[0]))
        #*****************************************************************************************************************
        
        '''
        Solving the system of equations with QR factorization. 
        '''
        Q,R = factorizationQR(matrix)
        
        transposeQ = Q.T
        
        y = transposeQ @ vector # y = Q.T @ b
        
        solution = np.linalg.inv(R) @ y # back substitution to find x
        
        
        return solution
        
        
        
        
        
        
    except Exception as e:
        print(e)
        sys.exit()
    
    


def main():
    #test system with known solution x = [1 -1].T
    A = np.array([[1,1],[2,1]])
    
    b = np.array([[0],[1]])
    
    
    print(solveLinearSystemQR(A,b)) # it worked
    
if __name__=="__main__":
    main()
