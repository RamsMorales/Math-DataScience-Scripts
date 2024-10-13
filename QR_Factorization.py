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
import copy
 
def factorizationQR(matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    
    '''
    Although the correct answer was being produced, R is not upper triangular so I cant use this for back-substitution algorithm
    '''
    
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
        Q,R = sp.linalg.qr(matrix)
        
        
        y =  Q.T @ vector # y = Q.T @ b
        
        
        x = np.zeros(y.shape) # initialize x
        
        #Back substitution algorithm
        for i in range(x.shape[0]-1,-1,-1):
            if i+1 == x.shape[0]:
                x[i] = y[i] / R[i][i]
            elif i < 0:
                break
            else:
                x[i] = (y[i] - R[i,i+1:] @ x[i+1:]) / R[i][i]

            
        
        solution = x
        
        print("Residual is %.e\n" %(np.linalg.norm(vector - (matrix @ solution))))
        
        return solution
        
        
        
        
        
        
    except Exception as e:
        print(e)
        sys.exit()
    
    
    
def houselholder(matrix: np.ndarray ) -> (np.ndarray, np.ndarray):
    
    try:
        #*****************************************************************************************************************
        if(len(matrix.shape) != 2):
            raise Exception("Invalid dimension. Expected dim 2 but got %d" %(len(matrix.shape)))
            
        nRows, nCols = matrix.shape[0],matrix.shape[1]
        
        #calculate rank of matrix
        rank = np.linalg.matrix_rank(matrix)

            
        if rank != nCols:
            raise Exception("Invalid Rank. Expected %d but got %d" % (nCols,rank))
        #*****************************************************************************************************************
        
        #Set R to be a copy of our input matrix
        R = copy.deepcopy(matrix)
        
        #Set Q to the m x m Identity matrix
        Q = np.eye(nRows)
        
        #defining sign lambda calc function
        sign = lambda x: 1 if x >= 0 else -1
        
        for k in range(nCols):
            #let U be a copy of R from kth row forward on the kth column. Deep copy is so we don't modify R yet
            U = copy.deepcopy(R[k:,k])
            #add sign * magnitude of U to the first element of u
            U[0] += sign(U[0]) * np.linalg.norm(U)
            #normolize u
            U = U * (np.linalg.norm(U) ** -1)
            #modify R with reflection
            R[k:,k:] = R[k:,k:] - (2 * np.outer(U,U.T @ R[k:,k:]))
            #modify Q with reflection
            Q[k:,:] = Q[k:,:] - (2 * np.outer(U,U.T @ Q[k:,:]))
            
        return Q.T, R
            
            
            

        
        
        
        
            
    except Exception as e:
        print(e)
        sys.exit()
    
        


def main():
   
    A = np.array([[1,2,3],[0,1,5],[5,6,0]])
    b = np.array([[0],[1],[17]])
    #test system with known solution x = [-1 1].T
    #A = np.array([[1,1],[1,2]])
    #b = np.array([[0],[1]])
    
    print("The following is the solution printout for Exercise 1:\n")
    solution = solveLinearSystemQR(A, b)
    print(solution)

    print("\nThe following is the solution printout for Exercise 2:\n")
    
    A = np.random.random((5,3))
    Q1,R1 = sp.linalg.qr(A)
    
    print("Shape for SciPy qr A.shape, Q.shape, R.shape\n")
    print(A.shape,Q1.shape,R1.shape)
    print("\nA is within tolerance of QR from SciPy")
    print(np.allclose(Q1 @ R1, A))
    print("A is within tolerance of QR from native HouseHolder")    
    Q2,R2 = houselholder(A)
    
    print(np.allclose(Q2 @ R2, A))
    
    
    
    
    
if __name__=="__main__":
    main()
