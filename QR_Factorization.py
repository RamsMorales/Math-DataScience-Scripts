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


def exercise1(matrix: np.ndarray):
    
    try:
        
        #This block is used to verify the inputs from the problem statement. end will be signified by ****** 
        
        '''Here we ensure a square n x n matrix that is invertible'''
        
        #*****************************************************************************************************************
        if(len(matrix.shape) != 2):
            raise Exception("Invalid dimension. Expected dim 2 but got %d" %(len(matrix.shape)))
            
        if(matrix.shape[0] != matrix.shape[1]):
            raise Exception("Please ensure a square matrix. Current matrix is %d by %d" %(matrix.shape[0],matrix.shape[1]))
            
        det_matrix = np.linalg.det(matrix)
        
        if np.isclose(det_matrix, 0):
            raise Exception("Matrix is not invertible. Please ensure invertible matrix")
        #*****************************************************************************************************************
        
        
    except Exception as e:
        print(e)
        sys.exit()
    
    


def main():
    
    #testTensor = np.random.rand(2,2,2)
    testNonSquare = np.random.rand(3,2)
    print(testNonSquare.shape)
    testNonInvertible = np.array([[0,0],[0,1]])
    exercise1(testNonSquare)
    
if __name__=="__main__":
    main()
