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
        
        if(len(matrix.shape) != 2):
            raise Exception("Invalid dimension. Expected dim 2 but got %d" %(len(matrix.shape)))
        elif(matrix.shape[0] != matrix.shape[1]):
            raise Exception("Please ensure a square matrix. Current matrix is %d by %d" %(matrix.shape[0],matrix.shape[1]))
        
    except Exception as e:
        print(e)
        sys.exit()
    
    


def main():
    
    testTensor = np.random.rand(2,2,2)
    exercise1(testTensor)
    
if __name__=="__main__":
    main()
