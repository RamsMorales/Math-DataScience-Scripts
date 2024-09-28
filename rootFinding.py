#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:04:04 2024

@author: ramsonmunoz
"""

import matplotlib.pyplot as plt
import numpy as np

def function(x):
    
    return x * (x + 2)

def bisectionMethod(function,left, right,tolerance, maxIter):
    
    
    i = 0;
    
    function_left = function(left)
    
    previous = None # initizalize previous
    
    # Print table header
    print(f"{'Iteration':<10}{'p':<20}{'Absolute Error':<20}{'Relative Error':<20}")
    
    while(i < maxIter):
        
        #calculates p
        current = left + float(right - left) / 2 
        
        function_current = function(current)
        
        # Calculate absolute and relative errors
        absolute_error = abs(function_current)
        if previous is not None:
            relative_error = abs((current - previous) / current) if current != 0 else 0
        else:
            relative_error = float('inf')  # No previous c in the first iteration

        # Print iteration details in a nice table format
        print(f"{i + 1:<10}{current:<20.10f}{absolute_error:<20.10f}{relative_error:<20.10f}")

        
        if(function_current == 0 or float(right - left) / 2 < tolerance):
            return current
        
        i += 1
        
        previous = current
        
        if(function_left * function_current > 0):
            left = current
        else:
            right = current
            
    print("Method failed after %d iterations, MAX ITERATIONS = %d\n" %(i, maxIter))
    return

def main():
    
            
    a = -100
    b = 5000
    tol = 1e-8
    maxIter = 100000
    
    x = np.linspace(-9, 7,1000)
    y = function(x)
    
    plt.plot(x,y)
    answer = bisectionMethod(function, a, b, tol, maxIter)
    print(answer)
    
if __name__ == "__main__":
    main()

    