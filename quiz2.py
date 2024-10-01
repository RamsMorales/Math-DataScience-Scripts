#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:31:29 2024

@author: ramsonmunoz

This code implements Newtons method for a root-finding problem for a given function.

We use the same algorithm from the project with a slight modification. 

This time, our stopping condition is |f(x)| < TOL
"""

import matplotlib.pyplot as plt
import numpy as np
import math

def f1(x):
    return  (x**3) - ((3*(x**2)) * (2 ** -x)) + ((3*x)*(4 ** -x)) - (8** -x)
def df1dx(x):
    return ((3*(x**2))) - ((6*x)* (2 ** -x) - math.log(2) * (2 ** -x) * (3*(x**2))) + (3*(4 ** -x) - math.log(4) * (4 ** -x) * (3*x)) + math.log(8) * (8 ** -x) 

def newtons_method(function, derivative, initial_guess, tolerance, maxIter):
    """
    Find the root of the function `function` using Newton's method
    and print iteration details in a table.

    Parameters:
    function : callable - the function for which we are finding the root
    derivative : callable - the derivative of the function
    initial_guess : float - the initial guess for the root
    tolerance : float - the tolerance for stopping criterion
    max_iter : int - maximum number of iterations

    Returns:
    float - the estimated root of the function
    """
    
    current = initial_guess
    iterationData = {}
    
    # Print table header
    print(f"{'Iteration':<10}{'p':<20}{'Function Value':<20}{'Absolute Error':<20}{'Relative Error':<20}")

    for i in range(maxIter):
        function_value = function(current)
        derivative_value = derivative(current)

        # Check if the derivative is zero to avoid division by zero
        if derivative_value == 0:
            raise ValueError("The derivative is zero. No solution found.")

        # Update the current guess using Newton's formula
        next_guess = current - function_value / derivative_value
        
        # Calculate absolute and relative errors
        absolute_error = abs(function(next_guess))
        relative_error = abs((next_guess - current) / next_guess) if next_guess != 0 else float('inf')

        # Print iteration details in a nice table format
        print(f"{i + 1:<10}{next_guess:<20.10f}{function_value:<20.10f}{absolute_error:<20.10f}{relative_error:<20.10f}")
        iterationData[i] = [current,absolute_error,relative_error]

        # Check for convergence
        if absolute_error < tolerance:
            return next_guess,iterationData

        current = next_guess  # Update current guess for the next iteration

    # If the maximum number of iterations is reached, return the last guess
    return current, iterationData

def main():
    
    initialGuess = 0.5
    
    tol = 1e-5
    
    maxIter = 10
    
    root,iterData = newtons_method(f1,df1dx , initialGuess, tol, maxIter)
    
    print("=====================================================================================\n")
    print("The approximate x within (%f,%f) for the given function is: %.9f\n" % (-tol,tol, root))
    print("=====================================================================================\n")
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    