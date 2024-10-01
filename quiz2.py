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
import cmath

def f1(x):
    return  (x**3) - ((3*(x**2)) * (2 ** -x)) + ((3*x)*(4 ** -x)) - (8** -x)
def df1dx(x):
    return ((3*(x**2))) - ((6*x)* (2 ** -x) - math.log(2) * (2 ** -x) * (3*(x**2))) + (3*(4 ** -x) - math.log(4) * (4 ** -x) * (3*x)) + math.log(8) * (8 ** -x)

def f2(x):
    return (600 * (x**4)) - (550 * (x**3)) + (200*(x**2)) - (20*x) - 1


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

def mullers_method(function, a, b, c, tolerance, max_iter):
    """
    Find the root of the function `function` using Müller’s method
    and print iteration details in a table.

    Parameters:
    function : callable - the function for which we are finding the root
    a, b, c : float - three initial guesses
    tolerance : float - the tolerance for stopping criterion
    max_iter : int - maximum number of iterations

    Returns:
    float - the estimated root of the function
    """
    
    # Print table header
    print(f"{'Iteration':<10}{'p':<20}{'Function Value':<20}{'Absolute Error':<20}{'Relative Error':<20}")
    iterationData = {}

    for i in range(max_iter):
        f1 = function(a)
        f2 = function(b)
        f3 = function(c)
        
        d1 = f1 - f3
        d2 = f2 - f3
        h1 = a - c
        h2 = b - c

        a0 = f3
        a1 = (d2 * h1**2 - d1 * h2**2) / (h1 * h2 * (h1 - h2))
        a2 = (d1 * h2 - d2 * h1) / (h1 * h2 * (h1 - h2))

        # Calculate potential roots
        sqrt_term = cmath.sqrt(a1**2 - 4 * a0 * a2)
        if a1 + abs(sqrt_term) != 0:
            x = (-2 * a0) / (a1 + abs(sqrt_term))
        else:
            x = float('inf')  # Avoid division by zero
        
        if a1 - abs(sqrt_term) != 0:
            y = (-2 * a0) / (a1 - abs(sqrt_term))
        else:
            y = float('inf')  # Avoid division by zero

        # Taking the root which is closer to c
        result = x + c if x >= y else y + c
        
        # Check for convergence
        absolute_error = abs(function(result))
        relative_error = abs((result - c) / result) if result != 0 else float('inf')

        # Print iteration details in a nice table format
        print(f"{i + 1:<10}{result:<20.10f}{absolute_error:<20.10f}{relative_error:<20.10f}")
        iterationData[i] = [result,absolute_error,relative_error]

        # Checking for resemblance to two decimal places
        if round(result, 2) == round(c, 2):
            break

        a, b, c = b, c, result  # Update guesses

    if i >= max_iter:
        print("Root can't be found using Müller method")
    else:
        print(f"The root is approximately: {result}")

    return result,iterationData

def main():
    
    initialGuess = 0.5
    
    tol = 1e-4
    
    maxIter = 10
    
    #root,iterData = newtons_method(f1,df1dx , initialGuess, tol, maxIter)
    
    root,iterData = mullers_method(f2, 0.1, 0.45, 1, tol, maxIter)
    
    print("=====================================================================================\n")
    print("The approximate x within (%f,%f) for the given function is: %.9f\n" % (-tol,tol, root))
    print("=====================================================================================\n")
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    