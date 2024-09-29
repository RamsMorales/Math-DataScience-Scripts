#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:04:04 2024

@author: ramsonmunoz
"""

import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    return  (x**3) - (5*(x**2)) + (2*x)
def df1dx(x):
    return ((3*(x**2))) - (10*x) + 2
def f2(x):
    return (x**3) - (2*(x**2)) - 5
def df2dx(x):
    return ((3*(x**2))) - (4*x) 
    

def bisectionMethod(function,left, right,tolerance, maxIter):
    
    
    i = 0
    
    function_left = function(left)
    
    iterationData = {}
    
    previous = None # initizalize previous
    
    # Print table header
    print(f"{'Iteration':<10}{'p':<20}{'Absolute Error':<20}{'Relative Error':<20}")
    
    while(i < maxIter):
        
        #calculates p
        current = left + float(right - left) / 2 
        
        function_current = function(current)
        
        # Calculate absolute and relative errors
        absolute_error = abs(function_current - 0)
        if previous is not None:
            relative_error = abs((current - previous) / current) if current != 0 else 0
        else:
            relative_error = float('inf')  # No previous in the first iteration

        # Print iteration details in a nice table format
        print(f"{i + 1:<10}{current:<20.10f}{absolute_error:<20.10f}{relative_error:<20.10f}")
        iterationData[i] = [current,absolute_error,relative_error]
        

        
        if(function_current == 0 or float(right - left) / 2 < tolerance):
            return current,iterationData
        
        i += 1
        
        previous = current
        
        if(function_left * function_current > 0):
            left = current
        else:
            right = current
            
    print("Method failed after %d iterations, MAX ITERATIONS = %d\n" %(i, maxIter))
    return iterationData

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

        # Check for convergence
        if absolute_error < tolerance:
            return next_guess

        current = next_guess  # Update current guess for the next iteration

    # If the maximum number of iterations is reached, return the last guess
    return current

def secant_method(function, initial_guess_1, initial_guess_2, tolerance, max_iter):
    """
    Find the root of the function `function` using the Secant method
    and print iteration details in a table.

    Parameters:
    function : callable - the function for which we are finding the root
    initial_guess_1 : float - the first initial guess
    initial_guess_2 : float - the second initial guess
    tolerance : float - the tolerance for stopping criterion
    max_iter : int - maximum number of iterations

    Returns:
    float - the estimated root of the function
    """

    current = initial_guess_1
    previous = initial_guess_2

    # Print table header
    print(f"{'Iteration':<10}{'p':<20}{'Function Value':<20}{'Absolute Error':<20}{'Relative Error':<20}")

    for i in range(max_iter):
        function_current = function(current)
        function_previous = function(previous)

        # Check if the function values are too close to avoid division by zero
        if function_current - function_previous == 0:
            raise ValueError("Function values are too close. No solution found.")

        # Update the current guess using the Secant formula
        next_guess = current - function_current * (current - previous) / (function_current - function_previous)

        # Calculate absolute and relative errors
        absolute_error = abs(function(next_guess))
        relative_error = abs((next_guess - current) / next_guess) if next_guess != 0 else float('inf')

        # Print iteration details in a nice table format
        print(f"{i + 1:<10}{next_guess:<20.10f}{function_current:<20.10f}{absolute_error:<20.10f}{relative_error:<20.10f}")

        # Check for convergence
        if absolute_error < tolerance:
            return next_guess

        previous, current = current, next_guess  # Update guesses for the next iteration

    # If the maximum number of iterations is reached, return the last guess
    return current

def mullers_method(function, initial_guess_1, initial_guess_2, initial_guess_3, tolerance, max_iter):
    """
    Find the root of the function `function` using Müller’s method
    and print iteration details in a table.

    Parameters:
    function : callable - the function for which we are finding the root
    initial_guess_1 : float - the first initial guess
    initial_guess_2 : float - the second initial guess
    initial_guess_3 : float - the third initial guess
    tolerance : float - the tolerance for stopping criterion
    max_iter : int - maximum number of iterations

    Returns:
    float - the estimated root of the function
    """

    x0, x1, x2 = initial_guess_1, initial_guess_2, initial_guess_3

    # Print table header
    print(f"{'Iteration':<10}{'p':<20}{'Function Value':<20}{'Absolute Error':<20}{'Relative Error':<20}")

    for i in range(max_iter):
        f0, f1, f2 = function(x0), function(x1), function(x2)

        # Calculate the denominator
        denominator = (f1 - f0) * (f2 - f0) * (f2 - f1)

        # Check if denominator is zero
        if denominator == 0:
            raise ValueError("Denominator is zero. No solution found.")

        # Calculate the next guess using Müller’s formula
        h1 = f1 - f0
        h2 = f2 - f0
        d = (h2 * h1 * (f1 - f0)) / denominator
        next_guess = x2 - (h2 * h1) / (h2 + h1 + (2 * d))

        # Calculate absolute and relative errors
        absolute_error = abs(function(next_guess))
        relative_error = abs((next_guess - x2) / next_guess) if next_guess != 0 else float('inf')

        # Print iteration details in a nice table format
        print(f"{i + 1:<10}{next_guess:<20.10f}{function(next_guess):<20.10f}{absolute_error:<20.10f}{relative_error:<20.10f}")

        # Check for convergence
        if absolute_error < tolerance:
            return next_guess

        # Update guesses for the next iteration
        x0, x1, x2 = x1, x2, next_guess

    # If the maximum number of iterations is reached, return the last guess
    return next_guess
        
    



def main():
    
            
    a = -10
    b = 1
    tol = 1e-4
    maxIter = 100000
    
    x = np.linspace(-9, 7,1000)
    y = f1(x)
    
    plt.plot(x,y)
    answer,iterData = bisectionMethod(f1, a, b, tol, maxIter)
    
    print(answer)
    
    print(type(iterData.values()))
    
if __name__ == "__main__":
    main()

    