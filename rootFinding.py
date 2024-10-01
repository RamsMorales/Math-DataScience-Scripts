#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:04:04 2024

@author: ramsonmunoz
"""

import matplotlib.pyplot as plt
import numpy as np
import math

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
    iterationData = {}
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
        iterationData[i] = [current,absolute_error,relative_error]
        
        # Check for convergence
        if absolute_error < tolerance:
            return next_guess,iterationData

        previous, current = current, next_guess  # Update guesses for the next iteration

    # If the maximum number of iterations is reached, return the last guess
    return current,iterationData

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
        sqrt_term = math.sqrt(a1**2 - 4 * a0 * a2)
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
        
   
def printHeader(name):
    print("==================================================================================\n")
    print("\t\t\t\t\t\t\t%s\n" %(name))
    print("==================================================================================\n")
    
def extractValues(method):
    
    i = 0
    BX = []
    BY = []
    
    for iteration in method.values():
        #print(i,iteration[0])
        BX.append(i)
        BY.append(iteration[0])
        i+=1
    return BX, BY
    
def graphConvergence(bisection, newtons, secant, mullers, figureTitle):
    
  
    BX,BY = extractValues(bisection) 
    NX,NY = extractValues(newtons)
    SX,SY = extractValues(secant)
    MX,MY = extractValues(mullers)  
    
    fig,ax = plt.subplots(2,2)
    ax[0,0].plot(BX,BY)
    ax[0, 0].set_title('P_0 = 10')
    ax[0,0].set(ylabel='Approximation value')
    ax[0,1].plot(NX,NY)
    ax[0, 1].set_title('P_0 = 100')
    ax[1,0].plot(SX,SY)
    ax[1, 0].set_title('P_0 = 1000')
    ax[1,0].set(xlabel='#-iteration', ylabel='Approximation value')
    ax[1,1].plot(MX,MY)
    ax[1, 1].set_title('P_0 = 10000')
    ax[1,1].set(xlabel='#-iteration')
    
    #for axis in ax.flat:
        #axis.set(xlabel='#-iteration', ylabel='Approximation value')
        #axis.label_outer()

     
    
    fig.suptitle('%s' %(figureTitle), fontsize=16)
    fig.tight_layout()


def part_a():
    
            
    a = -10
    b = 9
    tol = 1e-4
    maxIter = 100000
    
    #x = np.linspace(-1, 5.5,100000)
    #y = f1(x)
    
    #plt.plot(x,y)
    printHeader("Bisection Method")
    bisection_answer,b_iterData = bisectionMethod(f1, a, b, tol, maxIter)
    

    
    initial_guess = 5
    printHeader("Newton's Method")
    newtons_answer, n_iterData = newtons_method(f1, df1dx, initial_guess, tol, maxIter)
    printHeader("Secant Method")
    secant_answer, s_iterData = secant_method(f1, -1, 1, tol, maxIter)
    printHeader("Muller's Method")
    mullers_answer,m_iterData = mullers_method(f1, -1, 0, 1, tol, maxIter)
    
    printHeader("\t\tPart B")
    #print(bisection_answer,newtons_answer,secant_answer)
    
    #print(m_iterData.values())
    
    graphConvergence(b_iterData,n_iterData,s_iterData,m_iterData,"Part A Convergence")
    
def part_b():
    a = -10
    b = 9
    tol = 1e-4
    maxIter = 100000
    
    x = np.linspace(-10, 10,100000) #resolution is not enough here. 
    y = f2(x)
    
    plt.plot(x,y)
    printHeader("Bisection Method")
    #bisection_answer,b_iterData = bisectionMethod(f2, a, b, tol, maxIter)
    
    initial_guess = 10000
    printHeader("Newton's Method")
    newtons_answer, n0_iterData = newtons_method(f2, df2dx, 10, tol, maxIter)
    newtons_answer, n1_iterData = newtons_method(f2, df2dx, 100, tol, maxIter)
    newtons_answer, n2_iterData = newtons_method(f2, df2dx, 1000, tol, maxIter)
    newtons_answer, n3_iterData = newtons_method(f2, df2dx, initial_guess, tol, maxIter)
    #printHeader("Secant Method")
    #secant_answer, s_iterData = secant_method(f2, 1, 3, tol, maxIter)
    #printHeader("Muller's Method")
    #mullers_answer,m_iterData = mullers_method(f2, 0, 1, 3, tol, maxIter)
    graphConvergence(n0_iterData,n1_iterData,n2_iterData,n3_iterData,"Newton's method tested at P_0 = 10, 100, 1000, 10000")
    
if __name__ == "__main__":
    #part_a()
    part_b()

    