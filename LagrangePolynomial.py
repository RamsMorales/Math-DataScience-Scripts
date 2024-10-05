#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:13:14 2024

@author: ramsonmunoz
"""


def LFactor(dataset: dict, evalPoint: float, x_k: float) ->float:
    
    denominator = 1
    numerator = 1

    for x in dataset.keys():

        if x != x_k:
            denominator = denominator * (x_k - x)
            numerator = numerator * (evalPoint - x)
 
                
    result = float(numerator) / float(denominator)
            
    return result
            

def generateLagrangePolynomial(dataset: dict, evalPoint: float, degree: int) -> float:
    
    
    result = 0
    
    for i in range(degree + 1):
        
        x_k = list(dataset.keys())[i]
        
        result += dataset[x_k] * LFactor(dataset, evalPoint, x_k)
        
    return result

def func(x: float):
    
    return 1 / x


def main():
    
    dataset = {2:func(2),2.75:func(2.75),4:func(4)}
    
    print(generateLagrangePolynomial(dataset, 3,len(dataset)-1))
    
    
if __name__ == "__main__":
    main()