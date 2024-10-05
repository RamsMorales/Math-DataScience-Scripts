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
        #print(x)
        if x != x_k:
            denominator = denominator * (x_k - x)
            numerator = numerator * (evalPoint - x)
 
                
    result = float(numerator) / float(denominator)
    
   # print(result)
            
    return result
            

def generateLagrangePolynomial(dataset: dict, evalPoint: float) -> float:
    
    
    result = 0
    
    for x_k in dataset.keys():
        
        result += dataset[x_k] * LFactor(dataset, evalPoint, x_k)
        
    return result

def func(x: float):
    
    return 1 / x


def main():
    
    dataset = {2:func(2),2.75:func(2.75),4:func(4)}
    
    print(generateLagrangePolynomial(dataset, 3))
    
    
if __name__ == "__main__":
    main()