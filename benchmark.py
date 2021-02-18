import numpy as np
from numpy.random import normal
def benchmark_func(array):
    fitness = -sphere(array)
    return fitness + normal() # return a noisy output of the function


def schwefel(array):
    sum = 0
    fitness = 0
    for x in array:
        sum = sum + x * np.sin(np.sqrt(np.abs(x)))
    fitness = 418.9829 * len(array) - sum
    return fitness

def sphere(array):
    fitness = 0
    for i in range(len(array)):
        fitness = fitness + array[i]**2
    return fitness

def rastrigin(array):
    sum = 0
    fitness = 0
    for x in array:
        sum = sum + x**2 - 10 * np.cos(2 * np.pi * x)
    fitness = 10.0 * len(array) + sum
    return fitness

def eggholder(array):
    z = - (array[1] + 47) * np.sin(np.sqrt(abs(array[1] + (array[0]/2) +47))) - array[0] *np.sin(np.sqrt(abs(array[0] - (array[1]+47))))
    return z

def matyas(array):  
    z = 0.26 * (array[0]**2 + array[1]**2) - (0.48 * array[0] * array[1])
    return z

def levin13(array):
    x = array[0]
    y = array[1]
    z = np.sin(3 * np.pi * x)**2 + ((x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2) + ((y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2) ))
    return z

def three_deceptive(array):
    count = len([e for e in array if e == 1])
    
    if count == 0 return .9
    if count == 1 return .8
    if count == 2 return 0
    return 1

def trap5(array):
    count = len([e for e in array if e == 1])    
    
    if count < 5 return 4 - count
    return 5

def six_bipolar(array):
    count = len([e for e in array if e == 1])    
    three_deceptive(abs(3-count))
    
