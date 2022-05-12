# module to calculate Gauss-Legendre quadrature and test corectness of values
#   of found roots and weights

import numpy as np
import matplotlib.pyplot as plt
from Legendre import rwP,P

def LegInt(a,b,n,f):
    # funtion to calculate Gaussian-Legendre quadrature of f using n points in the
    # interval a,b
    
    roots, weights = rwP(n) # find roots and weights of nth Legendre Polynomial
    
    scale = (b-a)/2 # scale factor due to (a,b) -> (-1,1)
    result = 0    
    for i in range( len(roots) ):
        roots[i] *= scale
        roots[i] += (b+a)/2 # other scale factor for roots
        result += weights[i]*f(roots[i])
        
    return result*scale

def LegendreTest(n):
    # finds roots  of n-th Legendre polynomial and plots them 
    #   (their values in red and weights in black) on the scaled down plot of derivative 
    #    On the second plot are shown log10 of values at roots
    
    x = np.linspace(-1,1,100) # plotting range                               #   recursive formula would divide by 0
    y = P(n,x) # get values for plot of Pn
    
    r,w = rwP(n) # get roots
    yr = P(n,r) # value of Pn at roots
    
    plt.figure()
    
    plt.plot(x,y) # plot scaled down Pn' so weights are visible
    plt.scatter(r,yr, c = 'r') # plot zeros and their values
    plt.scatter(r,w, c='k') # plot zeros and their weights
    plt.grid()

    plt.figure()
    # check if some values at roots are exactly 0, if so put -20 instead of
    #   taking log10 of 0
    for i in range( len(r) ):
        if yr[i] != 0:
            yr[i] = np.log10(abs(yr[i]))
        else:
            yr[i] = -20
            
    plt.plot(r, yr)
    return 


