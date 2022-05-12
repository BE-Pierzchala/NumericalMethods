# module to calculate Gauss-Lobatto quadrature and test corectness of values
#   of found roots and weights

import numpy as np
import matplotlib.pyplot as plt
from Legendre import rwP_der, P_der

def LobInt(a,b,n,f):
    # funtion to calculate Gaussian-Lobatto quadrature of f using n points in the
    # interval a,b
    
    roots, weights = rwP_der(n-1) # find roots and weights of nth Legendre Polynomial
    scale = (b-a)/2 # scale factor due to (a,b) -> (-1,1) and weight
    result =  (f(a) + f(b))*2/(n*(n-1))    
    for i in range( len(roots) ):
        roots[i] *= scale
        roots[i] += (b+a)/2 # other scale factor for roots
        result += weights[i]*f(roots[i])
        
    return result*scale



def LobattoTest(n):
    # finds roots of derivative of n-th Legendre polynomial and plots them 
    #   (their values in red and weights in black) on the scaled down plot of derivative 
    #    On the second plot are shown log10 of values at roots
    
    x = np.linspace(-0.99,0.99,100) # exclude -+1 as derivative there through divide by 0                                #   recursive formula would divide by 0
    y = P_der(n,x) # get values for plot of Pn'
    m = max(y) # find max to scale down plot
    
    r,w = rwP_der(n) # get roots
    yr = P_der(n,r) # value of Pn' at roots
    
    
    plt.figure()
    
    plt.plot(x,y/m) # plot scaled down Pn' so weights are visible
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