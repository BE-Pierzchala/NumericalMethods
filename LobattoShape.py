# Generate Lobatto shape functions in meaning of Kohn log method
# by Manolopoulos
import numpy as np
import matplotlib.pyplot as plt
from Legendre import rwP_der

def LobPlot(n,a,b):
    # function to plot 0,1...n lobatto shape functions
    
    # find roots of Pn'
    roots = LobShapeRoots(n-1,b)
    roots = np.flip(roots)   
    # flip the roots because that's how the man defined it
    
    # plotting range
    x = np.linspace(0,b,100)
    
    # plot all functions, even ones with solid line odd with dashed
    for i in range(n):
        line = '-'
        if i%2 == 1:
            line = '--'
        plt.plot(x, LobShapeFun(i,x,roots), ls=line)
    plt.grid()
    plt.xlabel('r')
    plt.ylabel('$ u_i(r) $')
    plt.title('Lobatto shape functions for M = ' + str(n) )
    
#    plt.savefig('lobshape', dpi = 500)
    return 

def LobShapeRoots(n,b):
    # find roots for lobatto shape functions in (a,b) region
    a = 0
    
    roots = np.zeros(n+2)
    
    old_roots, weights = rwP_der(n+1) # find roots and weights of Pn+1'
    scale = (b-a)/2 # scale factor due to (a,b) -> (-1,1) and weight
    
    roots[0] = a
    roots[-1] = b # this needs to be scaled
    
    for i in range( len(old_roots) ):
        roots[i+1] = scale*old_roots[i]
        roots[i+1] += (b+a)/2 # other scale factor for roots
 
    return roots

def LobShapeFun(j, x, roots):
    # n defines how many roots, k which root function it is
    # x is point at which it is to be evaluated
    # this just follows formula
    
    result = 1
    
    for i in range( len(roots) ):
        if i == j:
            continue
        
        result *= (x - roots[i])/(roots[j] - roots[i])
    
    return result