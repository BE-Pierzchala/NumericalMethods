"""
Module for value and roots (and weights for quads) finding of Legendre polynomials and its derivatives
"""

import numpy as np

def P(n,x):
    """
    Generate value of Pn (n-th Legendre polynomial) at x using Bonnet's recursion formula
    """
    Pn = 1. ;  Pn_1 = 0. 
    # Pn_1 is P (n-1), Pn_2 is P(n-2)
    for j in range(1, n + 1):
        # n + 1, as after thr first iteration result is P1
        # index starts at 1 so formula has original form
        Pn_2 = Pn_1;   Pn_1 = Pn 
        Pn = ((2*j-1)*x*Pn_1 - (j-1)*Pn_2)/j
    return Pn


def P_derivative(n,x):
    """
    Calculate value of the derivative of Pn at x using other recursive formula
    """
    return n*( x*P(n,x) - P(n-1, x) )/( x*x - 1 )



def rootsWeightsP(n):
    """
    Function to find roots of the Legendre polynomial of order n ( Pn )
    and the associated weights for Gaussian-Legendre quadrature.

    Starting points are taken from Francesco Tricomi approximation and
    code closes down on zeros using Newton-Rhapson until difference between
    found root estimates is smaller than eps
    """
    roots = np.zeros(n)
    weights = np.zeros(n)
    eps = 1e-10
    
    for i in range(1, (n+1)//2  + 1 ):
        # approximation only gives the positive roots, hence divison by 2
        # get initial guesses for values of roots using Francesco Tricomi approximation
        r_new = (1 - 1/(8*n**2) + 1/(8*n**3)) * np.cos( np.pi *(4*i-1)/(4*n+2) )
        r_old = 1
          
        # close down on better approx. of roots using N-R until eps accuracy
        Pn_derivative = -1
        while( abs(r_new - r_old) > eps):
            Pn = P(n,r_new)
            Pn_derivative = P_derivative(n, r_new)

            r_old = r_new
            # actual N-R step
            r_new = r_new - Pn/Pn_derivative

        # store calculated root
        roots[i-1] = -r_new
        roots[n-i] = r_new    
        # store asociated weight
        weights[i-1] = weights[n-i] = 2/( (1 - r_new**2) * Pn_derivative**2)
    
    checkUniqueness(roots) # check uniqueness of found roots
    return roots, weights

def rootsWeightsP_derivative(n):
    """
    function to find roots of the Legendre polynomial's derivative of order n
    and the asociated weights for Gaussian-Lobatto quadrature
    """

    m = n - 1 # one root less than Pn
    
    roots = np.zeros(m)
    weights = np.zeros(m)
    eps = 1e-10  # precision for roots
    
    for i in range(1, m//2 + 1 ):
        # approximation only gives the positive roots, hence divison by 2
        # get initial guesses for values of roots of Pn using Francesco Tricomi approximation

        # use midpoint between roots of Pn as estimate for zero of Pn'
        r_left  = (1 - 1/(8*n**2) + 1/(8*n**3)) * np.cos( np.pi *(4*i-1)/(4*n+2) )
        r_right = (1 - 1/(8*n**2) + 1/(8*n**3)) * np.cos( np.pi *(4*(i+1)-1)/(4*n+2) )
        
        r_new = 0.5*(r_left + r_right)
        r_old = 1

        Pn = -1
        # close down on better approx. of roots using N-R until eps accuracy
        while( abs(r_new -r_old) > eps):
            Pn = P(n, r_new)
            Pn_derivative = P_derivative(n, r_new)
            # Calculate second order's derivative
            Pn_derivative2 = (2*r_new*Pn_derivative - n*(n+1)*Pn)/(1 - r_new**2)
            
            r_old = r_new
            r_new -= Pn_derivative/Pn_derivative2
            
        # store calculated root   
        roots[i-1] = -r_new
        roots[m-i] = r_new  
        
        # store asociated weight
        weights[i-1] = 2/( n*(n+1)*Pn**2)
        weights[m-i] = 2/( n*(n+1)*Pn**2)
        
    # if n is odd then Pn' has a root there
    if n%2 == 0:
        weights[m//2] = 2/( n*(n+1)*P(n,0)**2)

    checkUniqueness(roots)    
    return roots, weights

def checkUniqueness(toTest):
    """
    Function to check if elements in an 'odd' list are unique
    """
    if len(np.unique(toTest)) != len(toTest):
        raise Exception('List is not unique')
