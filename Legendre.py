import numpy as np
# module for value and roots (and weights for quads) finding of Legendre 
#   polynomials and its derivatives

def P(n,x):
    # generate value of Pn (n-th Legendre polynomial) at x 
    #   using Bonnet's recursion formula
    Pn = 1. ;  Pn_1 = 0. 
    # Pn_1 is P (n-1), Pn_2 is P(n-2)
    for j in range(1, n + 1):
        # n + 1 because after first iteration result is P1
        #   index starts at 1 so formula has original form
        Pn_2 = Pn_1;   Pn_1 = Pn 
        Pn = ((2*j-1)*x*Pn_1 - (j-1)*Pn_2)/j
    return Pn

def P_der(n,x):
    # Function to generate value of Pn' at x
    
    # generate value of Pn at x using Bonnet's recursion formula
    Pn = 1. ;  Pn_1 = 0. 
    # Pn_1 is P (n-1), Pn_2 is P(n-2)
    for j in range(1, n + 1):
        # n + 1 because after first iteration result is P1
        Pn_2 = Pn_1;   Pn_1 = Pn 
        Pn = ((2*j-1)*x*Pn_1 - (j-1)*Pn_2)/j
    # calculate value of derivative at x using other recursive formula
    Pn_der = n*( x*Pn - Pn_1 )/( x*x - 1 )  
    return Pn_der


def rwP(n):
# function to find roots of the Legendre polynomial of order n ( Pn )
#   and the asociated weights for Gaussian-Legendre quadrature.
#   
#   Starting points are taken from Francesco Tricomi approximation and 
#   code closes down on zeros using Newton-Rhapson until difference between
#   found root estimates is smaller than eps

    roots = np.zeros(n)
    weights = np.zeros(n)
    eps = 1e-10
    
    for i in range(1, (n+1)//2  + 1 ):
        # approximation only gives the positive roots, hence divison by 2
        # get initial guesses for values of roots using Francesco Tricomi approximation
        r_new = (1 - 1/(8*n**2) + 1/(8*n**3)) * np.cos( np.pi *(4*i-1)/(4*n+2) )
        r_old = 1
          
        # close down on better approx. of roots using N-R until eps accuracy
        while( abs(r_new -r_old) > eps):
            
            # generate value of Pn at r_new using Bonnet's recursion formula
            Pn = 1 ;  Pn_1 = 0  
            # Pn_1 is P (n-1), Pn_2 is P(n-2)
            for j in range(1, n+1):
                # n + 1 because after first iteration result is P1
                Pn_2 = Pn_1;   Pn_1 = Pn 
                Pn = ((2*j-1)*r_new*Pn_1 - (j-1)*Pn_2)/j
                
            # get derivative of Pn using other recursion formula     
            Pn_der = n*( r_new*Pn - Pn_1 )/( r_new*r_new - 1 )  
            r_old = r_new
            r_new = r_new - Pn/Pn_der # actual N-R step

        # store calculated root
        roots[i-1] = -r_new
        roots[n-i] = r_new    
        # store asociated weight
        weights[i-1] = 2/( (1 - r_new**2) * Pn_der**2)
        weights[n-i] = 2/( (1 - r_new**2) * Pn_der**2)
    
    checkUniq(roots) # check uniqueness of found roots
    return roots, weights

def rwP_der(n):
    # function to find roots of the Legendre polynomial's derivative of order n
    # and the asociated weights for Gaussian-Lobatto quadrature
    m = n - 1 # one root less than Pn
    
    roots = np.zeros(m)
    weights = np.zeros(m)
    eps = 1e-10  # precision for roots
    
    for i in range(1, m//2 + 1 ):
        # approximation only gives the positive roots, hence divison by 2
        #   get initial guesses for values of roots of Pn using Francesco Tricomi approximation

        # use midpoint between roots of Pn as estimate for zero of Pn'
        r_left  = (1 - 1/(8*n**2) + 1/(8*n**3)) * np.cos( np.pi *(4*i-1)/(4*n+2) )
        r_right = (1 - 1/(8*n**2) + 1/(8*n**3)) * np.cos( np.pi *(4*(i+1)-1)/(4*n+2) )
        
        r_new = 0.5*(r_left + r_right)
        r_old = 1
        
        # close down on better approx. of roots using N-R until eps accuracy
        while( abs(r_new -r_old) > eps):
            
            # generate value of Pn at r_new using Bonnet's recursion formula
            Pn = 1 ;  Pn_1 = 0  
            # Pn_1 is P (n-1), Pn_2 is P(n-2)
            for j in range(1, n+1):
                # n + 1 because after first iteration result is P1
                Pn_2 = Pn_1;   Pn_1 = Pn 
                Pn = ((2*j-1)*r_new*Pn_1 - (j-1)*Pn_2)/j
            
            Pn_der = n*(Pn_1 - r_new*Pn)/(1 - r_new**2) #Pn'
            Pn_der2 = (2*r_new*Pn_der - n*(n+1)*Pn)/(1 - r_new**2) #Pn''
            
            r_old = r_new
            r_new -= Pn_der/Pn_der2
            
        # store calculated root   
        roots[i-1] = -r_new
        roots[m-i] = r_new  
        
        # store asociated weight
        weights[i-1] = 2/( n*(n+1)*Pn**2)
        weights[m-i] = 2/( n*(n+1)*Pn**2)
        
    # if n is odd then Pn' has a root there
    if n%2 == 0:
        weights[m//2] = 2/( n*(n+1)*P(n,0)**2)

    checkUniq(roots)    
    return roots, weights

def checkUniq(xs):
    # function to check if elements in an 'odd' list are unique
    m = (len(xs)+1)//2 # half point through list
    
    for i in range(m):
        for j in range(i+1,m):
            if abs( xs[i] - xs[j]) < 1e-10 :
                print('list not unique')
            
    return