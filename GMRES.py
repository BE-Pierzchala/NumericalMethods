import numpy as np


def AMGS(A,m,b):
# Arnoldi-Modified Gram-Schmidt withing m-th Kyrlov space of A,b
#   creates an orthonormal basis in that space and prepares H and e,
#   matrix and vector, needed for GMRES. They undergo unitary transformation
#   to make hessenbergian matrix H upper triangular


# A is matrix generating Krylov subspace
# m defines which Krylov subspace
# b is a starting vector
    
#           SETUP; reserve memory and so on
#=============================================================================
    n = len(A) # I assume A is n x n matrix
    
    if m > n :
        m = n
        print('m bigger than size of A does not really make sense')
    
    Vs = np.zeros( (n,m+1) ) # matrix for storing m n-vectors in columns
    e = np.zeros(m+1)

    if len(b) != n:
        raise Exception( 'b vector has to be of length '+ str(n) )
    beta = np.linalg.norm(b)
    Vs[:,0] = b/beta
    
    e[0] = beta
    
    H = np.zeros( (m+1,m) ) # Hessenberg matrix
    
    tolerance = 1e-8
    
    Q = np.identity(m+1)
#=============================================================================    

    
    for i in range(m):
        w = np.dot( A, Vs[:,i] ) # generate next vector
        
        for j in range(i+1):    #include i == j
            
            H[j,i] = np.dot(w, Vs[:,j]) # Store those values for later use
            w -= H[j,i]*Vs[:,j] # This is just Gram-schmidt step

        # bring new column up to date with previous rotations
        H[:,i] = np.dot(Q,H[:,i] )   

        H[i+1,i] = np.linalg.norm(w)
        
        if abs( H[i+1,i] ) < tolerance:
        # break if new vector is already in the basis
        # don't return empty space
            Vs = Vs[:,0:(i+2)]
            H = H[0:(i+2), 0:(i+1)]
            e = e[0:(i+2)]
            break
        
        # Write in orthonormal new vector into basis
        Vs[:,i+1] = w/H[i+1,i]
        
        
        # Make H upper triangular at each step
        Q_i = np.identity(m+1) # prepare for triangulation manipulation
        s_i = H[i+1,i] / np.sqrt( H[i,i]**2 + H[i+1,i]**2 )
        c_i = H[i,i] / np.sqrt( H[i,i]**2 + H[i+1,i]**2 )

        Q_i[i,i] = c_i
        Q_i[i+1,i+1] = c_i
        Q_i[i,i+1] = s_i
        Q_i[i+1,i] = -s_i
      
        # rotate H and e with new rotation
        H = np.dot(Q_i, H)
        e = np.dot(Q_i,e)
        
        # store for updating new columns
        Q = np.dot(Q_i, Q)


    # last row is unnecessary 
    H = H[0:(len(H) - 1), :]    
    # replace elements smaller than tolerance with 0
    H = np.where(abs(H) < tolerance, 0, H)
    return Vs, H, e


def BackSub(A,b):
# A has to be upper triangular, solves Ax = b using gaussian elimination
# using back substituiton 
#
# It's rather self-axplanatory

    n = len(A)
    sol = np.zeros(n)
    
    for i in range(1, n+1):

        sol[-i] = b[-i]/A[-i,-i]
        
        for j in range(1, i): # don't include i == j
            
            sol[-i] -= sol[-j]* A[-i,-j]/A[-i,-i]
            

    return sol


def GMRES(A,b,m):
# This function is an m-step GMRES to solve Ax=b
# returns x' and ||b-Ax'|| where x is found solution
    
    # get orthonormal basis, hessenbergian matrix and transformed (b,0,0...)'
    vs,H,e = AMGS(A,m,b)
    
    # This is just how life goes really, last element is residue
    res = e[-1]
    # cut it out
    e = e[0:len(e)-1]
    
    # solve H y1 = e using back substitution
    y = BackSub(H,e)
    
    #get solution, all vs except the last one
    sol = np.dot(vs[:,0:-1], y) 
    
    return sol,res


if __name__ == '__main__':
    A = [ [1,2,1,-1], [3,2,4,4], [4,4,3,4], [2,0,1,5]]
    b = [5, 16,22,15]
    m = 4
    
    sol_exact = np.linalg.solve(A,b)
    res_exact = np.linalg.norm( b - np.dot(A, sol_exact))
    
    print( 'exact solution =', sol_exact  )
    print('exact residue =', res_exact)
    
    sol_gmres,res_gmres = GMRES(A,b,m)
    
    print('GMRES solution =', sol_gmres )
    print('GMRES residue =', res_gmres)
