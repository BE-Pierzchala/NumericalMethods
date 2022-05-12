# midpoint integration

def MidInt(a,b,n,f):
    
    h = (b-a)/n
    result = 0
    for i in range(n):
        result += f( a + (i+0.5)*h )
        
    return result*h

def TrapInt(a,b,n,f):
    h = (b-a)/n
    result = 0
    for i in range(n):
        result += f( a + i*h ) + f( a + (i+1)*h )
        
    return result*h/2

def SimpInt(a,b,n,f):
    
    if n%2 == 0:
        n += 1 # Simpson rule takes odd number of points
    
    h = (b-a)/(n-1)
    
    result = 0
    
    x = a + h

    for i in range( n//2 ):
        result += f(x - h) + 4*f(x) + f(x+h)
        x += 2*h
        

    
    return result*h/3