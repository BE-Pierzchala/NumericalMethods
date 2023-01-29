def RKE_ODE1(y_derivative, y, t, dt):
    """
    Runge-Kutta algorithm for solving 1st order ODE's
    """
    
    k1 = dt*y_derivative(y,t)
    k2 = dt*y_derivative(y + dt/2, t + k1/2)
    k3 = dt*y_derivative(y + dt/2, t + k2/2)
    k4 = dt*y_derivative(y + dt, t + k3)
    
    return y + (k1/6 + k2/3 + k3/3 + k4/6)


def RKE_ODE2(function, x, y, z, dt):
    """
    Runge-Kutta algorithm for solving 2nd order ODE's
    """
    helper_function = lambda x, y, z: z
    k1 = dt * helper_function(x, y, z)
    l1 = dt * function(x, y, z)

    k2 = dt * helper_function(x + dt / 2, y + k1 / 2, z + l1 / 2)
    l2 = dt * function(x + dt / 2, y + k1 / 2, z + l1 / 2)

    k3 = dt * helper_function(x + dt / 2, y + k2 / 2, z + l2 / 2)
    l3 = dt * function(x + dt / 2, y + k2 / 2, z + l2 / 2)

    k4 = dt * helper_function(x + dt, y + k3, z + l3)
    l4 = dt * function(x + dt, y + k3, z + l3)

    y_next = y + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
    z_next = z + (l1 / 6 + l2 / 3 + l3 / 3 + l4 / 6)

    return y_next, z_next
    
    