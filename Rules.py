"""
Script with simple rules for numerical integration
"""
import LegendrePolynomial


def midpoint(start: float, end: float, num_steps: int, function):
    """
    Function to calculate an integral of start function using Midpoint rule.
    """

    step = (end - start) / num_steps
    result = 0
    for i in range(num_steps):
        result += function(start + (i + 0.5) * step)

    return result * step


def trapezoid(start: float, end: float, num_steps: int, function):
    """
    Function to calculate an integral of start function using Trapezoid rule.
    """
    step = (end - start) / num_steps
    result = 0
    for i in range(num_steps):
        result += function(start + i * step) + function(start + (i + 1) * step)

    return result * step / 2


def simpsons(start: float, end: float, num_steps: int, function):
    """
    Function to calculate an integral of start function using Simpson's rule.
    """
    if num_steps % 2 == 0:
        num_steps += 1  # Simpson rule takes an odd number of points

    step = (end - start) / (num_steps - 1)
    result = 0
    x = start + step
    for i in range(num_steps // 2):
        result += function(x - step) + 4 * function(x) + function(x + step)
        x += 2 * step

    return result * step / 3


def Legendre(start, end, num_steps, function):
    """
    funtion to calculate Gaussian-Legendre quadrature of function using num_steps points in the
    interval start,end
    """

    roots, weights = LegendrePolynomial.rootsWeightsP(num_steps)  # find roots and weights of nth Legendre Polynomial

    scale = (end - start) / 2  # scale factor due to (start,end) -> (-1,1)
    result = 0
    for i in range(len(roots)):
        roots[i] *= scale
        roots[i] += (end + start) / 2  # other scale factor for roots
        result += weights[i] * function(roots[i])

    return result * scale


def Lobatto(start, end, num_steps, function):
    """
    funtion to calculate Gaussian-Lobatto quadrature of function using num_steps points in the
    interval start,end
    """

    roots, weights = LegendrePolynomial.rootsWeightsP_derivative(
        num_steps - 1)  # find roots and weights of nth Legendre Polynomial
    scale = (end - start) / 2  # scale factor due to (start,end) -> (-1,1) and weight
    result = (function(start) + function(end)) * 2 / (num_steps * (num_steps - 1))
    for i in range(len(roots)):
        roots[i] *= scale
        roots[i] += (end + start) / 2  # other scale factor for roots
        result += weights[i] * function(roots[i])

    return result * scale
