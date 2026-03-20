from scipy.optimize import minimize
import numpy as np
from shapely.geometry import Point
from objective.G import G
from objective.gradient import grad_G
from utils.sampling import sample_points_in_polygon

#hessp=lambda x, p: hessp(x, p, r, region),


def solve_inner_fixed_r(X0 ,region, r, m, max_iter=1700):
    

    x0 = X0.reshape(-1)
    def f(x):
        return G(x, r, region) 

    def grad(x):
        return grad_G(x, r, region)

    def hessp(x, p, r, region, eps=1e-4):
        return (grad_G(x + eps*p, r, region) - grad_G(x, r, region)) / eps
 
    res = minimize(
        f,
        x0,
        jac=grad,
        method='L-BFGS-B',
        options={"maxcor": 150, "gtol": 1e-5, "ftol": 1e-9, "maxls": 40}
    )
    return res.x, res.fun