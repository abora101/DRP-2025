import math
from scipy.integrate import solve_ivp
import numpy as np

f = lambda t, x: x

def get_hill_activated_with_removal(beta, K, n, alpha):
    return lambda t, x,: beta * x ** n / (K ** n + x ** n) - alpha * x

def forward_euler(f, time_interval, x0, steps):
    dt = (time_interval[1]-time_interval[0])/steps
    x = x0
    for i in range(steps):
        x = x + dt * f(i*dt+time_interval[0], x)
    return x

fe_solution = forward_euler(f, (0, 10), 3, 1_000_000_000)
rk_solution = solve_ivp(f, (0, 10), np.array([3]), max_step = 0.01).y[0, -1]
print(fe_solution) 
print(rk_solution)
print(3 * math.e ** 10)

#print(forward_euler(get_hill_activated_with_removal(1.0, 0.5, 4, 0.75), (0, 1), 0, 100))
#print(solve_ivp(get_hill_activated_with_removal(1.0, 0.5, 4, 0.75), (0, 1), np.array([0]), max_step = 0.01).y[0, -1])

"""
66079.39408045165
66079.39738443829
66079.39738442011
"""