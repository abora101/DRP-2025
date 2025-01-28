import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

w = 2 * np.pi

d = 0.25

A = np.array([[0, 1], [-w ** 2, -2 * d * w]])

dt = 0.01
T = 10

x0 = [2, 0]

num_time_slices = int(T / dt)
time_slices = np.linspace(0, T, num_time_slices)
xF = np.zeros((2, num_time_slices))

xF[:, 0] = x0
for k in range(num_time_slices - 1):
    xF[:, k+1] = (np.eye(2) + dt * A) @ xF[:, k]

xB = np.zeros((2, num_time_slices))
xB[:, 0] = x0
for k in range(num_time_slices - 1):
    xB[:, k+1] = np.linalg.pinv(np.eye(2) - A * dt) @ xB[:, k]

def ode_func(t, x):
    return A @ x

ivp_solution = solve_ivp(ode_func, (0, T), x0, t_eval = time_slices)
xRK4 = ivp_solution.y

plt.plot(time_slices, xF[0, :], 'k')
plt.grid(True)
plt.title("x(t)")
plt.plot(time_slices, xB[0, :], 'b')
plt.plot(time_slices, xRK4[0, :], 'r')
plt.legend(["Forward Euler", "Backward Euler", "Runge-Kutta"])
plt.show()