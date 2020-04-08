import time

import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

from environment import DoublePendulum

# Defining the variables

m = GEKKO()

start = 0
end = 1
n_point = 1000

m.time = np.linspace(start, end, n_point)
tau = m.time[1]

env = DoublePendulum(tau=tau)

final = np.zeros(len(m.time))
final = m.Param(value=final)

x, theta1, theta2, v, w1, w2 = env.state
x_g = m.Var(value=x)
theta1_g = m.Var(value=theta1)
theta2_g = m.Var(value=theta2)
v_g = m.Var(value=v)
w1_g = m.Var(value=w1)
w2_g = m.Var(value=w2)

u = m.Var(value=0)

# Setting the equations

m.Equation(x_g.dt() == v_g)
m.Equation(theta1_g.dt() == w1_g)
m.Equation(theta2_g.dt() == w2_g)

m0 = env.m
m1 = env.m1
m2 = env.m2
l1 = env.l1
l2 = env.l2
g = env.g

M = [[m0 + m1 + m2,
      l1 * (m1 + m2) * m.cos(theta1_g),
      m2 * l2 * m.cos(theta2_g)],
     [l1 * (m1 + m2) * m.cos(theta1_g),
      l1**2 * (m1 + m2),
      l1 * l2 * m2 * m.cos(theta1_g - theta2_g)],
     [l2 * m2 * m.cos(theta2_g),
      1 * l2 * m2 * m.cos(theta1_g - theta2_g),
      l2**2 * m2]]

G = [l1 * (m1 + m2) * w1_g**2 * m.sin(theta1_g) +
     m2 * l2 * w2_g**2 * m.sin(w2_g) + u,
     -l1 * l2 * m2 * w2_g**2 * m.sin(theta1_g - theta2_g) +
     g * (m1 + m2) * l1 * m.sin(theta1_g),
     l1 * l2 * m2 * w1_g**2 * m.sin(theta1_g - theta2_g) +
     g * m2 * l2 * m.sin(theta2_g)]

m.Equation(M[0][0]*v_g.dt() + M[0][1]*w1_g.dt() + M[0][2]*w2_g.dt() == G[0])
m.Equation(M[1][0]*v_g.dt() + M[1][1]*w1_g.dt() + M[1][2]*w2_g.dt() == G[1])
m.Equation(M[2][0]*v_g.dt() + M[2][1]*w1_g.dt() + M[2][2]*w2_g.dt() == G[2])

# Add the final constraints

m.fix(x_g, pos=n_point-1, val=0.0)
m.fix(theta1_g, pos=n_point-1, val=0)
m.fix(theta2_g, pos=n_point-1, val=0)
m.fix(v_g, pos=n_point-1, val=0.0)
m.fix(w1_g, pos=n_point-1, val=0.0)
m.fix(w2_g, pos=n_point-1, val=0.0)

final_list = np.zeros(len(m.time))
final_list[-1] = 1
final = m.Param(value=final_list)

# Define the objective function to minimize

m.Obj(final*x_g**2)
m.Obj(final*theta1_g**2)
m.Obj(final*theta2_g**2)
m.Obj(final*v_g**2)
m.Obj(final*w1_g**2)
m.Obj(final*w2_g**2)

m.Obj(x_g**2)
m.Obj(theta1_g**2)
m.Obj(theta2_g**2)
m.Obj(v_g**2)
m.Obj(w1_g**2)
m.Obj(w2_g**2)

m.Obj(0.0001*u**2)

# Solving the system

m.options.IMODE = 6
m.solve()

# Simulation

env = DoublePendulum(tau=tau, max_iter=1000000)

xs = np.zeros(n_point)
theta1s = np.zeros(n_point)
theta2s = np.zeros(n_point)
vs = np.zeros(n_point)
w1s = np.zeros(n_point)
w2s = np.zeros(n_point)

for t in range(n_point):
    if t % 5 == 0:
        env._render()

    action = u[t]
    obs, r, done, info = env._step(action)
    time.sleep(0.001)
    state = env.state

    xs[t] = state[0]
    theta1s[t] = state[1]
    theta2s[t] = state[2]
    vs[t] = state[3]
    w1s[t] = state[4]
    w2s[t] = state[5]

    if done:
        break
for t in range(n_point//5):
    if t % 5 == 0:
        env._render()
    action = 0
    time.sleep(0.001)
    obs, r, done, info = env._step(action)
    if done:
        break

# Ploting the values

fig, axs = plt.subplots(3, 3)

ax = axs[0, 0]
ax.plot(m.time, x_g.value, label='Theory')
ax.plot(m.time, xs, label='Observations')
ax.set(xlabel='time', ylabel='x(t)')
ax.legend()

ax = axs[0, 1]
ax.plot(m.time, theta1_g.value, label='Theory')
ax.plot(m.time, theta1s, label='Observations')
ax.set(xlabel='time', ylabel='theta1(t)')
ax.legend()

ax = axs[0, 2]
ax.plot(m.time, theta2_g.value, label='Theory')
ax.plot(m.time, theta2s, label='Observations')
ax.set(xlabel='time', ylabel='theta2(t)')
ax.legend()

ax = axs[1, 0]
ax.plot(m.time, v_g.value, label='Theory')
ax.plot(m.time, vs, label='Observations')
ax.set(xlabel='time', ylabel='v(t)')
ax.legend()

ax = axs[1, 1]
ax.plot(m.time, w1_g.value, label='Theory')
ax.plot(m.time, w1s, label='Observations')
ax.set(xlabel='time', ylabel='w1(t)')
ax.legend()

ax = axs[1, 2]
ax.plot(m.time, w2_g.value, label='Theory')
ax.plot(m.time, w2s, label='Observations')
ax.set(xlabel='time', ylabel='w2(t)')
ax.legend()

ax = axs[2, 1]
ax.plot(m.time, u.value)
ax.set(xlabel='time', ylabel='u(t)')

plt.show()
