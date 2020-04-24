import time

import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import pickle
from scipy.linalg import solve_continuous_are

from environment import DoublePendulum


def compute_optimal_control(env, timespan, n_points, r_u=0.01, r_end=10, max_iter=500, save=False, file=None):

    # Defining the variables

    m = GEKKO()

    start = 0
    end = timespan

    m.time = np.linspace(start, end, n_points)

    final = np.zeros(len(m.time))
    final = m.Param(value=final)

    x, theta1, theta2, v, w1, w2 = env.state
    x_g = m.Var(value=x)
    theta1_g = m.Var(value=theta1)
    theta2_g = m.Var(value=theta2)
    v_g = m.Var(value=v)
    w1_g = m.Var(value=w1)
    w2_g = m.Var(value=w2)

    u_g = m.Var(value=0)

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
          l1 * l2 * m2 * m.cos(theta1_g - theta2_g),
          l2**2 * m2]]

    G = [l1 * (m1 + m2) * w1_g**2 * m.sin(theta1_g) +
         m2 * l2 * w2_g**2 * m.sin(w2_g) + u_g,
         -l1 * l2 * m2 * w2_g**2 * m.sin(theta1_g - theta2_g) +
         g * (m1 + m2) * l1 * m.sin(theta1_g),
         l1 * l2 * m2 * w1_g**2 * m.sin(theta1_g - theta2_g) +
         g * m2 * l2 * m.sin(theta2_g)]

    m.Equation(M[0][0]*v_g.dt() + M[0][1]*w1_g.dt() + M[0][2]*w2_g.dt() == G[0])
    m.Equation(M[1][0]*v_g.dt() + M[1][1]*w1_g.dt() + M[1][2]*w2_g.dt() == G[1])
    m.Equation(M[2][0]*v_g.dt() + M[2][1]*w1_g.dt() + M[2][2]*w2_g.dt() == G[2])

    # Define the objective function to minimize

    final_list = np.zeros(len(m.time))
    final_list[-1] = r_end * n_points
    final = m.Param(value=final_list)

    m.Obj(final * x_g**2)
    m.Obj(final * theta1_g**2)
    m.Obj(final * theta2_g**2)
    m.Obj(final * v_g**2)
    m.Obj(final * w1_g**2)
    m.Obj(final * w2_g**2)
    m.Obj(final * u_g**2)

    m.Obj(x_g**2)
    m.Obj(theta1_g**2)
    m.Obj(theta2_g**2)
    m.Obj(v_g**2)
    m.Obj(w1_g**2)
    m.Obj(w2_g**2)

    m.Obj(r_u * u_g**2)

    # Solving the system

    m.options.IMODE = 6
    m.options.MAX_ITER = max_iter
    m.solve()

    if save:
        y_u = [x_g, theta1_g, theta2_g, v_g, w1_g, w2_g, u_g]
        if file:
            with open(file, 'wb') as fp:
                pickle.dump(y_u, fp)
        else:
            with open(f'y_u_solution_{time.time()}', 'wb') as fp:
                pickle.dump(y_u, fp)

    return [x_g, theta1_g, theta2_g, v_g, w1_g, w2_g, u_g]


def simulation(env, y_u, use_feedback=True, only_display=False, Q_fb=np.eye(6), r_fb=0.1, plot_values=True):

    x_g, theta1_g, theta2_g, v_g, w1_g, w2_g, u_g = y_u

    n_points = len(x_g)

    xs = np.zeros(2 * n_points)
    theta1s = np.zeros(2 * n_points)
    theta2s = np.zeros(2 * n_points)
    vs = np.zeros(2 * n_points)
    w1s = np.zeros(2 * n_points)
    w2s = np.zeros(2 * n_points)
    us = np.zeros(2 * n_points)
    feedbacks = np.zeros(2 * n_points)

    tau = env.tau

    times = [tau * i for i in range(2 * n_points)]
    norms = np.zeros(2 * n_points)
    # count = 0
    for t in range(2 * n_points):

        if t < n_points:
            u = u_g[t]
            theo_state = np.array([x_g[t], theta1_g[t], theta2_g[t], v_g[t], w1_g[t], w2_g[t]])
        else:
            u = 0
            theo_state = np.zeros(6)

        if use_feedback:
            A, b = env.get_feedback_matrices(theo_state, u)

            P = solve_continuous_are(A, b, Q_fb, r_fb)
            feedback = b.T @ P / r_fb @ (theo_state - env.state)
            norm = (b.T @ P / r_fb) @ (b.T @ P / r_fb).T
        else:
            feedback = 0
            norm = 0

        # if abs(feedback) > 5000:
        #     feedback = 0
        #     norm = 0

        # if abs((b.T @ P / r_fb) @ (b.T @ P / r_fb).T) > 1e8:
        #     feedback = 0
        #     norm = 0
        # else:
        #     norm = (b.T @ P / r_fb) @ (b.T @ P / r_fb).T

        action = u + feedback
        obs, r, done, info = env._step(action)

        xs[t] = obs[0]
        theta1s[t] = obs[1]
        theta2s[t] = obs[2]
        vs[t] = obs[3]
        w1s[t] = obs[4]
        w2s[t] = obs[5]
        us[t] = action
        feedbacks[t] = feedback
        norms[t] = norm

        # obs, r, done, info = env._step_oc(new_state)

        if done:
            break

    env._reset()
    render_simulation(env, [xs, theta1s, theta2s, vs, w1s, w2s])

    # Ploting the values

    if plot_values:
        fig, axs = plt.subplots(3, 3)

        ax = axs[0, 0]
        ax.plot(times, np.hstack((x_g.value, np.zeros(n_points))), label='Theory')
        ax.plot(times, xs, label='Observations')
        ax.set(xlabel='time', ylabel='x(t)')
        ax.legend()

        ax = axs[0, 1]
        ax.plot(times, np.hstack((theta1_g.value, np.zeros(n_points))), label='Theory')
        ax.plot(times, theta1s, label='Observations')
        ax.set(xlabel='time', ylabel='theta1(t)')
        ax.legend()

        ax = axs[0, 2]
        ax.plot(times, np.hstack((theta2_g.value, np.zeros(n_points))), label='Theory')
        ax.plot(times, theta2s, label='Observations')
        ax.set(xlabel='time', ylabel='theta2(t)')
        ax.legend()

        ax = axs[1, 0]
        ax.plot(times, np.hstack((v_g.value, np.zeros(n_points))), label='Theory')
        ax.plot(times, vs, label='Observations')
        ax.set(xlabel='time', ylabel='v(t)')
        ax.legend()

        ax = axs[1, 1]
        ax.plot(times, np.hstack((w1_g.value, np.zeros(n_points))), label='Theory')
        ax.plot(times, w1s, label='Observations')
        ax.set(xlabel='time', ylabel='w1(t)')
        ax.set_ylim([-20, 20])
        ax.legend()

        ax = axs[1, 2]
        ax.plot(times, np.hstack((w2_g.value, np.zeros(n_points))), label='Theory')
        ax.plot(times, w2s, label='Observations')
        ax.set(xlabel='time', ylabel='w2(t)')
        ax.set_ylim([-20, 20])
        ax.legend()

        ax = axs[2, 0]
        ax.plot(times, feedbacks)
        ax.set(xlabel='time', ylabel='feedback(t)')

        ax = axs[2, 1]
        ax.plot(times, us, color='xkcd:orange', label='With feedback')
        ax.plot(times, np.hstack((u_g.value, np.zeros(n_points))), color='xkcd:azure', label='Theory')
        ax.set(xlabel='time', ylabel='u(t)')
        ax.legend()

        ax = axs[2, 2]
        ax.plot(times, norms)
        ax.set(xlabel='time', ylabel='norm of feedback vector')

        plt.show()


def render_simulation(env, ys, sleep=0, f_render=3):
    xs, theta1s, theta2s, vs, w1s, w2s = ys
    n_points = len(xs)

    for t in range(n_points):
        time.sleep(sleep)
        if t % f_render == 0:
            env._render()
        obs, r, done, info = env._step_oc([xs[t], theta1s[t], theta2s[t], vs[t], w1s[t], w2s[t]])
        if done:
            break


def main():

    time = 10
    n_points = 2000
    tau = time/n_points

    state_init = [0, np.pi, 0, 0, 0, 0]

    # env = DoublePendulum(tau=tau, state_init=state_init)
    # y_u = compute_optimal_control(env, time, n_points, r_u=0.005, r_end=1000,
    #                               max_iter=1000, save=True, file="data/test_3.pk")

    with open("data/solution_pi_zero.pk", 'rb') as fp:
        y_u = pickle.load(fp)

    env = DoublePendulum(tau=tau, state_init=state_init)
    Q = np.diag([1, 1, 1, 1, 1, 1])
    simulation(env, y_u, use_feedback=True, only_display=False, Q_fb=Q, r_fb=1, plot_values=True)


if __name__ == '__main__':
    main()
