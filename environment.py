import logging

import gym
import numpy as np
from scipy.integrate import ode
from gym.envs.classic_control import rendering

from utils import normalize_angle

logger = logging.getLogger(__name__)


class DoublePendulum(gym.Env):

    def __init__(self, display=None, max_iter=10000, tau=0.02):
        self.g = 9.8  # gravity constant
        self.m = 50  # mass of cart
        self.m1 = 0.001  # mass of sphere 1
        self.m2 = 0.001  # mass of sphere 2
        self.l1 = 1  # length of pole 1
        self.l2 = 1  # length of pole 2

        self.tau = tau  # seconds between state updates
        self.counter = 0  # used to stop the simulation

        self.x_threshold = 15

        self._reset()
        self.viewer = None

        self.max_iter = max_iter

        self.display = display
        self.action_space = gym.spaces.Discrete(2)  # to update

    def get_derivate_function(self):
        """
        Returns the function that gives the derivate of the state. This is
        necessary to don't pass the parameters of the system as parameters
        of the function, which must have a given signature for the solver.
        """

        m = self.m
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g

        def derivate_func(t, state, u):

            theta1 = state[1]
            theta2 = state[2]
            v = state[3]
            w1 = state[4]
            w2 = state[5]

            M = np.array([[m + m1 + m2,
                           l1 * (m1 + m2) * np.cos(theta1),
                           m2 * l2 * np.cos(theta2)],
                          [l1 * (m1 + m2) * np.cos(theta1),
                           l1**2 * (m1 + m2),
                           l1 * l2 * m2 * np.cos(theta1 - theta2)],
                          [l2 * m2 * np.cos(theta2),
                           l1 * l2 * m2 * np.cos(theta1 - theta2),
                           l2**2 * m2]])

            G = np.array([l1 * (m1 + m2) * w1**2 * np.sin(theta1) +
                          m2 * l2 * w2**2 * np.sin(w2),
                          -l1 * l2 * m2 * w2**2 * np.sin(theta1 - theta2) +
                          g * (m1 + m2) * l1 * np.sin(theta1),
                          l1 * l2 * m2 * w1**2 * np.sin(theta1 - theta2) +
                          g * m2 * l2 * np.sin(theta2)])

            end_state_dot = np.linalg.inv(M) @ (G + np.array([u, 0, 0]))

            state_dot = np.concatenate((np.array([v, w1, w2]), end_state_dot))

            return state_dot

        return derivate_func

    def get_feedback_matrices(self, theo_state, u):
        """
        Returns the function that gives the derivate of the state. This is
        necessary to don't pass the parameters of the system as parameters
        of the function, which must have a given signature for the solver.
        """

        m = self.m
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g

        theta1 = theo_state[1]
        theta2 = theo_state[2]
        w1 = theo_state[4]
        w2 = theo_state[5]

        M = np.array([[m + m1 + m2,
                       l1 * (m1 + m2) * np.cos(theta1),
                       m2 * l2 * np.cos(theta2)],
                      [l1 * (m1 + m2) * np.cos(theta1),
                       l1**2 * (m1 + m2),
                       l1 * l2 * m2 * np.cos(theta1 - theta2)],
                      [l2 * m2 * np.cos(theta2),
                       l1 * l2 * m2 * np.cos(theta1 - theta2),
                       l2**2 * m2]])

        M_inv = np.linalg.inv(M)

        dM_theta1 = np.array([[0,
                               -l1 * (m1 + m2) * np.sin(theta1),
                               0],
                              [-l1 * (m1 + m2) * np.sin(theta1),
                               0,
                               -l1 * l2 * m2 * np.sin(theta1 - theta2)],
                              [0,
                               -l1 * l2 * m2 * np.sin(theta1 - theta2),
                               0]])

        dM_theta2 = np.array([[0,
                               0,
                               -m2 * l2 * np.sin(theta2)],
                              [0,
                               0,
                               l1 * l2 * m2 * np.sin(theta1 - theta2)],
                              [-m2 * l2 * np.sin(theta2),
                               l1 * l2 * m2 * np.sin(theta1 - theta2),
                               0]])

        G = np.array([l1 * (m1 + m2) * w1**2 * np.sin(theta1) +
                      m2 * l2 * w2**2 * np.sin(w2),
                      -l1 * l2 * m2 * w2**2 * np.sin(theta1 - theta2) +
                      g * (m1 + m2) * l1 * np.sin(theta1),
                      l1 * l2 * m2 * w1**2 * np.sin(theta1 - theta2) +
                      g * m2 * l2 * np.sin(theta2)])

        dG_theta1 = np.array([l1 * (m1 + m2) * w1**2 * np.cos(theta1),
                              -l1 * l2 * m2 * w2**2 * np.cos(theta1 - theta2) +
                              g * (m1 + m2) * l1 * np.cos(theta1),
                              l1 * l2 * m2 * w1**2 * np.cos(theta1 - theta2)])

        dG_theta2 = np.array([0,
                              l1 * l2 * m2 * w2**2 * np.cos(theta1 - theta2),
                              -l1 * l2 * m2 * w1**2 * np.cos(theta1 - theta2) +
                              g * m2 * l2 * np.cos(theta2)])

        dG_w1 = np.array([2 * l1 * (m1 + m2) * w1 * np.sin(theta1),
                          0,
                          2 * l1 * l2 * m2 * w1 * np.sin(theta1 - theta2)])

        dG_w2 = np.array([2 * m2 * l2 * w2 * np.sin(w2),
                          -2 * l1 * l2 * m2 * w2 * np.sin(theta1 - theta2),
                          0])

        d_end_state_theta1 = -M_inv @ dM_theta1 @ M_inv @ (G + np.array([u, 0, 0])) + M_inv @ dG_theta1
        d_end_state_theta2 = -M_inv @ dM_theta2 @ M_inv @ (G + np.array([u, 0, 0])) + M_inv @ dG_theta2
        d_end_state_w1 = M_inv @ dG_w1
        d_end_state_w2 = M_inv @ dG_w2

        zeros = np.zeros(3)
        d_state_y_end = np.vstack((zeros,
                                   d_end_state_theta1,
                                   d_end_state_theta2,
                                   zeros,
                                   d_end_state_w1,
                                   d_end_state_w2)).T

        d_state_y_begin = np.concatenate((np.zeros((3, 3)), np.eye(3)), axis=1)

        d_state_y = np.concatenate((d_state_y_begin, d_state_y_end), axis=0)

        d_state_u_end = M_inv @ np.array([1, 0, 0])
        d_state_u = np.concatenate((np.array([0, 0, 0]), d_state_u_end), axis=0)

        d_state_u = np.reshape(d_state_u, (6, 1))

        return (d_state_y, d_state_u)

    def _step(self, action):

        u = action
        self.counter += 1

        derivate_func = self.get_derivate_function()

        solver = ode(derivate_func)
        solver.set_f_params(u)
        t0 = 0
        solver.set_initial_value(self.state, t0)
        solver.integrate(self.tau)
        state = solver.y

        self.state = state
        x, theta1, theta2 = state[0], state[1], state[2]

        cost = normalize_angle(theta1) + normalize_angle(theta2)

        reward = -cost

        done = bool(self.counter > self.max_iter or
                    np.abs(x) > self.x_threshold)

        return self.state, reward, done, {}

    def _step_oc(self, new_state):
        self.counter += 1
        self.state = new_state
        x, theta1, theta2 = new_state[0], new_state[1], new_state[2]

        cost = normalize_angle(theta1) + normalize_angle(theta2)

        reward = -cost
        done = bool(self.counter > self.max_iter or
                    np.abs(x) > self.x_threshold)

        return self.state, reward, done, {}

    def _reset(self):
        self.state = np.array([0, -np.pi, -np.pi, 0, 0, 0])
        self.counter = 0
        return self.state

    def _render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold*2
        scale = screen_width/world_width

        cart_y = 300
        pole_width = 2.0
        pole1_length = scale * self.l1 * 2
        pole2_length = scale * self.l2 * 2
        cart_width = 50.0
        cart_height = 30.0
        circle_radius = 15

        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)

            top, buttom = cart_height/2, -cart_height/2
            left, right = -cart_width/2, cart_width/2
            cart = rendering.FilledPolygon([(left, buttom), (left, top),
                                            (right, top), (right, buttom)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            top, buttom = pole1_length-pole_width/2, -pole_width/2
            left, right = -pole_width/2, pole_width/2
            pole1 = rendering.FilledPolygon([(left, buttom), (left, top),
                                            (right, top), (right, buttom)])
            pole1.set_color(.8, .6, .4)
            self.pole1_trans = rendering.Transform(translation=(0, 0))
            pole1.add_attr(self.pole1_trans)
            pole1.add_attr(self.cart_trans)
            self.viewer.add_geom(pole1)

            self.sphere1 = rendering.make_circle(circle_radius/2)
            self.sphere1_trans = rendering.Transform(translation=(0, pole1_length))
            self.sphere1.add_attr(self.sphere1_trans)
            self.sphere1.add_attr(self.pole1_trans)
            self.sphere1.add_attr(self.cart_trans)
            self.sphere1.set_color(.5, .5, .8)
            self.viewer.add_geom(self.sphere1)

            top, buttom = pole2_length-pole_width/2, -pole_width/2
            left, right = -pole_width/2, pole_width/2
            pole2 = rendering.FilledPolygon([(left, buttom), (left, top),
                                            (right, top), (right, buttom)])
            pole2.set_color(.2, .6, .4)
            self.pole2_trans = rendering.Transform(translation=(0, pole1_length))
            pole2.add_attr(self.pole2_trans)
            pole2.add_attr(self.pole1_trans)
            pole2.add_attr(self.cart_trans)
            self.viewer.add_geom(pole2)

            self.sphere2 = rendering.make_circle(circle_radius/2)
            self.sphere2_trans = rendering.Transform(translation=(0, pole2_length))
            self.sphere2.add_attr(self.sphere2_trans)
            self.sphere2.add_attr(self.pole2_trans)
            self.sphere2.add_attr(self.pole1_trans)
            self.sphere2.add_attr(self.cart_trans)
            self.sphere2.set_color(.5, .5, .5)
            self.viewer.add_geom(self.sphere2)

            self.track = rendering.Line((0, cart_y), (screen_width, cart_y))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        state = self.state
        cart_x = state[0]*scale + screen_width/2.0
        self.cart_trans.set_translation(cart_x, cart_y)
        self.pole1_trans.set_rotation(-state[1])
        self.pole2_trans.set_rotation(-(state[2] - state[1]))

        return self.viewer.render()
