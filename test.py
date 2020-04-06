from environment import DoublePendulum
import time

env = DoublePendulum()

for t in range(10000):
    env._render()
    action = (2 * env.action_space.sample() - 1) * 100
    time.sleep(0.001)
    obs, r, done, info = env._step(action)
    if done:
        break
