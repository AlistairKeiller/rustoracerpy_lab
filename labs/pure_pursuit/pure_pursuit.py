import time

import numpy as np

from rustoracerpy import RustoracerEnv

LOOKAHEAD = 1.5
WHEELBASE = 0.3302
STEER_FACTOR = 1 / 0.4189
SPEED = -0.6

env = RustoracerEnv(yaml="maps/berlin.yaml", render_mode="human")
obs, info = env.reset()
waypoints = env.skeleton.reshape(-1, 2)


try:
    while True:
        loop_start = time.perf_counter()

        (x, y, theta, *_) = info["state"][0]
        pos = np.array([x, y])

        # Find lookahead point

        # Pure pursuit steering
        action = np.array(
            [[0.0, SPEED]]
        )  # calculation action (steer, speed). Steer: [full right=-1, full left=1], Speed: [0.5 m/s=-1, 20.0 m/s=1] (m/s)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, 1.0 / 60.0 - elapsed))

        if terminated[0] or truncated[0]:
            obs, info = env.reset()
except KeyboardInterrupt:
    pass
finally:
    env.close()
