import time

from env import make_env
from config import RENDER, SLEEP


def main():
    env = make_env()

    try:
        obs = env.reset()
        done = True

        while True:
            if done:
                obs = env.reset()

            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            if RENDER:
                env.render()
                time.sleep(SLEEP)

    finally:
        env.close()


if __name__ == "__main__":
    main()
