import time

from env import make_train_env, make_eval_env
from config import EVAL_RENDER_MODE, SLEEP, ACTION_REPEAT

def main():
    mode = "eval"
    
    if mode == "eval":
        env = make_eval_env()
    else:
        env= make_train_env()

    try:
        obs = env.reset()
        done = False
        action = None
        action_time = ACTION_REPEAT

        while True:
            if done:
                obs = env.reset()
                action = None
                action_time = ACTION_REPEAT
            
            if action_time == ACTION_REPEAT:
                action = env.action_space.sample()
                action_time = 0

            obs, reward, done, info = env.step(action)
            action_time += 1

            if mode == "eval":
                env.render(mode=EVAL_RENDER_MODE)
                time.sleep(SLEEP)


    finally:
        env.close()


if __name__ == "__main__":
    main()
