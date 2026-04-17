import time
import numpy as np

from agent import DQNAgent
from env import make_train_env, make_eval_env
from config import EVAL_RENDER_MODE, SLEEP, ACTION_REPEAT, NUM_EPISODES

def main():
    mode = "train"
    reward_history = []
    
    if mode == "eval":
        env = make_eval_env()
    else:
        env = make_train_env()

    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
    )

    try:
        for episode in range(1, NUM_EPISODES + 1):
            obs = env.reset()
            done = False
            action = None
            action_time = ACTION_REPEAT
            episode_reward = 0.0
            step_count = 0
            last_loss = None

            while not done:
                if action is None or action_time == ACTION_REPEAT:
                    action = agent.select_action(obs, training=(mode == "train"))
                    action_time = 0

                next_obs, reward, done, info = env.step(action)

                if mode == "train":
                    agent.store_transition(obs, action, reward, next_obs, done)
                    last_loss = agent.update()

                obs = next_obs
                episode_reward += reward
                step_count += 1
                action_time += 1


                env.render(mode=EVAL_RENDER_MODE)
                time.sleep(SLEEP)
                               
                # if mode == "eval":
                #     env.render(mode=EVAL_RENDER_MODE)
                #     time.sleep(SLEEP)

            if mode == "train":
                agent.decay_epsilon()

            reward_history.append(episode_reward)
            print(
                f"Episode {episode}/{NUM_EPISODES} | "
                f"steps={step_count} | reward={episode_reward:.2f} | "
                f"epsilon={agent.epsilon:.3f} | loss={last_loss}"
            )

        recent_count = min(5, len(reward_history))
        print(
            f"Recent average {recent_count} episodes reward="
            f"{np.mean(reward_history[-recent_count:]):.2f}"
        )

        if mode == "train":
            eval_env = make_eval_env()
            try:
                obs = eval_env.reset()
                done = False
                action = None
                action_time = ACTION_REPEAT
                eval_reward = 0.0
                eval_steps = 0

                while not done:
                    if action is None or action_time == ACTION_REPEAT:
                        action = agent.select_action(obs, training=False)
                        action_time = 0

                    obs, reward, done, info = eval_env.step(action)
                    eval_reward += reward
                    eval_steps += 1
                    action_time += 1

                    eval_env.render(mode=EVAL_RENDER_MODE)
                    time.sleep(SLEEP)

                print(
                    f"Final evaluation | steps={eval_steps} | reward={eval_reward:.2f}"
                )
            finally:
                eval_env.close()

    finally:
        env.close()


if __name__ == "__main__":
    main()
