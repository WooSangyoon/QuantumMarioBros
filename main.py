import csv
import os
import time
import argparse
import numpy as np

from agent import DDQNAgent, QuantumDDQNAgent
from env import make_train_env, make_eval_env
from config import EVAL_RENDER_MODE, SLEEP, ACTION_REPEAT, NUM_EPISODES

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["ddqn", "quantum"], default="ddqn")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--render-train", action="store_true")
    return parser.parse_args()


def main(agent_type="ddqn", mode="train", episodes=NUM_EPISODES, render_train=False):
    reward_history = []
    checkpoint_path = os.path.join("models", f"{agent_type}_mario.pth")
    log_path = os.path.join("logs", f"{agent_type}_train_log.csv")
    
    if mode == "eval":
        env = make_eval_env()
    else:
        env = make_train_env()

    agent_class = DDQNAgent if agent_type == "ddqn" else QuantumDDQNAgent
    agent = agent_class(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
    )
    print(f"Using device: {agent.device}")

    if os.path.exists(checkpoint_path):
        try:
            agent.load(checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_path}")
        except RuntimeError as error:
            print(
                f"Skipped incompatible checkpoint: {checkpoint_path}\n"
                f"Reason: {error}"
            )

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["episode", "steps", "loss", "epsilon", "reward", "mean_q", "max_q"]
            )

    try:
        for episode in range(1, episodes + 1):
            obs = env.reset()
            done = False
            action = None
            action_time = ACTION_REPEAT
            episode_reward = 0.0
            step_count = 0
            last_loss = None
            episode_losses = []
            episode_mean_qs = []
            episode_max_qs = []

            while not done:
                if action is None or action_time == ACTION_REPEAT:
                    action = agent.select_action(obs, training=(mode == "train"))
                    action_time = 0

                next_obs, reward, done, info = env.step(action)

                if mode == "train":
                    agent.store_transition(obs, action, reward, next_obs, done)
                    update_info = agent.update()
                    if update_info is not None:
                        last_loss = update_info["loss"]
                        episode_losses.append(update_info["loss"])
                        episode_mean_qs.append(update_info["mean_q"])
                        episode_max_qs.append(update_info["max_q"])

                obs = next_obs
                episode_reward += reward
                step_count += 1
                action_time += 1


                # env.render(mode=EVAL_RENDER_MODE)
                # time.sleep(SLEEP)
                               
                if mode == "eval":
                    env.render(mode=EVAL_RENDER_MODE)
                    time.sleep(SLEEP)
                elif mode == "train" and render_train:
                    env.render(mode=EVAL_RENDER_MODE)
                    time.sleep(SLEEP)

            if mode == "train":
                if episode % 5 == 0:
                    agent.decay_epsilon()
                    agent.save(checkpoint_path)

            mean_loss = float(np.mean(episode_losses)) if episode_losses else None
            mean_q = float(np.mean(episode_mean_qs)) if episode_mean_qs else None
            max_q = float(np.max(episode_max_qs)) if episode_max_qs else None

            reward_history.append(episode_reward)
            print(
                f"Episode {episode}/{episodes} | "
                f"steps={step_count} | reward={episode_reward:.2f} | "
                f"epsilon={agent.epsilon:.3f} | loss={last_loss}"
            )

            if mode == "train":
                with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [
                            episode,
                            step_count,
                            mean_loss,
                            agent.epsilon,
                            episode_reward,
                            mean_q,
                            max_q,
                        ]
                    )

        recent_count = min(5, len(reward_history))
        print(
            f"Recent average {recent_count} episodes reward="
            f"{np.mean(reward_history[-recent_count:]):.2f}"
        )

        if mode == "train":
            agent.save(checkpoint_path)
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
    args = parse_args()
    main(
        agent_type=args.agent,
        mode=args.mode,
        episodes=args.episodes,
        render_train=args.render_train,
    )
