import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
from gym.spaces import Box
import torch

from torchvision import transforms as T
from config import ENV_ID
from actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack

def make_train_env():
    env = gym_super_mario_bros.make(ENV_ID)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env

def make_eval_env():
    env = gym_super_mario_bros.make(ENV_ID)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env



# 전처리 1: 프레임 스킵
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# 전처리 2: 그레이스케일 변환
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
    
    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32)
        return observation
    
    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        observation = observation.squeeze(0)
        observation = observation / 255.0
        return observation.numpy().astype(np.float32)

# 전처리 3: 관측값 리사이징
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        self.observation_space = Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        observation = torch.tensor(observation.copy(), dtype=torch.float32).unsqueeze(0)
        transform = T.Resize(self.shape, antialias=True)
        observation = transform(observation).squeeze(0)
        return observation.numpy().astype(np.float32)
    
