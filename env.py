import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from config import ENV_ID
from actions import SIMPLE_MOVEMENT


def make_env():

    env = gym_super_mario_bros.make(ENV_ID)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    return env
