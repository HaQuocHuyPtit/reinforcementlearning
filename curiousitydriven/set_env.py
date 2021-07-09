# action space
# COMPLEX_MOVEMENT = [
#  ['NOOP'],
#  ['right'],
#  ['right', 'A'],
#  ['right', 'B'],
#  ['right', 'A', 'B'],
#  ['A'],
#  ['left'],
#  ['left', 'A'],
#  ['left', 'B'],
#  ['left', 'A', 'B'],
#  ['down'],
#  ['up'],
#  ]
# shape of ob: (240, 256, 3)
# size of act: 12
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


def setup_env(env_name: str) -> gym.Env:
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env