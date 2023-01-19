from vizdoom import * 

import cv2
import random
import time 
import numpy as np

from gym import Env 
from gym.spaces import Discrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from DeathmatchEnv import DoomWithBots
from stable_baselines3.common import vec_env


# Create rendered environment
env = vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: DoomWithBots(render=True)]))

# Reload model from disc
model = PPO.load('train_deathmatch_EVAL/best_model.zip')

for episode in range(20): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.03)
        total_reward += reward
    print('Total reward for run {} is {}'.format(episode, total_reward))
    time.sleep(3)