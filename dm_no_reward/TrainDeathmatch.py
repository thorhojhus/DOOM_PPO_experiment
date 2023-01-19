from stable_baselines3 import PPO
from TrainAndLoggingCallback import TrainAndLoggingCallback
from DeathmatchEnv import DoomWithBots

from stable_baselines3.common import vec_env
from stable_baselines3.common.callbacks import EvalCallback

CHECKPOINT_DIR = 'train_deathmatch/'
LOG_DIR = 'log_deathmatch/'
CHECKPOINT_DIR_EVAL = 'train_deathmatch_EVAL/'
LOG_DIR_EVAL = 'log_deathmatch_EVAL/'


callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)


env = vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: DoomWithBots(render=False)] * 4))
eval_env = vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: DoomWithBots(render=False)] * 1))

eval_callback = EvalCallback(
    eval_env, 
    n_eval_episodes=5, 
    eval_freq=16384, 
    log_path=LOG_DIR_EVAL,
    best_model_save_path=CHECKPOINT_DIR_EVAL)

model = PPO('CnnPolicy', env, n_epochs=3, tensorboard_log=LOG_DIR, n_steps = 8192, verbose=1, ent_coef=0.001, gamma=0.908, learning_rate=2.6e-4)

model.learn(total_timesteps=12e6, callback=[callback, eval_callback])