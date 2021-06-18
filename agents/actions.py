import time
import gym
import numpy as np
import pandas as pd

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import TD3

def prediction(args, device, model=None,):

    if model == None:
        if args.agent == 'A2C':
            model = A2C.load(f"trained_model/A2C_{args.stock}_{int(args.time_steps // 1000)}K")
        elif args.agent == 'DDPG':
            model = DDPG.load(f"trained_model/DDPG_{args.stock}_{int(args.time_steps // 1000)}K")
        elif args.agent == 'PPO':
            model = PPO.load(f"trained_model/PPO_{args.stock}_{int(args.time_steps // 1000)}K")
        elif args.agent == 'TD3':
            model = TD3.load(f"trained_model/TD3_{args.stock}_{int(args.time_steps // 1000)}K")

    env = gym.make(f'stock_env:{args.stock}_predict-v0')

    account_memory = []
    actions_memory = []
    
    reward_list = []
    cum_reward_list = []

    cost_list = []
    tot_reward_list = []

    for iter in range(30):
        obs = env.reset()
        r = []
        cum = []
        for i in range(len(env.stock_df.index.unique())):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            r.append(rewards)

            if i == 0:
                cum.append(rewards)
            else:
                cum.append(rewards + cum[i-1])

            if iter == 0 and i == (len(env.stock_df.index.unique()) - 2):
                account_memory = env.save_asset_memory()
                actions_memory = env.save_action_memory()

            if done:
                tot_reward_list.append(info['total_reward'])
                cost_list.append(info['total_cost'])
                reward_list.append(r)
                cum_reward_list.append(cum)
                break
    
    account_name = f"{args.agent}_{args.stock}_account.csv"
    action_name = f"{args.agent}_{args.stock}_action.csv"

    reward_array = np.array(reward_list)
    mean_reward = np.mean(reward_array, axis=0)

    cum_reward_array = np.array(cum_reward_list)
    mean_cum_reward = np.mean(cum_reward_array, axis=0)

    account_memory['reward'] = mean_reward
    account_memory['cum_reward'] = mean_cum_reward

    account_memory.to_csv(f'./results/{account_name}')
    actions_memory.to_csv(f'./results/{action_name}')

    print("reward mean = {}, reward std = +/- {}, cost mean = {}, cost std = +/- {}".format(np.mean(tot_reward_list), np.std(tot_reward_list), np.mean(cost_list), np.std(cost_list)))

def train_a2c(args, device):
    env = gym.make(f'stock_env:{args.stock}_train-v0')
    env.reset()
    
    model_params = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0001}
    start_time = time.time()

    model = A2C(policy='MlpPolicy', env = env, verbose=1, device=device, **model_params)
    model = model.learn(total_timesteps=args.time_steps)

    end_time = time.time()

    model.save(f"trained_model/A2C_{args.stock}_{int(args.time_steps//1000)}K")

    prediction(args, device, model)

def train_ddpg(args, device):
    env = gym.make(f'stock_env:{args.stock}_train-v0')
    env.reset()

    model_params = {"batch_size": 64, "buffer_size": 1000, "learning_rate": 0.001}
    start_time = time.time()

    model = DDPG(policy='MlpPolicy', env = env, verbose=1, device=device, **model_params)
    model = model.learn(total_timesteps=args.time_steps)

    end_time = time.time()

    model.save(f"trained_model/DDPG_{args.stock}_{int(args.time_steps//1000)}K")

    prediction(args, device, model)

def train_ppo(args, device):
    env = gym.make(f'stock_env:{args.stock}_train-v0')
    env.reset()

    model_params = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64}
    start_time = time.time()
    model = PPO(policy='MlpPolicy', env=env, verbose=1, device=device, **model_params)
    model = model.learn(total_timesteps=args.time_steps)

    end_time = time.time()

    model.save(f"trained_model/PPO_{args.stock}_{int(args.time_steps//1000)}K")

    prediction(args, device, model)

def train_td3(args, device):
    env = gym.make(f'stock_env:{args.stock}_train-v0')
    env.reset()

    model_params = {"batch_size": 1, "buffer_size": 100, "learning_rate": 0.0001}
    start_time = time.time()
    model = TD3(policy='MlpPolicy', env=env, verbose=1, device=device, **model_params)
    model = model.learn(total_timesteps=args.time_steps)

    end_time = time.time()

    model.save(f"trained_model/TD3_{args.stock}_{int(args.time_steps//1000)}K")

    prediction(args, device, model)
