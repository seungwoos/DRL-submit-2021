import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding

class StockMarketEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    def __init__(self, stock_df, nasdaq_df, stock_dim, state_space):

        self.day = 0
        self.hmax = 100
        self.init_budget = 100000
        self.transaction_fee = 0.001
        self.reward_scaling = 1e-4

        self.stock_df = pd.read_csv(stock_df)
        self.nasdaq_df = pd.read_csv(nasdaq_df)

        # stock dim = action space = 1
        # state space = 7 + 1
        self.stock_dim = stock_dim
        self.state_space = state_space
        self.action_space = stock_dim

        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.stock_dim,)) 
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.state_space,))

        self.data = self.stock_df.loc[self.day, :]
        self.nasdaq = self.nasdaq_df.loc[self.day, :]

        self.terminal = False     

        # initalize state: inital budget + close price + shares + open price + highest price + lowest price + daily return + NASDAQ
        self.state = self._init_state()

        # initialize reward
        self.reward = 0
        self.cost = 0

        # memorize all the total balance change
        self.asset_memory = [self.init_budget]
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self.data.date]
        self.close_price_memory = [self.data.close]
        self.num_trades = 0
        self._seed()

    def _sell_stock(self, action):
        if self.state[2] > 0:

            self.state[0] += self.state[1] * min(abs(action), self.state[2]) * (1 - self.transaction_fee)
            self.state[2] -= min(abs(action), self.state[2])
            self.cost += self.state[1]*min(abs(action), self.state[2]) * self.transaction_fee

            self.num_trades += 1

    def _buy_stock(self, action):

        available_amount = self.state[0] // self.state[1]

        self.state[0] -= self.state[1] * min(available_amount, action) * (1 + self.transaction_fee)
        self.state[2] += min(available_amount, action)
        self.cost += self.state[1] * min(available_amount, action) * self.transaction_fee

        self.num_trades+=1
        
    def step(self, actions):
        self.terminal = self.day >= len(self.stock_df.index.unique())-1

        if self.terminal:
            end_total_asset = self.state[0] + np.array(self.state[1]) * np.array(self.state[2])

            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(end_total_asset))

            total_reward = self.state[0] + (np.array(self.state[1]) * np.array(self.state[2])) - self.init_budget
            print("total_reward:{}".format(total_reward ))
            print("total_cost: ", self.cost)
            print("total_num_trades: ", self.num_trades)
            
            if "train" in self.stock_df:
                return self.state, self.reward, self.terminal, {}
            else:
                return self.state, self.reward, self.terminal, {'total_reward': total_reward, 'total_cost': self.cost}

        else:
            actions = float(actions * self.hmax)
            # print("state: {}, action: {}".format(self.state, actions))
            self.actions_memory.append(actions)
            
            begin_total_asset = self.state[0]+ np.array(self.state[1]) * np.array(self.state[2])
            
            sell = actions < 0
            buy = actions > 0

            if sell:
                self._sell_stock(actions)

            if buy:
                self._buy_stock(actions)

            self.day += 1
            self.data = self.stock_df.loc[self.day, :]         

            self.state =  self._update_state()
            
            end_total_asset = self.state[0]+ np.array(self.state[1]) * np.array(self.state[2])

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.data.date)
            self.close_price_memory.append(self.data.close)
            
            self.reward = end_total_asset - begin_total_asset            

            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward * self.reward_scaling
            
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.init_budget]
        self.day = 0
        self.data = self.stock_df.loc[self.day, :]
        self.nasdaq = self.nasdaq_df.loc[self.day, :]
        self.cost = 0
        self.num_trades = 0
        self.terminal = False 
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self.data.date]
        self.state = self._init_state()

        return self.state
    
    def render(self, mode='human'):
        return self.state
    
    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        stock_df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return stock_df_account_value

    def save_action_memory(self):
        date_list = self.date_memory[:-1]
        close_price_list = self.close_price_memory[:-1]

        action_list = self.actions_memory
        stock_df_actions = pd.DataFrame({'date':date_list,'actions':action_list,'close_price':close_price_list})
        stock_df_actions['daily_return']=stock_df_actions.close_price.pct_change()
        return stock_df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _init_state(self):

        self.state = [self.init_budget] + [self.data.close] + [0] + [self.data.open] \
                    + [self.data.high] + [self.data.low] +[self.data.daily_return] + [self.nasdaq.open]
        
        return self.state

    def _update_state(self):

        self.state = [self.state[0]] + [self.data.close] + [self.state[2]] \
                    + [self.data.open] + [self.data.high] + [self.data.low] +[self.data.daily_return] + [self.nasdaq.open]
        
        return self.state