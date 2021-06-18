from mpl_finance import candlestick2_ohlc

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import argparse
import pandas as pd
import numpy as np

def plot_graph(action_csv, stock_csv, save_name):

    fig = plt.figure(figsize=(25,15))
    top_axes = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=4)
    bottom_axes = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4, sharex=top_axes)
    bottom_axes.get_yaxis().get_major_formatter().set_scientific(False)

    stock_df = pd.read_csv(stock_csv)
    stock_df['MA3'] = stock_df['close'].rolling(3).mean()
    stock_df['MA10'] = stock_df['close'].rolling(10).mean()
    stock_df['MA20'] = stock_df['close'].rolling(20).mean()

    index = stock_df.date

    top_axes.plot(index, stock_df['MA3'], label='MA3', linewidth=0.7)
    top_axes.plot(index, stock_df['MA10'], label='MA10', linewidth=0.7)
    top_axes.plot(index, stock_df['MA20'], label='MA20', linewidth=0.7)

    top_axes.xaxis.set_major_locator(ticker.MaxNLocator(10))

    top_axes.set_title(save_name, fontsize=22)
    top_axes.set_xlabel('Date')

    candlestick2_ohlc(top_axes, stock_df['open'], stock_df['high'], 
                  stock_df['low'], stock_df['close'],
                  width=0.5, colorup='r', colordown='b')
    
    action_df = pd.read_csv(action_csv)
    color_fuc = lambda x : 'r' if x > 0 else 'b'
    color_list = list(action_df['actions'].fillna(0).apply(color_fuc))

    idx = action_df.date
    bottom_axes.bar(idx, action_df['actions'], width=0.5, 
                    align='center', color=color_list)

    bottom_axes.set_xlabel('Date')

    top_axes.legend()
    plt.grid()

    plt.savefig(f"plots/{save_name}.png")

def plot_reward(account_csv, save_name):
    account_df = pd.read_csv(account_csv)
    fig = plt.figure(figsize=(20, 10))

    account_df[['date', 'reward']].plot()
    plt.xlabel('Date')
    plt.ylabel('Mean Reward')
    plt.title(f"{save_name} 30 sampling")
    plt.savefig(f"plots/{save_name}.png")

def plot_cum_reward(account_csv, save_name):
    account_df = pd.read_csv(account_csv)
    fig = plt.figure(figsize=(20, 10))

    account_df[['date', 'cum_reward']].plot()
    plt.xlabel('Date')
    plt.ylabel('Mean Cumulative Reward')
    plt.title(f"{save_name} 30 sampling")
    plt.savefig(f"plots/{save_name}.png")

def argparser():
    parser = argparse.ArgumentParser(description='plot the result')

    parser.add_argument('--action', type=str, required=True)
    parser.add_argument('--account', type=str, required=True)
    parser.add_argument('--stock', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()

    file_name = args.account.split('results/')[1]
    file_name = file_name.split('_account.csv')[0]
    plot_graph(args.action, args.stock, file_name)
    plot_reward(args.account, f'{file_name}_reward')
    plot_cum_reward(args.account, f'{file_name}_cum_reward')

    # plot_graph('results/A2C_AAPL_action.csv', 'data/AAPL_test.csv', save_name='A2C_Apple')
    # plot_reward('results/PPO_AAPL_account.csv', save_name='PPO_AAPL_reward')
    plot_cum_reward('results/A2C_AAPL_account.csv', save_name='A2C_Apple_cum_reward')
