import argparse
import torch

from agents.actions import train_a2c, train_ddpg, train_ppo, train_td3, prediction
from fetch_data import get_data

def argparser():
    parser = argparse.ArgumentParser(description='Stock market trading bot')
    
    parser.add_argument('--gpu', type=str, default='0',
                        help='number of gpu device id')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'],
                        help='train or test the agent')
    parser.add_argument('--stock', type=str, default=None, help='the name of stock interested in')
    parser.add_argument('--agent', type=str, default='A2C', choices=['A2C', 'DDPG', 'TD3', 'PPO'],
                        help='an DRL algorithm used for agent')
    parser.add_argument('--time_steps', type=int, default=1e5)

    return parser.parse_args()

def main():
    args = argparser()

    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'

    if args.stock is not None:
        get_data(args.stock)

    if args.phase == 'train':
        if args.agent == 'A2C':
            train_a2c(args, device)
        elif args.agent == 'DDPG':
            train_ddpg(args, device)
        elif args.agent == 'PPO':
            train_ppo(args, device)
        elif args.agent == 'TD3':
            train_td3(args, device)

    elif args.phase == 'test':
        prediction(args, device=device)

if __name__ == "__main__":
    main()