# Stock Market Trading Bot using Reinforcement Learning

20215084 김승우

## Installation

### Using Anaconda

```
conda env create -f stock_market.yml
conda activate drl
```

### Using Pip

```
pip install -r reqruiements.txt
```

## Run

### Default
```
python3 main.py 
```
- agent: A2C, stock: Google, phase: train, gpu: 0, time_steps: 100,000

### Train a model

```
python3 main.py --gpu GPU_ID --phase 'train' --stock STOCK --agent MODEL --time_steps NUM_EPISODES
```

- STOCK: ['GOOG', 'APPL']
- MODEL: ['A2C', 'PPO', 'DDPG', 'TD3']

### Test a model

```
python3 main.py --gpu GPU_ID --phase 'test' --stock STOCK --agent MODEL --time_steps NUM_EPISODES
```

- STOCK: ['GOOG', 'APPL']
- MODEL: ['A2C', 'PPO', 'DDPG', 'TD3']

### Plot 

```
python3 plot.py --account ACCOUNT_CSV --action ACTION_CSV --stock STOCK_CSV
```

- ACCOUNT_CSV: 'results/MODEL_STOCK_account.csv'
- ACTION_CSV: 'results/MODEL_STOCK_action.csv'
- STOCK_CSV: 'data/STOCK_test.csv'

- STOCK: ['GOOG', 'APPL']
- MODEL: ['A2C', 'PPO', 'DDPG', 'TD3']

### Environment를 불러올 수 없다면

```
pip install -e stock-env
```

## Dependencies

- Ubuntu 20.04
- Python 3.8
- CUDA 10.2
- PyTorch 1.9.0
- YahooFinance 0.1.59
- Stable Baselines3 1.0

## References

- FinRL - https://github.com/AI4Finance-LLC/FinRL

- OpenAI custom environment - https://github.com/openai/gym/blob/master/docs/creating-environments.md

- Stable Baselines3 - https://github.com/DLR-RM/stable-baselines3