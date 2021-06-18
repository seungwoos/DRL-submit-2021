from gym.envs.registration import register

register(
    id='GOOG_train-v0',
    entry_point='stock_env.envs:StockMarketEnv',
    kwargs={'stock_df': './data/GOOG_train.csv', 'nasdaq_df': './data/^IXIC_train.csv', 'stock_dim': 1, 'state_space': 8}
)

register(
    id='GOOG_predict-v0',
    entry_point='stock_env.envs:StockMarketEnv',
    kwargs={'stock_df': './data/GOOG_test.csv', 'nasdaq_df': './data/^IXIC_test.csv', 'stock_dim': 1, 'state_space': 8}
)
register(
    id='AAPL_train-v0',
    entry_point='stock_env.envs:StockMarketEnv',
    kwargs={'stock_df': './data/AAPL_train.csv', 'nasdaq_df': './data/^IXIC_train.csv', 'stock_dim': 1, 'state_space': 8}
)

register(
    id='AAPL_predict-v0',
    entry_point='stock_env.envs:StockMarketEnv',
    kwargs={'stock_df': './data/AAPL_test.csv', 'nasdaq_df': './data/^IXIC_test.csv', 'stock_dim': 1, 'state_space': 8}
)
