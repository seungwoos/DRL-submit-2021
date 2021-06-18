import pandas as pd
import yfinance as yf

def get_data(stock):

    modes = ['train', 'test']

    for m in modes:
        if stock == '^IXIC':
            if m == 'train':
                start_date = '2010-01-02'
                end_date = '2020-01-01'
            elif m == 'test':
                start_date = '2020-01-02'
                end_date = '2021-06-01'
        else:     
            if m == 'train':
                start_date = '2010-01-01'
                end_date = '2019-12-31'
            elif m == 'test':
                start_date = '2020-01-01'
                end_date = '2021-05-31'

        
        df = pd.DataFrame()
        df = yf.download(stock, start=start_date, end=end_date)
        df = df.reset_index()

        df.columns = ['date', 'open', 'high', 'low', 'close', 'adjcp', 'volume']

        if stock == '^IXIC':
            drop_cols = ['high', 'low', 'close', 'adjcp', 'volume']
            df = df.drop(drop_cols, 1)

        else:
            df['close'] = df['adjcp']
            df = df.drop('adjcp', 1)
            df['daily_return'] = df.close.pct_change(1)

        df['date']=df.date.apply(lambda x: x.strftime('%Y-%m-%d'))

        df = df.dropna()
        df = df.reset_index(drop=True)

        df.to_csv(f'./data/{stock}_{m}.csv')

if __name__ == '__main__':
    # get_data('GOOG', '2010-01-01', '2019-12-31', 'train')
    # get_data('GOOG', '2020-01-01', '2021-05-31', 'test')

    get_data('^IXIC')
    get_data('^IXIC')
