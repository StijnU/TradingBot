API_KEY = '4WDB3w52jwYaYnESCzc4KzFgOQyUHJK9ZoyKy57SBQvEYQ6I9ptgiPHNbVPo4ZBBq'
API_SECRET = 'M2bwdt2zPvQM628hpJpoKfojMtEg3H82H3uWYMcaaa4ukHQso6VIgnkhS9GxLJQC'
from datetime import datetime
import time
import pandas as pd
import ccxt

def get_data(market='BTC/USDT', time_interval='1h', from_date='2021-06-17 00:00:00', until_date=None):
    exchange_id = 'binance'
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'timeout': 30000
    })
    hour = 3600 * 1000
    limit = 1000

    market_in_exchange = False
    for ma in exchange.fetch_markets():
        if ma['symbol'] == market:
            market_in_exchange = True

    t_start = exchange.parse8601(from_date) - hour*19
    if until_date is not None:
        t_stop = exchange.parse8601(until_date)
    else:
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:00:00")
        t_stop = exchange.parse8601(now)
    
    if market_in_exchange:
        step = hour * limit
        data = []

        total_steps = (t_stop-t_start)/hour
        while total_steps > 0:
            if total_steps < limit: # recalculating ending steps
                step = total_steps * hour

            data += exchange.fetch_ohlcv(market, exchange.timeframes[time_interval], t_start, limit=1000)               
            t_start = t_start + step
            total_steps -= limit
            time.sleep(exchange.rateLimit/1000)
        
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = pd.DataFrame(data, columns=columns)
        return data.dropna().reset_index(drop=True)