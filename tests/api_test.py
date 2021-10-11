API_KEY = '4WDB3w52jwYaYnESCzc4KzFgOQyUHJK9ZoyKy57SBQvEYQ6I9ptgiPHNbVPo4ZBBq'
API_SECRET = 'M2bwdt2zPvQM628hpJpoKfojMtEg3H82H3uWYMcaaa4ukHQso6VIgnkhS9GxLJQC'
import schedule
import time
import datetime

import ccxt


lookbackwindow = 50

exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'timeout': 30000
})

print(exchange.timeframes)


def get_ohlcv(exchange, symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, exchange.timeframes['1m'], limit=lookbackwindow)
    print('Received at: ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('Received timestamp: ',time.time())
    print('Data from: ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ohlcv[49][0]/1000)))
    print('Data timestamp: ',ohlcv[49][0]/1000)

markets = exchange.fetch_markets()

while True:
    if datetime.datetime.now().second == 50:
        schedule.every().minute.do(get_ohlcv, exchange, markets[0]['symbol'])
        time_start = time.time()
        duration = 3000
        while time.time() < time_start + duration:
            schedule.run_pending()
            time.sleep(0.1)
        break

