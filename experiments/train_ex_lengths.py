from datetime import datetime
from dateutil.relativedelta import relativedelta

from tensorflow.keras.optimizers import Adam
import json

from core.data_scraping import get_data
from core.data_preprocessing import preprocess_data
from core.agent import Agent
from core.environment import TradingEnv
from core.train import train_agent, test_agent


execution_lengths = [1, 3, 6, 12] # Amount of months used for execution of bot
train_lengths = [1, 3, 6, 12] # Amount of months used for training
market_name = 'BTC/USDT'

month = 24 * 30 # Amount of hours in a month
year = 12 # Amount of months in a year
lookback_window_size = 100

outcomes = dict()

for exl in execution_lengths:
    for trl in train_lengths:
        total_data_length = year + trl
        
        now = datetime.now()
        delta = relativedelta(months=total_data_length)
        data_date = now - delta

        df = get_data(market=market_name, from_date=str(data_date))
        df , df_normalized = preprocess_data(df)
        amount_agents = int(12/exl)

        dfs = [df[exl * x * month : (trl + exl * (x + 1)) * month] for x in range(amount_agents)]
        dfs_normalized = [df_normalized[exl * x * month : (trl + exl * (x + 1)) * month] for x in range(amount_agents)]

        dfs = [[df[:trl * month], df[trl*month:]] for df in dfs]
        dfs_normalized = [[df_normalized[:trl * month], df_normalized[trl*month:]] for df_normalized in dfs_normalized]

        for x in range(amount_agents):
            df = dfs[x][0]
            df_normalized = dfs_normalized[x][0]

            agent = Agent(
                name='Train_'+trl+'_Ex_'+exl+'_nr_'+x, 
                lookback_window_size=lookback_window_size, 
                lr=0.00001, 
                depth = len(list(df.columns[1:])),
                epochs=5, 
                optimizer=Adam, 
                batch_size = 32, 
                model="CNN")

            train_env = TradingEnv(
                df=df, 
                df_normalized=df_normalized, 
                lookback_window_size=lookback_window_size)

            train_agent(train_env, agent, visualize=False, train_episodes=50000, training_batch_size=500)
        
        net_worth = 1000
        for x in range(amount_agents):
            df = dfs[x][1]
            df_normalized = dfs_normalized[x][1]

            agent = Agent(
                name='Train_'+trl+'_Ex_'+exl+'/nr_'+x, 
                lookback_window_size=lookback_window_size, 
                lr=0.00001, 
                depth = len(list(df.columns[1:])),
                epochs=5, 
                optimizer=Adam, 
                batch_size = 32, 
                model="CNN")

            train_env = TradingEnv(
                df=df, 
                df_normalized=df_normalized, 
                lookback_window_size=lookback_window_size, 
                initial_balance=net_worth)

            net_worth += test_agent(train_env, agent, test_episodes=50, folder='trained_bots', name='Train_'+trl+'_Ex_'+exl+'_nr_'+x)
        
        outcomes['Train_length_' + trl + '_Execute_length_' + exl] = net_worth

with open('outcomes/Outcomes.json', 'w') as write_file:
    json.dump(outcomes, write_file, indent=4)
        



"""
market_name = 'BTC/USDT'
market_name_log = 'BTC_USDT'

# Get different amounts of train data and store trained models
df = get_data(market_name)
df = df.sort_values('Date')
df, df_normalized = preprocess_data(df)
depth = len(list(df.columns[1:]))

lookback_window_size = 100
test_window = 720 # 1 month


# split training and testing datasets
train_df = df[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
test_df = df[-test_window-lookback_window_size:]

# split training and testing normalized datasets
train_df_nomalized = df_normalized[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
test_df_nomalized = df_normalized[-test_window-lookback_window_size:]

# single processing training
agent = Agent(
    market_name=market_name_log, 
    lookback_window_size=lookback_window_size, 
    lr=0.00001, 
    depth=depth, 
    epochs=5, 
    optimizer=Adam, 
    batch_size = 32, 
    model="CNN")

train_env = TradingEnv(
    df=train_df, 
    df_normalized=train_df_nomalized, 
    lookback_window_size=lookback_window_size)

train_agent(train_env, agent, visualize=False, train_episodes=50000, training_batch_size=500)
"""