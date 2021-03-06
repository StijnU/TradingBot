from tensorflow.keras.optimizers import Adam
import resource

from core.data_scraping import get_data
from core.data_preprocessing import preprocess_data
from core.agent import Agent
from core.environment import TradingEnv
from core.train_test import train_agent

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
train_df_normalized = df_normalized[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
test_df_normalized = df_normalized[-test_window-lookback_window_size:]

# single processing training
agent = Agent(
    name=market_name_log+'_Training',
    market=market_name,
    lookback_window_size=lookback_window_size, 
    lr=0.00001, 
    depth=depth, 
    epochs=5, 
    optimizer=Adam, 
    batch_size = 32, 
    model="CNN")

train_env = TradingEnv(
    train_df=train_df, 
    train_df_normalized=train_df_normalized,
    test_df=test_df,
    test_df_normalized=test_df_normalized, 
    lookback_window_size=lookback_window_size)

train_agent(train_env, agent, visualize=False, train_episodes=10000, training_batch_size=100)
