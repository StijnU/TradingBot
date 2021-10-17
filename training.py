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
train_df_nomalized = df_normalized[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
test_df_nomalized = df_normalized[-test_window-lookback_window_size:]

# single processing training
agent = Agent(
    name=market_name+'_Training',
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

#limit resources for training (required to run on ulyssis server)
#def set_max_runtime(seconds):
#    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))

#set_max_runtime(20)

train_agent(train_env, agent, visualize=False, train_episodes=10000, training_batch_size=100)
