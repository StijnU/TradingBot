import random

import numpy as np
from collections import deque


class TradingEnv:
    """
    A class representing the environment for the trading agents to act in.

    Attr df: Pandas.DataFrame
        pandas dataframe containing trading data
     
    Attr init_balance: Int 
        initial balance of the agent

    Attr lookback_window_size: Int 
        the size of the lookback window (the window used as input for the agent)

    Attr render_range: Int
        the range for rendering the trading graph
    """
    def __init__(self, train_df, train_df_normalized, test_df, test_df_normalized, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=False, Show_indicators=False, normalize_value=40000):
        # Define action space and state size and other custom parameters
        self.df = train_df.reset_index()#.reset_index()#.dropna().copy().reset_index()
        self.df_normalized = train_df_normalized.reset_index()#.reset_index()#.copy().dropna().reset_index()
        self.test_df = test_df.reset_index() #test df is given so we can save it for later testing
        self.test_df_normalized = test_df_normalized.reset_index() # not sure if this is needed

        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range # render range in visualization
        self.Show_reward = Show_reward # show order reward in rendered visualization
        self.Show_indicators = Show_indicators # show main indicators in rendered visualization

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value

        self.fees = 0.001 # default Binance 0.1% order fees

        self.columns = list(self.df_normalized.columns[2:])

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.trades = deque(maxlen=self.Render_range) # limited orders memory for visualization
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0 # track episode orders count
        self.prev_episode_orders = 0 # track previous episode orders count
        self.rewards = deque(maxlen=self.Render_range)
        self.env_steps_size = env_steps_size
        self.punish_value = 0
        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance / self.normalize_value,
                                        self.net_worth / self.normalize_value,
                                        self.crypto_bought / self.normalize_value,
                                        self.crypto_sold / self.normalize_value,
                                        self.crypto_held / self.normalize_value
                                        ])

            # one line for loop to fill market history withing reset call
            self.market_history.append([self.df_normalized.loc[current_step, column] for column in self.columns])
            
        state = np.concatenate((self.orders_history, self.market_history), axis=1)

        return state

    # Get the data points for the given current_step
    def next_observation(self):
        self.market_history.append([self.df_normalized.loc[self.current_step, column] for column in self.columns])
        obs = np.concatenate((self.orders_history, self.market_history), axis=1)
        
        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        #current_price = random.uniform(
        #    self.df.loc[self.current_step, 'Open'],
        #    self.df.loc[self.current_step, 'Close'])
        current_price = self.df.loc[self.current_step, 'Open']
        Date = self.df.loc[self.current_step, 'Date'] # for visualization
        High = self.df.loc[self.current_step, 'High'] # for visualization
        Low = self.df.loc[self.current_step, 'Low'] # for visualization

        if action == 0: # Hold
            pass

        elif action == 1 and self.balance > self.initial_balance*0.05:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.crypto_bought *= (1-self.fees) # substract fees
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_bought, 'type': "buy", 'current_price': current_price})
            self.episode_orders += 1

        elif action == 2 and self.crypto_held*current_price> self.initial_balance*0.05:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.crypto_sold *= (1-self.fees) # substract fees
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_sold, 'type': "sell", 'current_price': current_price})
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance / self.normalize_value,
                                        self.net_worth / self.normalize_value,
                                        self.crypto_bought / self.normalize_value,
                                        self.crypto_sold / self.normalize_value,
                                        self.crypto_held / self.normalize_value
                                        ])

        # Receive calculated reward
        reward = self.get_reward()

        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self.next_observation()
        
        return obs, reward, done

    # Calculate reward
    def get_reward(self):
        if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
            self.prev_episode_orders = self.episode_orders
            if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
                reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - self.trades[-2]['total']*self.trades[-1]['current_price']
                self.trades[-1]["Reward"] = reward
                return reward
            elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
                reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - self.trades[-2]['total']*self.trades[-2]['current_price']
                self.trades[-1]["Reward"] = reward
                return reward
        else:
            return 0