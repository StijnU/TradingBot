import json
from collections import deque
import pandas as pd
import os
from core.data_scraping import get_data
from core.data_preprocessing import preprocess_data
from core.agent import Agent
from core.environment import TradingEnv
from core.train_test import test_agent

# This script tests all the trained bots in the 'trained bots' folder
dir = 'trained_bots'

# for loop over all folders in trained_bots folder
for subdir in os.walk(dir):
    if len(subdir[2]) > 0:
        h5_list = deque(maxlen=6)
        avg_incomes = deque(maxlen=3)
        for file in subdir[2]:
            if file[len(file) - 8:] == 'Actor.h5':
                if len(avg_incomes) > 0:
                    if ( min(avg_incomes) < float(file[:6]) ):
                        h5_list.append(file[:len(file) - 9])
                        avg_incomes.append(float(file[:7]))
                else:
                    h5_list.append(file[:len(file) - 9])
                    avg_incomes.append(float(file[:7]))
        
        if len(h5_list) > 0:
            print(h5_list)
            
            # Open test csv to pandas dataframe
            agentdir = subdir[0][:len(subdir[0]) - 7]
            test_df = pd.read_csv(agentdir + '/test.csv')

            # for loop for every model
            for h5 in h5_list:            
                agent = Agent('test', 'testing')
                env = TradingEnv(train_df= pd.DataFrame(), train_df_normalized= pd.DataFrame(), test_df=test_df, test_df_normalized= pd.DataFrame())
                agent.load(subdir[0], h5)
                test_agent(env, agent, test_episodes=10, folder="", name="", comment="")