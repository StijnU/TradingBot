from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots
from datetime import datetime

def Write_to_file(Date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
    for i in net_worth: 
        Date += " {}".format(i)
    #print(Date)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file = open("logs/"+filename, 'a+')
    file.write(Date+"\n")
    file.close()

class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, df, net_worth, order_history):
        self.coin_df = df
        self.net_worth_df = net_worth
        self.order_history = order_history

    # Render the environment to the screen
    def render(self):   
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.01, shared_xaxes=True)
        fig.add_trace(
            go.Scatter(
                x=self.net_worth_df['date'],
                y=self.net_worth_df['worth']),
                row=2,
                col=1
        )
        fig.add_trace(
            go.Candlestick(
                x=self.coin_df['Date'],
                open=self.coin_df['Open'],
                high=self.coin_df['High'],
                low=self.coin_df['Low'],
                close=self.coin_df['Close'])
        )
        fig.layout.update(showlegend=False)
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            annotations=self.build_order_annotations()
        )
        
        fig.show()

    def build_order_annotations(self):
        annotations = list()
        buy_dates = self.order_history.loc[self.order_history['action'] == 1, 'date']
        sell_dates = self.order_history.loc[self.order_history['action'] == 2, 'date']
        
        for buy_date in buy_dates:
            annotations.append(
                {'x': buy_date, 'y': 0.05, 'xref': 'x', 'yref': 'paper',
                    'showarrow': False, 'xanchor': 'left', 'text': 'B' }
            )
        
        for sell_date in sell_dates:
            annotations.append(
                {'x': sell_date, 'y': 0.05, 'xref': 'x', 'yref': 'paper',
                    'showarrow': False, 'xanchor': 'left', 'text': 'S' }
            )
        return annotations
