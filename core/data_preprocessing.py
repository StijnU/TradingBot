API_KEY = '4WDB3w52jwYaYnESCzc4KzFgOQyUHJK9ZoyKy57SBQvEYQ6I9ptgiPHNbVPo4ZBBq'
API_SECRET = 'M2bwdt2zPvQM628hpJpoKfojMtEg3H82H3uWYMcaaa4ukHQso6VIgnkhS9GxLJQC'
import pandas as pd
import numpy as np
from ta import add_all_ta_features

def preprocess_data(df):
    df = add_indicators_dataframe(df)
    df_normalized = Normalize(df[99:])[1:].dropna()
    df = df[100:].dropna()

    df.reset_index(drop=True)
    df_normalized.reset_index(drop=True)

    return df, df_normalized

def DropCorrelatedFeatures(df, threshold):
    df_copy = df.copy()

    # Remove OHCL columns
    df_drop = df_copy.drop(["Date", "Open", "High", "Low", "Close", "Volume"], axis=1)

    # Calculate Pierson correlation
    df_corr = df_drop.corr()

    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    for i in range(df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[0]):
            if df_corr.iloc[i,j] >= threshold or df_corr.iloc[i,j] <= -threshold:
                if columns[j]:
                    columns[j] = False
                    
    selected_columns = df_drop.columns[columns]
    df_dropped = df_drop[selected_columns] 
    return df_dropped

def get_all_indicators(df, threshold=0.5):
    df_all = df.copy()
    df_all = add_all_ta_features(df_all, open="Open", high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_all, threshold)

def add_indicators_dataframe(df):
    indicators = get_all_indicators(df)
    df = pd.concat([df, indicators], axis=1)
    return df

def Normalize(df_original):
    df = df_original.copy()
    column_names = df.columns.tolist()
    for column in column_names[1:]:
        # Logging and Differencing
        test = np.log(df[column]) - np.log(df[column].shift(1))
        if test[1:].isnull().any():
            df[column] = df[column] - df[column].shift(1)
        else:
            df[column] = np.log(df[column]) - np.log(df[column].shift(1))
        # Min Max Scaler implemented
        #Min = df[column].min()
        #Max = df[column].max()
        #df[column] = (df[column] - Min) / (Max - Min)

    return df

