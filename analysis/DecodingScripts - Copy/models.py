import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge



def perform_custom_split(X, Y, test_size=0.25):
    # Find all unique episodes
    episodes = X['episode_id'].unique()
    
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []
    
    for ep in episodes:
        # Filter data for this episode
        mask = X['episode_id'] == ep
        X_ep = X[mask]
        Y_ep = Y[mask]
        
        # Find split index
        n = len(X_ep)
        split_idx = int(n * (1 - test_size))
        
        # Split into train and test
        X_train_list.append(X_ep.iloc[:split_idx])
        Y_train_list.append(Y_ep.iloc[:split_idx])
        
        X_test_list.append(X_ep.iloc[split_idx:])
        Y_test_list.append(Y_ep.iloc[split_idx:])
    
    # Combine all episodes' train and test data
    X_train = pd.concat(X_train_list, ignore_index=True)
    Y_train = pd.concat(Y_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    Y_test = pd.concat(Y_test_list, ignore_index=True)
    
    # Remove episode_id column from X and convert to numpy
    X_train = X_train.drop(columns='episode_id').values
    X_test = X_test.drop(columns='episode_id').values
    
    # Convert Y to numpy
    Y_train = Y_train.values
    Y_test = Y_test.values
    
    return X_train, X_test, Y_train, Y_test

def correlate(df_merged, custom_train_test_split=False):
    # Drop rows with NaN values in 'delta_X' and 'delta_Y'
    y = df_merged[['target_x', 'target_y']]
    
    # Align 'df' data to match 'y' dimensions by dropping the last 'future_steps' rows
    if custom_train_test_split:
        features = df_merged[[f'Node {i}' for i in range(1,513)] + ['episode_id']]
        X_train, X_test, y_train, y_test = perform_custom_split(features, y)
    else:
        features = df_merged[[f'Node {i}' for i in range(1, 513)]]
        X = np.array(features)
        Y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25) 
    # Kernel
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)    
    return model, X_test, y_test



def RMSE_angle(y_test, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE) of angle predictions.
    """
    # Extract cosine and sine components for predictions
    cos_theta_pred = y_pred[:, 0]
    sin_theta_pred = y_pred[:, 1]
    
    # Extract cosine and sine components for real values
    cos_theta_real = y_test[:, 0]
    sin_theta_real = y_test[:, 1]
    
    # Compute cosine of the angle difference
    cos_diff = cos_theta_pred * cos_theta_real + sin_theta_pred * sin_theta_real
    
    # Clip the cosine values to the valid range [-1, 1] to avoid numerical issues
    cos_diff = np.clip(cos_diff, -1.0, 1.0)
    
    # Calculate the angle difference in radians
    theta_diff = np.arccos(cos_diff)
    
    # Square the angle differences
    squared_diff = theta_diff ** 2
    
    # Sum all squared differences
    sum_squared = np.sum(squared_diff)/len(squared_diff)
    
    # Compute the RMSE
    rmse = np.sqrt(sum_squared)

    rmse = np.degrees(rmse)
    
    return rmse

