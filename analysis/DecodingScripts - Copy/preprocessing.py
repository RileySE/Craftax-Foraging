import pandas as pd
import numpy as np  
# Add the relative displacement based on multiple episodes here!

class DataPreprocessor:
    def __init__(self, file_path=None, file_path_scalar=None):
        self.file_path = file_path
        self.file_path_scalar = file_path_scalar

    def merge_files_single(self, file_path, file_path_scalar):
        df_scalar = pd.read_csv(file_path_scalar, low_memory=False)
        df_scalar = df_scalar[df_scalar['# action'] != '# action']
        df = pd.read_csv(file_path, header=None, nrows=len(df_scalar))
        df.columns = [f"Node {i}" for i in range(1, df.shape[1] + 1)]
        df_scalar = df_scalar[df_scalar['player_position_x'] != 'player_position_x']
        merged_df = pd.concat([df_scalar.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
        return self.get_episode(merged_df)
    
    def merge_files_multiple(self, file_paths, file_paths_scalar):
        df_list = []
        for file_path, file_path_scalar in zip(file_paths, file_paths_scalar):
            # Get the full merged dataframe first, don't get episode immediately
            df_scalar = pd.read_csv(file_path_scalar, low_memory=False)
            df_scalar = df_scalar[df_scalar['# action'] != '# action']
            df = pd.read_csv(file_path, header=None, nrows=len(df_scalar))
            df.columns = [f"Node {i}" for i in range(1, df.shape[1] + 1)]
            df_scalar = df_scalar[df_scalar['player_position_x'] != 'player_position_x']
            df = pd.concat([df_scalar.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
            
            # Now process episodes from this dataframe
            while True:
                episode_df = self.get_episode(df)
                if len(episode_df) < 3000:
                    break
                episode_df = episode_df[500:-500]
                df_list.append(episode_df)
                
                # Filter out rows with matching episode_id
                df = df[~df['episode_id'].isin(episode_df['episode_id'].unique())]
                
                if len(df) < 3000:
                    break
        return df_list

    def merge_files(self, file_path=None, file_path_scalar=None):
        # Use instance variables if no arguments provided
        if file_path is None:
            file_path = self.file_path
        if file_path_scalar is None:
            file_path_scalar = self.file_path_scalar

        if isinstance(file_path, list) and isinstance(file_path_scalar, list):
            return self.merge_files_multiple(file_path, file_path_scalar)
        else:
            return self.merge_files_single(file_path, file_path_scalar)

    def get_episode(self, df):
        """Extract longest episode from dataframe."""
        most_common_episode_id = df['episode_id'].mode()[0]
        episode_df = df[df['episode_id'] == most_common_episode_id].copy()
        
        # Convert columns to numeric type
        for col in ['player_position_x', 'player_position_y', 'delta_x', 'delta_y']:
            try:
                # First try to convert directly to numeric
                episode_df.loc[:, col] = pd.to_numeric(episode_df[col])
            except:
                # If that fails, try string processing
                try:
                    episode_df.loc[:, col] = episode_df[col].astype(str).str.split('.').str[0]
                except:
                    print(f"Warning: Could not process column {col}")
                    continue
            
            with pd.option_context("future.no_silent_downcasting", True):
                episode_df.loc[:, col] = episode_df.loc[:, col].fillna(False).infer_objects(copy=False)
        
        return episode_df
    
    

    def _handle_steps_shift(self, df, steps):
        """Helper method to handle DataFrame shifting based on steps."""
        if steps > 0:
            df = df.iloc[:-steps]
        elif steps < 0:
            df = df.iloc[-steps:]
            
        if len(df) <= 1:
            return None
        return df

    def preprocess_speed(self, steps, df_merged):
        """Preprocess data for speed prediction."""
        # Calculate MA
        df_merged['MA_x'] = df_merged['player_position_x'].rolling(window=10, center=True).mean()
        df_merged['MA_y'] = df_merged['player_position_y'].rolling(window=10, center=True).mean()
        df_merged = df_merged.dropna()
        df_merged['target_x'] = df_merged['MA_x'].diff()
        df_merged['target_y'] = df_merged['MA_y'].diff()
        df_merged = df_merged.dropna()
        return self._handle_steps_shift(df_merged, steps)
    
    def preprocess_relative_position(self, steps, df_merged):
        # Calculate delta values for future steps
        df_merged['target_x'] = df_merged['player_position_x'].shift(-steps) - df_merged['player_position_x']
        df_merged['target_y'] = df_merged['player_position_y'].shift(-steps) - df_merged['player_position_y']
        return self._handle_steps_shift(df_merged, steps)

    def preprocess_true_distance(self, steps, df_merged):
        # Calculate delta values for future steps
        df_merged['target_x'] = df_merged['delta_x'].shift(-steps)
        df_merged['target_y'] = df_merged['delta_y'].shift(-steps)
        return self._handle_steps_shift(df_merged, steps)

    def preprocess_relative_angle(self, steps, df_merged):
        """Preprocess data for angle prediction."""
        # Calculate delta values for future steps
        df_merged['delta_X'] = pd.to_numeric(df_merged['player_position_x'].shift(-steps) - df_merged['player_position_x'])
        df_merged['delta_Y'] = pd.to_numeric(df_merged['player_position_y'].shift(-steps) - df_merged['player_position_y'])

        df_merged = self._handle_steps_shift(df_merged, steps)

        # Calculate the angle θ using atan2
        df_merged['theta'] = np.arctan2(df_merged['delta_Y'], df_merged['delta_X'])

        # Calculate cosine and sine of the angle θ as targets
        df_merged['target_x'] = np.cos(df_merged['theta']) 
        df_merged['target_y'] = np.sin(df_merged['theta'])

        # Drop temporary columns
        df_merged.drop(columns=['theta', 'delta_X', 'delta_Y'], inplace=True)

        return df_merged
    
    def preprocess_absolute_angle(self, steps, df_merged):
        """Preprocess data for angle prediction."""
        # Calculate delta values for future steps
        df_merged['delta_X'] = pd.to_numeric(df_merged['delta_x'].shift(-steps))
        df_merged['delta_Y'] = pd.to_numeric(df_merged['delta_y'].shift(-steps))

        df_merged = self._handle_steps_shift(df_merged, steps)
        print(f"delta_Y dtype: {df_merged['delta_Y'].dtype}")
        print(f"delta_X dtype: {df_merged['delta_X'].dtype}")
        print(type(np.arctan2))

        # Calculate the angle θ using atan2
        df_merged['theta'] = np.arctan2(df_merged['delta_Y'], df_merged['delta_X'])

        # Calculate cosine and sine of the angle θ as targets
        df_merged['target_x'] = np.cos(df_merged['theta']) 
        df_merged['target_y'] = np.sin(df_merged['theta'])

        # Drop temporary columns
        df_merged.drop(columns=['theta', 'delta_X', 'delta_Y'], inplace=True)

        return df_merged

        

