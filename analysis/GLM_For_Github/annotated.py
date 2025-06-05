import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration for thresholds and parameters
config = {
    "max_axes_pos": 100,             # Max position on x or y axis (assumed square-shaped arena)
    "num_quads_wide": 10,           # Number of quadrants along one axis
    "eat_mean_threshold": 0.5,        # Threshold for eating events to qualify as a patch
    "drink_mean_threshold": 3,      # Threshold for drinking events to qualify as a patch
    "visit_mean_threshold": 5,      # Number of timesteps in a quadrant to count as a patch
    "revisit_time_threshold": 14,   # Time away from a patch to count as a revisitation
    "memory_threshold": 75,
    "ep_len_normalization": 2000 # Threshold for memory tracking (for angle calculations)
}
config["num_quads"] = config["num_quads_wide"] ** 2 # Calculate the number of quadrants
config["quadrant_size"] = config['max_axes_pos'] / config['num_quads_wide']

def save_merged_files(dir_raw: str, dir_merge: str):
    """
    Loads and merges all CSV files in the specified directory into a single DataFrame.
    """
    files = [os.path.join(dir_raw, f) for f in os.listdir(dir_raw) if f.endswith('.csv')]
    df_list = [pd.read_csv(f,low_memory=False) for f in files]
    i = 0
    for f in files:
        df_list[i]['id'] = df_list[i]['episode_id'].values
        df_list[i] = df_list[i][df_list[i]['id'] != df_list[i]['id'][0]]
        i = i + 1    
    
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(os.path.join(dir_merge, 'Merge.csv'), index = False)

def load_merged_file(filepath: str) -> pd.DataFrame:
    """
    Saves the merged DataFrame to the specified file path.
    """
    file_name_load2 = os.path.join(filepath, 'Merge.csv')
    return pd.read_csv(file_name_load2,low_memory=False)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies necessary preprocessing such as rolling averages for position coordinates.
    """

    df['food'] = df['food'].astype(float)
    df['drink'] = df['drink'].astype(float)
    df['# action'] = df['# action'].astype(float)
    df['player_position_x'] = df['player_position_x'].astype(float)
    df['player_position_y'] = df['player_position_y'].astype(float)
    df['is_sleeping'] = df['is_sleeping'].astype(float)

    df['player_position_x'] = df['player_position_x'].astype(float)
    df['player_position_y'] = df['player_position_y'].astype(float)
    df['x_avg'] = df['player_position_x'].rolling(window=5, min_periods=1, center=True).mean()
    df['y_avg'] = df['player_position_y'].rolling(window=5, min_periods=1, center=True).mean()
    df['time'] = range(1, len(df) + 1)
    df['is_eat'] = ((df['# action'] == 5) & (df['food'].diff() > 0)).astype(int)
    df['is_drink'] = ((df['# action'] == 5) & (df['drink'].diff() > 0)).astype(int)
                    
    return df

def distance_and_angle_from_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns to the DataFrame for the Euclidean distance, Manhattan distance, and angle from the start position.
    The start position is assumed to be the coordinates at the first row of the DataFrame.
    """
    # Assume the first row is the start position
    df['player_position_x'] = df['player_position_x'].astype(float)
    df['player_position_y'] = df['player_position_y'].astype(float)
    start_x = df.iloc[0]['player_position_x']
    start_y = df.iloc[0]['player_position_y']

    # Calculate Euclidean distance from start for each point
    df['euclidean_distance_from_start'] = np.sqrt((df['player_position_x'] - start_x) ** 2 + (df['player_position_y'] - start_y) ** 2)

    # Calculate Manhattan distance from start for each point
    df['manhattan_distance_from_start'] = np.abs(df['player_position_x'] - start_x) + np.abs(df['player_position_y'] - start_y)

    # Calculate angle from start for each point
    df['angle_from_start'] = np.arctan2(df['player_position_y'] - start_y, df['player_position_x'] - start_x)

    return df

def identify_patches(df: pd.DataFrame) -> pd.DataFrame: # Does not use this for now as I am trying new one
    """
    Identifies patches based on the agent's interaction with food, drink, and time spent in each quadrant.
    Labels new patches and revisited patches.
    """
    # Calculate quadrant information
    df['player_position_x'] = df['player_position_x'].astype(float)
    df['player_position_y'] = df['player_position_y'].astype(float)
    df['quadrant'] = ((df['player_position_x'] // (config['max_axes_pos'] / config['num_quads_wide'])).astype(int) +
                      (df['player_position_y'] // (config['max_axes_pos'] / config['num_quads_wide'])).astype(int) * config['num_quads_wide'])

    # Initialize patch labels
    df['is_patch'] = 0
    df['quad_visit_count'] = 0
    df['# drink'] = 0
    df['# sleep'] = 0
    df['# eat'] = 0

    # Group by quadrants and apply patch criteria
    for quad in df['quadrant'].unique():
        quad_data = df[df['quadrant'] == quad] # it is the dataframe when the agent is in the quadrant
        eat_events = ((quad_data["# action"] == 5) & (quad_data["food"].diff() > 0)).sum()
        drink_events = ((quad_data["# action"] == 5) & (quad_data["drink"].diff() > 0)).sum()
        drink_events = ((quad_data["# action"] == 5) & (quad_data["drink"].diff() > 0)).sum()

        visit_count = len(quad_data)
        df.loc[df['quadrant'] == quad, 'quad_visit_count'] = visit_count
        df.loc[df['quadrant'] == quad, '# drink'] = drink_events
        df.loc[df['quadrant'] == quad, '# eat'] = eat_events
        df.loc[df['quadrant'] == quad, '# sleep'] = quad_data['is_sleeping'].sum()
        normalization_factor = len(df['player_position_y'])/config['ep_len_normalization']
        #print(len(df['player_position_y']))
        #print(normalization_factor)
        #or drink_events >= (config['drink_mean_threshold']*normalization_factor) ) 
        
        if eat_events >= (config['eat_mean_threshold']*normalization_factor) and visit_count > (config['visit_mean_threshold']*normalization_factor):
            df.loc[df['quadrant'] == quad, 'is_patch'] = 1
            
    return df

# Need to rethink the way I do this. I should create to lists with patch_start_idx and patch_end_idx. Then I should iterate over the rows between thos idxs
# What I do below is iterating over all indexes, an then for each idx iterate over the idx_start to end.

def label_revisit_patches(df: pd.DataFrame) -> pd.DataFrame:
    df['is_revisit_patch'] = 0
    df['is_new_patch'] = 0
    # df['eat_drink_patch'] = 0
    visited_patches = set()

    patch_start_idx = []
    patch_end_idx = []

    in_patch = False
    current_quadrant = 1000
    current_start_idx = None

    # Step 1: Identify patch start and end indices
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        if row['is_patch'] == 1 and current_quadrant != row['quadrant'] and in_patch:
            # From patch to patch
            in_patch = True
            patch_start_idx.append(current_start_idx)
            patch_end_idx.append(idx)
            current_start_idx = idx

        if row['is_patch'] == 1 and current_quadrant != row['quadrant'] and not in_patch:
            # Entering a patch
            in_patch = True
            current_start_idx = idx  

        if row['is_patch'] == 0 and current_quadrant != row['quadrant'] and in_patch:
            # Exiting a patch
            patch_start_idx.append(current_start_idx)
            patch_end_idx.append(idx)
            in_patch = False
        
        current_quadrant = row['quadrant']
    
    # Handle case where the last patch continues until the end of the data
    if in_patch:
        patch_start_idx.append(current_start_idx)
        patch_end_idx.append(len(df))

    # Step 2: Iterate over patch ranges and mark eat_drink_patch, new, or revisit
    for start_idx, end_idx in zip(patch_start_idx, patch_end_idx):
        patch_data = df.iloc[start_idx:end_idx]

        # Check if the patch contains any eat or drink event
        #if patch_data[['is_eat', 'is_drink']].any().any():
        if patch_data[['is_eat']].any().any():
            # Mark the entire patch as an eat_drink_patch
            # df.iloc[start_idx:end_idx, df.columns.get_loc('eat_drink_patch')] = 1
            
            current_quadrant = patch_data['quadrant'].iloc[0]  # Use the first row's quadrant for simplicity

            # Determine if it's a revisit
            if current_quadrant in visited_patches:
                df.iloc[start_idx:end_idx, df.columns.get_loc('is_revisit_patch')] = 1
            else:
                df.iloc[start_idx:end_idx, df.columns.get_loc('is_new_patch')] = 1
                visited_patches.add(current_quadrant)

    # Step 3: Add new column 'new_patch_eat_drink' and 'revisit_patch_eat_drink' which is True if a patch has an eat or drink event
    #df['new_patch_eat_drink'] = ((df['is_new_patch'] == 1) & ((df['is_eat'] == 1) | (df['is_drink'] == 1))).astype(int)
    #df['revisit_patch_eat_drink'] = ((df['is_revisit_patch'] == 1) & ((df['is_eat'] == 1) | (df['is_drink'] == 1))).astype(int)
    df['new_patch_eat_drink'] = ((df['is_new_patch'] == 1) & ((df['is_eat'] == 1) )).astype(int)
    df['revisit_patch_eat_drink'] = ((df['is_revisit_patch'] == 1) & ((df['is_eat'] == 1) )).astype(int)

    return df





    

def calculate_distance_angle_manhattan_to_quadrant(x, y, quadrant_x_min, quadrant_x_max, quadrant_y_min, quadrant_y_max):
    """
    Calculates the Euclidean distance, Manhattan distance, and the angle from point (x, y) to the boundaries of
    a quadrant defined by (quadrant_x_min, quadrant_x_max, quadrant_y_min, quadrant_y_max).
    The angle is the direction from (x, y) to the closest boundary point of the quadrant.
    """
    # Find the closest x and y boundary positions
    if x < quadrant_x_min:
        closest_x = quadrant_x_min
    elif x > quadrant_x_max:
        closest_x = quadrant_x_max
    else:
        closest_x = x  # Inside the x-boundaries

    if y < quadrant_y_min:
        closest_y = quadrant_y_min
    elif y > quadrant_y_max:
        closest_y = quadrant_y_max
    else:
        closest_y = y  # Inside the y-boundaries

    # Calculate Euclidean distance
    distance = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)

    # Calculate Manhattan distance
    manhattan_distance = abs(x - closest_x) + abs(y - closest_y)

    # Calculate angle (atan2 gives angle in radians)
    angle = np.arctan2(closest_y - y, closest_x - x)

    return distance, angle, manhattan_distance


def calculate_distances_and_angles_to_patches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds six columns to the DataFrame: distance_to_next_patch, distance_to_previous_patch,
    angle_to_next_patch, angle_to_previous_patch, manhattan_distance_to_next_patch,
    and manhattan_distance_to_previous_patch.
    These columns store the Euclidean distance, Manhattan distance, and angle to the next and previous patches
    in the episode. If the agent is in a patch, the distances are set to zero and angles are set to NaN.
    """
    # Initialize new columns with NaN
    df['distance_to_next_patch'] = np.nan
    df['distance_to_previous_patch'] = np.nan
    df['angle_to_next_patch'] = np.nan
    df['angle_to_previous_patch'] = np.nan
    df['manhattan_distance_to_next_patch'] = np.nan
    df['manhattan_distance_to_previous_patch'] = np.nan

    # Get the indices of the rows where there is a patch
    patch_indices = df[df['is_patch'] == 1].index

    # If there are no patches, return the dataframe as it is
    if len(patch_indices) == 0:
        return df

    # Define the quadrant size
    quadrant_size = config['quadrant_size']

    # Calculate distance, angle, and Manhattan distance to the previous patch
    last_patch_idx = None
    for idx in df.index:
        if df.loc[idx, 'is_patch'] == 1:
            # If agent is in a patch, set distance to zero and angle to NaN
            df.at[idx, 'distance_to_previous_patch'] = 0
            df.at[idx, 'angle_to_previous_patch'] = np.nan
            df.at[idx, 'manhattan_distance_to_previous_patch'] = 0
        elif last_patch_idx is not None:
            current_x = df.loc[idx, 'player_position_x']
            current_y = df.loc[idx, 'player_position_y']

            # Get the quadrant information for the last patch
            last_patch_quad = df.loc[last_patch_idx, 'quadrant']
            last_patch_x_min = (last_patch_quad % config['num_quads_wide']) * quadrant_size
            last_patch_x_max = last_patch_x_min + quadrant_size - 1
            last_patch_y_min = (last_patch_quad // config['num_quads_wide']) * quadrant_size
            last_patch_y_max = last_patch_y_min + quadrant_size - 1

            # Calculate the distance, angle, and Manhattan distance to the closest point on the last patch quadrant
            distance_to_prev, angle_to_prev, manhattan_distance_to_prev = calculate_distance_angle_manhattan_to_quadrant(
                current_x, current_y, last_patch_x_min, last_patch_x_max, last_patch_y_min, last_patch_y_max
            )
            df.at[idx, 'distance_to_previous_patch'] = distance_to_prev
            df.at[idx, 'angle_to_previous_patch'] = angle_to_prev
            df.at[idx, 'manhattan_distance_to_previous_patch'] = manhattan_distance_to_prev
        
        if idx in patch_indices:
            last_patch_idx = idx

    # Calculate distance, angle, and Manhattan distance to the next patch
    next_patch_idx = None
    for idx in reversed(df.index):
        if df.loc[idx, 'is_patch'] == 1:
            # If agent is in a patch, set distance to zero and angle to NaN
            df.at[idx, 'distance_to_next_patch'] = 0
            df.at[idx, 'angle_to_next_patch'] = np.nan
            df.at[idx, 'manhattan_distance_to_next_patch'] = 0
        elif next_patch_idx is not None:
            current_x = df.loc[idx, 'player_position_x']
            current_y = df.loc[idx, 'player_position_y']

            # Get the quadrant information for the next patch
            next_patch_quad = df.loc[next_patch_idx, 'quadrant']
            next_patch_x_min = (next_patch_quad % config['num_quads_wide']) * quadrant_size
            next_patch_x_max = next_patch_x_min + quadrant_size - 1
            next_patch_y_min = (next_patch_quad // config['num_quads_wide']) * quadrant_size
            next_patch_y_max = next_patch_y_min + quadrant_size - 1

            # Calculate the distance, angle, and Manhattan distance to the closest point on the next patch quadrant
            distance_to_next, angle_to_next, manhattan_distance_to_next = calculate_distance_angle_manhattan_to_quadrant(
                current_x, current_y, next_patch_x_min, next_patch_x_max, next_patch_y_min, next_patch_y_max
            )
            df.at[idx, 'distance_to_next_patch'] = distance_to_next
            df.at[idx, 'angle_to_next_patch'] = angle_to_next
            df.at[idx, 'manhattan_distance_to_next_patch'] = manhattan_distance_to_next
        
        if idx in patch_indices:
            next_patch_idx = idx

    return df

def save_episode_to_csv(episode_df: pd.DataFrame, episode_id, dir_out: str):
    # Construct the output file path
    output_filename = f"annotated_{episode_id}.csv"
    output_filepath = os.path.join(dir_out, output_filename)
    
    # Write the episode dataframe to CSV in the 'dir_out' folder
    episode_df.to_csv(output_filepath, index=False)

# Step 4: Visualization Functions
def plot_agent_path(df: pd.DataFrame):
    """
    Plots the path of the agent with color-coded patches.
    """
    
    df['player_position_x'] = df['player_position_x'].astype(float)
    df['player_position_y'] = df['player_position_y'].astype(float)
    
    #plt.scatter(df['x_avg'][:1024], df['y_avg'][:1024], c=df['is_patch'][:1024], cmap='coolwarm')
    plt.scatter(df['player_position_x'][:100], df['player_position_y'][:100], c=df['is_patch'][:100], cmap='coolwarm')
    #plt.scatter(df['x_avg'][:100], df['y_avg'][:100], c=df['is_patch'][:100], cmap='coolwarm')

    # Set x and y ticks to match quadrant size
    plt.xticks(np.arange(-0.5, config['max_axes_pos'] + config['quadrant_size']-0.5, config['quadrant_size']))
    plt.yticks(np.arange(-0.5, config['max_axes_pos'] + config['quadrant_size']-0.5, config['quadrant_size']))
    
    # Add grid lines to match quadrant size
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

    plt.title("Agent Movement with Patch Annotations")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.colorbar(label="Patch Status")
    plt.show()

# Main script workflow
def main():
    
    MainDir = '/Baseline/' 
    
    for DirIdx in range(0,5):
        print(DirIdx)
        # Directory for raw data and output files
              

        dir_raw = MainDir + str(DirIdx) + "/"
        dir_merged = MainDir + str(DirIdx) + "M/"
        dir_out = MainDir + str(DirIdx) + "O/"
        
        CHECK_FOLDER_raw = os.path.isdir(dir_raw)
        CHECK_FOLDER_merged = os.path.isdir(dir_merged)
        CHECK_FOLDER_out = os.path.isdir(dir_out)
        
        if not CHECK_FOLDER_raw:
            os.makedirs(dir_raw)
            print("created folder : ", dir_raw)
        else:
            print(dir_raw, "folder already exists.")
            
        if not CHECK_FOLDER_merged:
            os.makedirs(dir_merged)
            print("created folder : ", dir_merged)
        else:
            print(dir_merged, "folder already exists.")
            
        if not CHECK_FOLDER_out:
            os.makedirs(dir_out)
            print("created folder : ", dir_out)
        else:
            print(dir_out, "folder already exists.")    
    
    # you have to manually load in the data files you want analyzed, we did >100k in Baseline_Longer for the paper, load into
    # 0/, 1/, ... etc. 
    for DirIdx in range(0,5):
        print(DirIdx)               
        dir_raw = MainDir + str(DirIdx) + "/"
        dir_merged = MainDir + str(DirIdx) + "M/"
        dir_out = MainDir + str(DirIdx) + "O/"
        # Step 1: Load csv files and save merged data as 'Merged.csv'
        save_merged_files(dir_raw, dir_merged) 

        # Step 2: Load the merged csv file to DataFrame
        df = load_merged_file(dir_merged)
        df = df[~df['episode_id'].isin(['episode_id'])]
        
        # Step 3: Iterate over each unique episode ID
        for episode_id in df['episode_id'].unique():
            # Step 3.1: Filter dataframe for the current episode
            episode_df = df[df['episode_id'] == episode_id].copy()

            # Step 3.2: Preprocess data - can add whatever if neccesarry
            episode_df = preprocess_data(episode_df)

            # Step 3.3: Calculate distances from start pos
            episode_df = distance_and_angle_from_start(episode_df)

            # Step 3.4: Identify patches
            episode_df = identify_patches(episode_df)

            # Step 3.5: Revisit patches
            episode_df = label_revisit_patches(episode_df)

            # Step 3.6: Calculate distances to previous and next patches
            episode_df = calculate_distances_and_angles_to_patches(episode_df)

            # Step 3.7: Save episode dataframe to CSV
            save_episode_to_csv(episode_df, episode_id, dir_out)

            # Step 3.8: Can plot episode if wanted


if __name__ == "__main__":
    main()
