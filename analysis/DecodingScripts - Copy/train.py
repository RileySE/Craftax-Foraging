from sklearn.metrics import root_mean_squared_error
import warnings
from rich.progress import track
from preprocessing import DataPreprocessor
from models import correlate
import argparse
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from models import RMSE_angle

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
    parser = argparse.ArgumentParser(description='Train models for player movement prediction')
    #parser.add_argument('--data-path', type=str, required=True, help='Path to the data folder')
    parser.add_argument('--mode', type=str, choices=['speed', 'position', 'distance', 'rel_angle', 'abs_angle'], 
                       default='position', help='Prediction mode')
    parser.add_argument('--custom-split', action='store_true', 
                       help='Use custom train-test split by episode')
    parser.add_argument('--max-step', type=int, default=500,
                       help='Maximum timestep to predict') 
    parser.add_argument('--step-size', type=int, default=10,
                       help='Step size for timestep range')
    parser.add_argument('--file-identifier', type=str, default='',
                       help='Identifier for the file')

    args = parser.parse_args()

    # Initialize preprocessor
    file_names_hstates = ['FelixData/hstates_34816_0_spi.csv', 'FelixData/hstates_36864_0_spi.csv']#, 'FelixData/hstates_38912_0_spi.csv', 'FelixData/hstates_40960_0_spi.csv', 'FelixData/hstates_43008_0_spi.csv', 'FelixData/hstates_45056_0_spi.csv']
    file_names_scalars = ['FelixData/scalars_34816_0_spi.csv', 'FelixData/scalars_36864_0_spi.csv']#, 'FelixData/scalars_38912_0_spi.csv', 'FelixData/scalars_40960_0_spi.csv', 'FelixData/scalars_43008_0_spi.csv', 'FelixData/scalars_45056_0_spi.csv']
    #file_names_hstates = 'FelixData/hstates_10240_0.csv'
    #file_names_scalars = 'FelixData/scalars_10240_0.csv'
    preprocessor = DataPreprocessor(file_names_hstates,file_names_scalars)
    episode_df = preprocessor.merge_files()
    print(len(episode_df))
    
    # Generate timestep range
    range1 = np.arange(-args.max_step, -20, args.step_size)
    range2 = np.arange(-20, 20, 1)
    range3 = np.arange(20, args.max_step + 1, args.step_size)
    dt_list = np.concatenate((range1, range2, range3))

    # Initialize storage lists
    scores_model = []
    rmse_model = []
    rmse_naive = []
    ci = []
    models = []
    # Train models for each timestep
    for i in track(dt_list):
        df_merged_list = []
        if isinstance(episode_df, list):
            for df in episode_df:
                if args.mode == 'speed':
                    df_merged = preprocessor.preprocess_speed(i, df)
                elif args.mode == 'position':
                    df_merged = preprocessor.preprocess_relative_position(i, df)
                elif args.mode == 'distance':
                    df_merged = preprocessor.preprocess_true_distance(i, df)
                elif args.mode == 'rel_angle':
                    df_merged = preprocessor.preprocess_relative_angle(i, df)
                elif args.mode == 'abs_angle':
                    df_merged = preprocessor.preprocess_absolute_angle(i, df)
                df_merged_list.append(df_merged)
            merged_df = pd.concat(df_merged_list, axis=0, ignore_index=True)
        else:
            if args.mode == 'speed':
                merged_df = preprocessor.preprocess_speed(i, episode_df)
            elif args.mode == 'position':
                merged_df = preprocessor.preprocess_relative_position(i, episode_df)
            elif args.mode == 'distance':
                merged_df = preprocessor.preprocess_true_distance(i, episode_df)
            elif args.mode == 'rel_angle':
                merged_df = preprocessor.preprocess_relative_angle(i, episode_df)
            elif args.mode == 'abs_angle':
                merged_df = preprocessor.preprocess_absolute_angle(i, episode_df)
        
        model, X_test, y_test = correlate(merged_df, custom_train_test_split=args.custom_split)
        models.append(model)
        if i == 0:
            X_test_std = np.std(X_test, axis=0)
        
        # Model predictions and metrics
        predictions = model.predict(X_test)
        scores_model.append(model.score(X_test, y_test))
        if args.mode == 'rel_angle' or args.mode == 'abs_angle':
            rmse = RMSE_angle(y_test, predictions)
        else:
            rmse = root_mean_squared_error(y_test, predictions)
        rmse_model.append(rmse)
        
        # Calculate confidence interval
        errors = y_test - predictions
        squared_errors = errors ** 2
        se = np.sqrt(np.var(squared_errors, ddof=1) / len(y_test))
        
        # 95% CI
        z = 1.96  # For 95% CI
        ci.append((rmse - z * se, rmse + z * se))
        
        if args.mode == 'rel_angle' or args.mode == 'abs_angle':
            rmse_naive.append(RMSE_angle(y_test, np.zeros_like(y_test)))
        else:
            rmse_naive.append(root_mean_squared_error(y_test, np.zeros_like(y_test)))

    
    # Plot RMSE comparison
    plt.figure(figsize=(10, 6))
    rmse_l = [i[0] for i in ci]
    rmse_u = [i[1] for i in ci]

# Add confidence interval shading
    plt.fill_between(dt_list, rmse_l, rmse_u, color='blue', alpha=0.2, label=r"$95\%$ CI")
    plt.plot(dt_list, rmse_model, label='Model RMSE')
    plt.plot(dt_list, rmse_naive, label='Naive RMSE')
    
    plt.xlabel('Time Steps')
    plt.ylabel('RMSE')
    plt.title('Model vs Naive RMSE Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare data for storage
    data_to_store = {
        'naive': rmse_naive,
        'rmse_kernel': rmse_model,
        'ci': ci,
        'models': models,
        'dt_list': dt_list,
        'X_test_std': X_test_std
    }

    # Create filename based on parameters
    filename = f'out/data_{args.file_identifier}_{args.max_step}_{args.step_size}_{args.mode}'
    if args.custom_split:
        filename += '_custom_split'
    filename += '.pkl'

    # Save the data to a file
    with open(filename, 'wb') as f:
        pickle.dump(data_to_store, f)

if __name__ == '__main__':
    # python src/train.py --mode distance --custom-split --file-identifier "test"
    main()