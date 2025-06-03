import matplotlib.pyplot as plt
import pickle
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# make the plots look nicer and using times new roman font
def set_style():
    """Set matplotlib style parameters for consistent plot formatting"""
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

def plot_rmse(filename):
    """
    Plot RMSE comparison from saved model data.
    
    Args:
        filename (str): Path to the pickle file containing model data
    """
    # Load the data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    rmse_naive = data['naive']
    rmse_model = data['rmse_kernel'] 
    ci = data['ci']
    dt_list = data['dt_list']

    # Get confidence interval bounds
    rmse_l = [i[0] for i in ci]
    rmse_u = [i[1] for i in ci]

    # Create plot
    plt.figure(figsize=(6, 4))
    
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

def plot_contributions(filename, start_dt=0, end_dt=20):
    """
    Plot the contributions of each neuron to the model's RMSE.
    
    Args:
        filename (str): Path to the pickle file containing model data
    """
    bin_size = 0.1
    # Load the data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    models = data['models']
    X_test_std = data['X_test_std']
    dt_list = data['dt_list']

    contributions = []
    for model in models:
        contributions.append(model.coef_ * np.array([X_test_std, X_test_std]))

    contributions = np.abs(np.array(contributions)[start_dt:end_dt,:,:])
    contributions = contributions.flatten()
    contributions = contributions/np.sum(contributions)*1e4
    num_bins = int((max(contributions)-min(contributions))/bin_size)
    plt.figure(figsize=(6, 4))
    plt.hist(contributions, bins=num_bins)
    plt.xlabel('Contribution')
    plt.ylabel('Frequency')
    plt.title(rf'Distribution of Neuron Contributions $\Delta t \in [{dt_list[start_dt]}, {dt_list[end_dt]}]$')
    plt.show()

    
def confidence_interval(data, confidence=0.95):
    n = len(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)  # Standard error of the mean
    margin = 1.96 * std_err  # Z-score for 95% CI
    return margin

def plot_top_contributors(filenames, k_percent=10, start_dt=0, end_dt=20):
    """
    Plot the average and standard deviation of top k% contributors across multiple files.
    
    Args:
        filenames (list): List of paths to pickle files containing model data
        k_percent (float): Percentage of top contributors to analyze
        start_dt (int): Start time step index
        end_dt (int): End time step index
    """
    all_top_contribs = []
    
    # Get contributions from each file
    for filename in filenames:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        models = data['models']
        X_test_std = data['X_test_std']
        
        # Calculate contributions
        contributions = []
        for model in models:
            contributions.append(model.coef_ * np.array([X_test_std, X_test_std]))
            
        contributions = np.abs(np.array(contributions)[start_dt:end_dt,:,:])
        contributions = contributions.flatten()
        contributions = contributions/np.sum(contributions)*1e4
        
        # Get top k% contributors
        k_count = int(len(contributions) * k_percent/100)
        top_k = np.sort(contributions)[-k_count:]
        all_top_contribs.append(top_k)
    
    # Calculate statistics
    means = [np.mean(contribs) for contribs in all_top_contribs]
    margins = [confidence_interval(contribs) for contribs in all_top_contribs]

    print(means)
    print(margins)
    
    # Create bar plot
    plt.figure(figsize=(8, 5))
    x = range(len(filenames))
    plt.bar(x, means, yerr=margins, capsize=5)
    
    # Customize plot
    plt.xlabel('File Index')
    plt.ylabel(f'Average Contribution of Top {k_percent}%')
    plt.title(f'Top {k_percent}% Contributors Across Files')
    plt.xticks(x, [f'File {i+1}' for i in x])
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()

def plot_rolling_contributions(filename, window_size=4, k_percents=[1, 10, 30]):
    """
    Creates a rolling plot of top contributor percentages over time steps.
    
    Args:
        filename (str): Path to pickle file containing model data
        window_size (int): Size of rolling window
        k_percents (list): List of percentages of top contributors to analyze
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
    models = data['models']
    X_test_std = data['X_test_std']
    dt_list = data['dt_list']
    
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    
    for k_idx, k_percent in enumerate(k_percents):
        means = []
        cis = []
        plot_dts = []
        
        # Calculate rolling statistics
        for i in range(len(models) - window_size + 1):
            window_models = models[i:i+window_size]
            
            # Calculate contributions for window
            contributions = []
            for model in window_models:
                contributions.append(model.coef_ * np.array([X_test_std, X_test_std]))
            
            contributions = np.abs(np.array(contributions))
            contributions = contributions.flatten()
            contributions = contributions/np.sum(contributions)*1e4
            
            # Get top k% contributors
            k_count = int(len(contributions) * k_percent/100)
            top_k = np.sort(contributions)[-k_count:]
            
            means.append(np.mean(top_k))
            cis.append(confidence_interval(top_k))
            plot_dts.append(dt_list[i + window_size//2])
            
        means = np.array(means)
        cis = np.array(cis)
        
        plt.fill_between(plot_dts, means - cis, means + cis, 
                        alpha=0.2, color=colors[k_idx])
        plt.plot(plot_dts, means, label=f'Top {k_percent}%',
                color=colors[k_idx])
    
    plt.xlabel('Time Steps')
    plt.ylabel('Average Contribution')
    plt.title('Rolling Top Contributors Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



