# Plots a histogram of weights for one neuron, for structured sparsity reasons.

import sys, re
import matplotlib.pyplot as plt
import numpy as np

def make_histogram(arr, title, xlabel='Weight Magnitude', ylabel='% of Weights', normalization_factor=1, n_bins=20):
    hist, bins = np.histogram(arr, bins=n_bins)
    # Express counts as percentage of all weights
    hist = (hist / normalization_factor) * 100.
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.yticks([100, 10, 1, 0.1], ['100', '10', '1', '0.1'])
    plt.stairs(hist, bins, fill=True)
    plt.show()

weight_filename = sys.argv[1]

weight_file = open(weight_filename)

layer_size = 512
max_weights = []
non_0_counts = []
non_0_per_layer = [[]]
count = 0
all_weights_count = 0
# Each row is one neuron's weights
for row in weight_file:
    count += 1
    row_arr = re.split(',', row)
    row_np = np.asarray(row_arr, dtype=np.float32)
    row_np = np.abs(row_np)
    total_weights = row_np.size
    all_weights_count += total_weights
    max_weights.append(row_np.max())
    weights_not_0 = (row_np > 0.01).sum()
    non_0_counts.append(weights_not_0)
    non_0_per_layer[-1].append(weights_not_0)
    if count % layer_size == 0:
        non_0_per_layer.append([])

    if False and count % 256 == 0:
        print('Plotting neuron #', count)
        make_histogram(row_np, 'Weight distribution of a single neuron', normalization_factor=total_weights, n_bins=50)


max_weights_np = np.asarray(max_weights)

make_histogram(max_weights_np, 'Max weight of each neuron', ylabel='% of Neurons', normalization_factor=count, n_bins=50)

non_0_np = np.asarray(non_0_counts)

make_histogram(non_0_np, 'Fraction of non-0 weights per neuron', '# Non-0 weights', '% of Neurons', count, 100)

for layer in range(len(non_0_per_layer)):
    make_histogram(non_0_per_layer[layer], 'Fraction of non-0 weights per neuron, layer ' + str(layer+1), '# Non-0 weights', '% of Neurons', layer_size, 100)
