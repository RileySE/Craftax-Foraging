# Plots a histogram of weights for one neuron, for structured sparsity reasons.

import sys, re
import matplotlib.pyplot as plt
import numpy as np

weight_filename = sys.argv[1]

weight_file = open(weight_filename)

max_weights = []
non_0_counts = []
count = 0
# Each row is one neuron's weights
for row in weight_file:
    count += 1
    row_arr = re.split(',', row)
    row_np = np.asarray(row_arr, dtype=np.float32)
    row_np = np.abs(row_np)
    max_weights.append(row_np.max())
    weights_not_0 = (row_np > 0.1).sum()
    non_0_counts.append(weights_not_0)
    if False and count % 256 == 0:
        print('Plotting neuron #', count)
        plt.figure()
        plt.title('Weight histogram of one neuron- Late training')
        plt.xlabel('Weight magnitude')
        plt.ylabel('# Weights')
        plt.hist(row_np, bins=20)
        plt.show()

plt.figure()
plt.title('Max weights of all neurons- Late training')
plt.xlabel('Weight magnitude')
plt.ylabel('# Weights')
plt.hist(max_weights, bins=20)
plt.show()

plt.figure()
plt.title('Number of non-0 weights per neuron- Late training')
plt.xlabel('# Non-0 weights')
plt.ylabel('# Neurons')
plt.hist(non_0_counts, bins=50, range=(0,50))
plt.show()