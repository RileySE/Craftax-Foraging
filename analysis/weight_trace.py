# Script to try and trace connections within a feed-forward network, to test the interconnected-ness of a network

import sys, re
import matplotlib.pyplot as plt
import numpy as np


weight_filename = sys.argv[1]

weight_file = open(weight_filename)

#
count = 0
# Each row is one neuron's weights
for row in weight_file:
    count += 1
    row_arr = re.split(',', row)
    row_np = np.asarray(row_arr, dtype=np.float32)
    row_np = np.abs(row_np)
    total_weights = row_np.size
    weights_not_0 = (row_np > 0.1).sum()
