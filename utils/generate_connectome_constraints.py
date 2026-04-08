import sys

import numpy as np
import pandas as pd
import sys

# TODO this whole thing is kinda slow, can it be parallelized more?

datafile_name = sys.argv[1]
outfile_name = sys.argv[2]
#pre_cell_type_field = 'pre_cell_type'
#post_cell_type_field = 'post_cell_type'
pre_cell_type_field = 'pre_level_1'
post_cell_type_field = 'post_level_1'

print('Loading CSV and parsing usable cell types...')
data = pd.read_csv(datafile_name)
data = data[data['syn_count'] > 5]
unique_cell_ids = data['pre_root_id'].unique()
unique_cell_types = data[pre_cell_type_field].unique()
unique_post_cell_types = data[post_cell_type_field].unique()
unique_cell_types = np.array(unique_cell_types, dtype=np.str_)
unique_post_cell_types = np.array(unique_post_cell_types, dtype=np.str_)
usable_cell_types = np.intersect1d(unique_post_cell_types, unique_cell_types)
# Synapse counts are stored as pre_type x post_type x array of cell pair counts (losing individual cell ID)
syn_counts_per_type_pair = dict()
# Connection counts are stored as pre_type x post_type x array of counts (per individual presynaptic neuron)
connection_counts_per_cell_per_type_pair = dict()
for cell_type in usable_cell_types:
    syn_counts_per_type_pair[cell_type] = dict()
    connection_counts_per_cell_per_type_pair[cell_type] = dict()
    for cell_type_2 in usable_cell_types:
        syn_counts_per_type_pair[cell_type][cell_type_2] = []
        connection_counts_per_cell_per_type_pair[cell_type][cell_type_2] = []

print('Accumulating connection and synapse counts...')
for cell_id in unique_cell_ids:
    entries = data.loc[data['pre_root_id'] == cell_id]
    this_cell_type = ''
    seen_cell_types = []
    for connection in entries.iterrows():
        pre_type = str(connection[1][pre_cell_type_field])
        # This should not change once set, right?
        this_cell_type = pre_type
        post_type = str(connection[1][post_cell_type_field])
        syn_count = connection[1]['syn_count']
        if not pre_type in usable_cell_types:
            break
        if not post_type in usable_cell_types:
            continue
        syn_counts_per_type_pair[pre_type][post_type].append(syn_count)
        # If we haven't seen this post cell type yet for this specific cell, append a new element before incrementing the most recent count
        if not post_type in seen_cell_types:
            seen_cell_types.append(post_type)
            connection_counts_per_cell_per_type_pair[pre_type][post_type].append(0)
        connection_counts_per_cell_per_type_pair[pre_type][post_type][-1] += 1
    # Add 0's for cell types this cell did NOT connect to
    # TODO refactor to do this for all post types before the above (low priority, just for cleanliness/efficiency)
    if this_cell_type in usable_cell_types:
        for post_type in usable_cell_types:
            if not post_type in seen_cell_types:
                connection_counts_per_cell_per_type_pair[this_cell_type][post_type].append(0)

print('Sampling constraint distributions...')
pre_n = 0
post_n = 0
# Sample from distributions to define constraints
rnn_units_per_type = 1
constraints = np.zeros((len(connection_counts_per_cell_per_type_pair.keys()), len(usable_cell_types), rnn_units_per_type, rnn_units_per_type),dtype=np.float32)
pre_n = 0
for pre_cell_type in connection_counts_per_cell_per_type_pair.keys():
    post_n = 0
    for post_cell_type in connection_counts_per_cell_per_type_pair[pre_cell_type].keys():
        curr_counts = np.asarray(connection_counts_per_cell_per_type_pair[pre_cell_type][post_cell_type], dtype=np.float32)
        curr_syn_counts = np.asarray(syn_counts_per_type_pair[pre_cell_type][post_cell_type], dtype=np.float32)
        # TODO why is this sometimes an empty array?
        if curr_syn_counts.size == 0:
            curr_syn_counts = np.zeros(1)
        if curr_counts.size == 0:
            curr_counts = np.zeros(1)
        # Normalize and scale counts relative to the number of neurons in the rnn
        curr_counts /= curr_counts.max() + 0.000001
        curr_counts *= rnn_units_per_type
        curr_counts = np.ceil(curr_counts)
        curr_syn_counts /= curr_syn_counts.max() + 0.000001
        curr_count_dist = np.random.choice(curr_counts, (rnn_units_per_type,))
        curr_constraints = np.zeros((rnn_units_per_type, rnn_units_per_type))
        for curr_unit in range(rnn_units_per_type):
            curr_nonzero_weights = np.random.choice(curr_syn_counts, (int(curr_count_dist[curr_unit]),))
            curr_constraints[curr_unit, :int(curr_count_dist[curr_unit])] = curr_nonzero_weights
        curr_constraints = np.flip(np.sort(curr_constraints, 1),1)
        constraints[pre_n][post_n] = curr_constraints
        post_n += 1
    pre_n += 1


np.save(outfile_name, constraints)