import numpy as np
import pandas as pd

data = pd.read_csv('../Downloads/fafb_pre_post_cell_types.csv')
data = data[data['syn_count'] > 5]
unique_cell_ids = data['pre_root_id'].unique()
unique_cell_types = data['pre_cell_type'].unique()
unique_post_cell_types = data['post_cell_type'].unique()
# Synapse counts are stored as pre_type x post_type x array of cell pair counts (losing individual cell ID)
syn_counts_per_type_pair = dict()
# Connection counts are stored as pre_type x post_type x array of counts (per individual presynaptic neuron)
connection_counts_per_cell_per_type_pair = dict()
for cell_type in unique_cell_types:
    syn_counts_per_type_pair[cell_type] = dict()
    connection_counts_per_cell_per_type_pair[cell_type] = dict()
    for cell_type_2 in unique_post_cell_types:
        syn_counts_per_type_pair[cell_type][cell_type_2] = []
        connection_counts_per_cell_per_type_pair[cell_type][cell_type_2] = []

for cell_id in unique_cell_ids:
    entries = data.loc[data['pre_root_id'] == cell_id]
    seen_cell_types = []
    for connection in entries.iterrows():
        pre_type = connection[1]['pre_cell_type']
        post_type = connection[1]['post_cell_type']
        syn_count = connection[1]['syn_count']
        syn_counts_per_type_pair[pre_type][post_type].append(syn_count)
        # If we haven't seen this post cell type yet for this specific cell, append a new element before incrementing the most recent count
        if not post_type in seen_cell_types:
            seen_cell_types.append(post_type)
            connection_counts_per_cell_per_type_pair[pre_type][post_type].append(0)
        connection_counts_per_cell_per_type_pair[pre_type][post_type][-1] += 1

breakpoint()