import argparse

import numpy as np
import pandas as pd

# TODO this whole thing is kinda slow, can it be parallelized more?

parser = argparse.ArgumentParser(
    description='Generate a connectome constraint .npy from a synapse table.')
parser.add_argument('datafile_name', help='Input CSV of pre/post cell pairs.')
parser.add_argument('outfile_name', help='Output .npy path.')
parser.add_argument(
    '--cell_type_field', default='level_3',
    help="Cell type column, without the pre_/post_ prefix: 'level_3' reads "
         "pre_level_3/post_level_3, 'cell_type' reads pre_cell_type/post_cell_type. "
         "(default: %(default)s)")
parser.add_argument(
    '--rnn_units_per_type', type=int, default=16,
    help='RNN units to sample per cell type (default: %(default)s).')
parser.add_argument(
    '--sort_constraints', action=argparse.BooleanOptionalAction, default=True,
    help='Sort each row of a block by descending magnitude, which is the layout '
         'connectome_constraint_loss assumes. Pass --no-sort_constraints only '
         'for runs using ppo_rnn.py --fixed_connectome_targets, whose per-weight '
         'loss does no within-block sorting. (default: sorted)')
parser.add_argument(
    '--transmitter_signs', action=argparse.BooleanOptionalAction, default=True,
    help='Sign each sampled constraint value by the neurotransmitter of the '
         'connection it was drawn from: ACH positive, GABA/GLUT negative. '
         'Connections carrying any other transmitter are dropped from the CSV '
         'before any distribution is built, so they contribute to neither the '
         'synapse-count nor the connection-count statistics. Pass '
         '--no-transmitter_signs for the original unsigned, transmitter-agnostic '
         'constraints. (default: signed)')
parser.add_argument(
    '--syn_count_normalization',
    choices=['per_pair', 'global_percentile'],
    default='per_pair',
    help="How synapse counts are scaled into constraint magnitudes. 'per_pair' "
         "divides each pre/post type pair's counts by that pair's own maximum, so "
         "every block's largest constraint is ~1 regardless of how strong its "
         "connections are. 'global_percentile' divides every pair by one shared "
         "count, the --syn_count_percentile percentile of the per-connection "
         "counts pooled across all usable type pairs, and clips values above 1. "
         "That keeps the relative strength of connections between blocks, while "
         "the few strongest connections saturate rather than setting the scale "
         "for everything else. Only magnitudes change: the RNG draws, and hence "
         "which entries are non-zero and their signs, are identical for a given "
         "--seed, except that under global_percentile the sort keeps entries "
         "clipped to the same value in draw order, so their signs can trade places "
         "within a row. (default: %(default)s)")
parser.add_argument(
    '--syn_count_percentile', type=float, default=99.,
    help="Percentile of the per-connection synapse counts, pooled across all "
         "usable type pairs, that maps to 1 under --syn_count_normalization "
         "global_percentile. (default: %(default)s)")
parser.add_argument(
    '--aggregate_neuropils', action=argparse.BooleanOptionalAction, default=True,
    help='Sum the synapse counts of CSV rows that share a pre/post root-ID pair '
         '(the CSV has one row per neuropil a pair connects in) into a single '
         'connection, before the syn_count > 5 threshold, so each pair counts '
         'once with its total synapse count. Pass --no-aggregate_neuropils for '
         'the original per-row behaviour, where each of a pair\'s neuropil rows '
         'is a separate connection and is thresholded on its own. '
         '(default: aggregated)')
parser.add_argument('--seed', type=int, default=0,
                    help='RNG seed (default: %(default)s).')
args = parser.parse_args()

datafile_name = args.datafile_name
outfile_name = args.outfile_name
pre_cell_type_field = 'pre_' + args.cell_type_field
post_cell_type_field = 'post_' + args.cell_type_field
rnn_units_per_type = args.rnn_units_per_type
sort_constraints = args.sort_constraints
transmitter_signs = args.transmitter_signs
syn_count_normalization = args.syn_count_normalization
aggregate_neuropils = args.aggregate_neuropils
nt_field = 'nt_type'
nt_positives = ['ACH']
nt_negatives = ['GABA', 'GLUT']
nt_sign_field = 'nt_sign'

# One Generator for the whole run, so the output is reproducible from --seed.
rng = np.random.default_rng(args.seed)
np.random.seed(args.seed)

print('Loading CSV and parsing usable cell types...')
data = pd.read_csv(datafile_name)
if aggregate_neuropils:
    # Every column but neuropil and syn_count describes the pair (its neurons'
    # cell types and transmitter), so the pair's first row stands in for it,
    # with the summed count; its neuropil is left as that row's and is unused.
    # Summing before the threshold below keeps the synapses of a pair's
    # sub-threshold neuropils, and keeps pairs whose synapses are all spread
    # thinly across neuropils.
    pair_fields = ['pre_root_id', 'post_root_id']
    pair_rows = data.groupby(pair_fields, sort=False)
    if transmitter_signs:
        n_mixed = int((pair_rows[nt_field].nunique() > 1).sum())
        if n_mixed:
            raise SystemExit(f'{n_mixed} root-ID pairs have rows with different '
                             f'{nt_field}s, so their summed connection has no single sign')
    n_rows_before = len(data)
    data = data.assign(syn_count=pair_rows['syn_count'].transform('sum')).drop_duplicates(pair_fields)
    print(f'Aggregated {n_rows_before} neuropil rows into {len(data)} pre/post root-ID pairs.')
data = data[data['syn_count'] > 5]
if transmitter_signs:
    # Only the transmitters we model carry a usable sign; connections labelled
    # with anything else (DA/SER/OCT/...) are dropped here rather than at
    # sampling time so that they are excluded from every distribution below,
    # including the per-cell connection counts and the cell type vocabulary.
    n_rows_before = len(data)
    data = data[data[nt_field].isin(nt_positives + nt_negatives)].copy()
    data[nt_sign_field] = np.where(
        data[nt_field].isin(nt_positives), 1., -1.).astype(np.float32)
    print(f'Dropped {n_rows_before - len(data)} of {n_rows_before} connections '
          f'with an unmodelled {nt_field}.')
unique_cell_types = data[pre_cell_type_field].unique()
unique_post_cell_types = data[post_cell_type_field].unique()
unique_cell_types = np.array(unique_cell_types, dtype=np.str_)
unique_post_cell_types = np.array(unique_post_cell_types, dtype=np.str_)
usable_cell_types = np.intersect1d(unique_post_cell_types, unique_cell_types)
# Synapse counts are stored as pre_type x post_type x array of cell pair counts (losing individual cell ID)
syn_counts_per_type_pair = dict()
# Transmitter signs are stored the same way, index-aligned with the synapse
# counts above, so a sampled weight keeps the sign of the connection it came from
syn_signs_per_type_pair = dict()
# Connection counts are stored as pre_type x post_type x array of counts (per individual presynaptic neuron)
connection_counts_per_cell_per_type_pair = dict()
for cell_type in usable_cell_types:
    syn_counts_per_type_pair[cell_type] = dict()
    syn_signs_per_type_pair[cell_type] = dict()
    connection_counts_per_cell_per_type_pair[cell_type] = dict()
    for cell_type_2 in usable_cell_types:
        syn_counts_per_type_pair[cell_type][cell_type_2] = []
        syn_signs_per_type_pair[cell_type][cell_type_2] = []
        connection_counts_per_cell_per_type_pair[cell_type][cell_type_2] = []

print('Accumulating connection and synapse counts...')
total_connections = 0
# groupby(sort=False) visits the cells in order of first appearance, each
# with its rows in CSV order, so this is the order a .loc scan of the whole
# table per unique() cell gave, in one pass instead of one per cell.
for cell_id, entries in data.groupby('pre_root_id', sort=False):
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
        if transmitter_signs:
            syn_signs_per_type_pair[pre_type][post_type].append(
                connection[1][nt_sign_field])
        # If we haven't seen this post cell type yet for this specific cell, append a new element before incrementing the most recent count
        if not post_type in seen_cell_types:
            seen_cell_types.append(post_type)
            connection_counts_per_cell_per_type_pair[pre_type][post_type].append(0)
        connection_counts_per_cell_per_type_pair[pre_type][post_type][-1] += 1
        total_connections += 1
    # Add 0's for cell types this cell did NOT connect to
    # TODO refactor to do this for all post types before the above (low priority, just for cleanliness/efficiency)
    if this_cell_type in usable_cell_types:
        for post_type in usable_cell_types:
            if not post_type in seen_cell_types:
                connection_counts_per_cell_per_type_pair[this_cell_type][post_type].append(0)

print('Total connections:', total_connections)
if syn_count_normalization == 'global_percentile':
    # Pooled over the accumulated pairs rather than the whole CSV, so pairs with
    # an unusable pre or post type can't set the scale. float32 to match the
    # dtype of the counts it divides.
    all_syn_counts = np.asarray(
        [count for post_counts in syn_counts_per_type_pair.values()
         for counts in post_counts.values() for count in counts], dtype=np.float64)
    print('Global max synapse count:', all_syn_counts.max(initial=0.))
    global_syn_count_percentile = np.float32(
        np.percentile(all_syn_counts, args.syn_count_percentile)
        if all_syn_counts.size else 0.)
    print(f'Global {args.syn_count_percentile:g}th percentile synapse count: '
          f'{global_syn_count_percentile} ({100. * np.mean(all_syn_counts > global_syn_count_percentile):.2f}% '
          f'of {all_syn_counts.size} connections clipped to 1)')
print('Sampling constraint distributions...')
pre_n = 0
post_n = 0
# Sample from distributions to define constraints
constraints = np.zeros((len(connection_counts_per_cell_per_type_pair.keys()), len(usable_cell_types), rnn_units_per_type, rnn_units_per_type),dtype=np.float32)
pre_n = 0
for pre_cell_type in connection_counts_per_cell_per_type_pair.keys():
    post_n = 0
    for post_cell_type in connection_counts_per_cell_per_type_pair[pre_cell_type].keys():
        curr_counts = np.asarray(connection_counts_per_cell_per_type_pair[pre_cell_type][post_cell_type], dtype=np.float32)
        curr_syn_counts = np.asarray(syn_counts_per_type_pair[pre_cell_type][post_cell_type], dtype=np.float32)
        curr_syn_signs = np.asarray(syn_signs_per_type_pair[pre_cell_type][post_cell_type], dtype=np.float32)
        # TODO why is this sometimes an empty array?
        if curr_syn_counts.size == 0:
            curr_syn_counts = np.zeros(1)
            # Sign is irrelevant for a zero-valued weight, but keep the two
            # arrays the same length so the shared draw indices stay valid.
            curr_syn_signs = np.ones(1)
        if curr_counts.size == 0:
            curr_counts = np.zeros(1)
        # Normalize and scale counts relative to the number of neurons in the rnn
        curr_counts /= curr_counts.max() + 0.000001
        curr_counts *= rnn_units_per_type
        curr_counts = np.ceil(curr_counts)
        if syn_count_normalization == 'global_percentile':
            curr_syn_counts = np.minimum(
                curr_syn_counts / (global_syn_count_percentile + 0.000001), 1.)
        else:
            curr_syn_counts /= curr_syn_counts.max() + 0.000001
        curr_count_dist = np.random.choice(curr_counts, (rnn_units_per_type,))
        curr_constraints = np.zeros((rnn_units_per_type, rnn_units_per_type))
        for curr_unit in range(rnn_units_per_type):
            # Draw indices rather than values so each sampled synapse count can be
            # paired with the transmitter sign of the connection it was drawn from,
            # which reproduces the sign distribution of this pre/post type pair.
            # np.random.choice consumes the RNG identically either way, so the
            # unsigned output is unchanged from the value-sampling version.
            curr_draw_idx = np.random.choice(curr_syn_counts.size, (int(curr_count_dist[curr_unit]),))
            curr_nonzero_weights = curr_syn_counts[curr_draw_idx]
            if transmitter_signs:
                curr_nonzero_weights = curr_nonzero_weights * curr_syn_signs[curr_draw_idx]
            curr_constraints[curr_unit, :int(curr_count_dist[curr_unit])] = curr_nonzero_weights
        if sort_constraints:
            # Descending by magnitude, so the zero padding stays at the end of the
            # row for signed targets too. Identical to a descending value sort when
            # the constraints are unsigned.
            curr_sort_idx = np.argsort(-np.abs(curr_constraints), axis=1, kind='stable')
            curr_constraints = np.take_along_axis(curr_constraints, curr_sort_idx, axis=1)
        else:
            # Permute the order of the weights so each downstream neuron has the
            # same odds of receiving input. Skipped when sorting, which would
            # discard the permutation anyway.
            for curr_unit in range(rnn_units_per_type):
                curr_constraints[curr_unit] = rng.permutation(curr_constraints[curr_unit], axis=0)
        constraints[pre_n][post_n] = curr_constraints
        post_n += 1
    pre_n += 1


n_nonzero = int(np.count_nonzero(constraints))
n_positive = int((constraints > 0).sum())
if n_nonzero:
    print(f'Non-zero constraints: {n_nonzero} of {constraints.size} '
          f'({100. * n_nonzero / constraints.size:.2f}%), of which '
          f'{100. * n_positive / n_nonzero:.2f}% positive / '
          f'{100. * (n_nonzero - n_positive) / n_nonzero:.2f}% negative.')

np.save(outfile_name, constraints)