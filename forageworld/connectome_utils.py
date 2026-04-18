import jax.numpy as jnp


def connectome_constraint_loss(hh_weights, weight_targets, block_size):
    """L1 loss between magnitude-sorted hh weights and block-sorted targets.

    Each RNN unit is assigned to a contiguous block along the downstream
    (column) axis: units [0, block_size) belong to block 0, [block_size,
    2*block_size) to block 1, and so on. Within each (row, downstream-block)
    tile, |hh_weights| is sorted in descending order and compared elementwise
    to weight_targets, which is assumed to already be sorted that way.

    Args:
        hh_weights: (n_hidden, n_hidden) RNN hidden-to-hidden kernel.
        weight_targets: (n_hidden, n_hidden) pre-sorted target matrix.
        block_size: number of units per downstream block; must divide n_hidden.

    Returns:
        Scalar mean absolute error.
    """
    n_hidden = hh_weights.shape[-1]
    n_blocks = n_hidden // block_size
    abs_w = jnp.abs(hh_weights)
    blocked = abs_w.reshape(*abs_w.shape[:-1], n_blocks, block_size)
    sorted_desc = -jnp.sort(-blocked, axis=-1)
    sorted_flat = sorted_desc.reshape(abs_w.shape)
    return jnp.mean(jnp.abs(sorted_flat - weight_targets))
