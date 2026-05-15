import jax
import jax.nn as jnn
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


def connectome_loss_nonzero(hh_weights, weight_targets, block_size, k=10.0):
    """Sigmoid loss matching the zero / non-zero structure of weight_targets.

    Sorts |hh_weights| descending within each (row, downstream-block) tile and
    pairs each entry with the corresponding entry in weight_targets, which is
    assumed pre-sorted. Unlike connectome_constraint_loss, the magnitude of
    the target is ignored: this only cares whether the paired entry is zero
    or non-zero. Where the target is zero we penalise the weight for being
    non-zero; where it is non-zero we penalise the weight for being zero. The
    penalty saturates at 1 in the wrong direction and 0 in the right one, so
    far-correct weights stop contributing meaningful gradient.

    Args:
        hh_weights: (n_hidden, n_hidden) RNN hidden-to-hidden kernel.
        weight_targets: (n_hidden, n_hidden) pre-sorted target matrix.
        block_size: number of units per downstream block; must divide n_hidden.
        k: sigmoid steepness. Default 5 is calibrated so the curve bends
            noticeably by |w|≈0.1 (value ≈ 0.24, slope ≈ 2.4) while leaving
            the gradient at |w|=2.0 around 4.5e-4 — well above float32
            epsilon (~1.2e-7), so weights deep in the saturated region still
            receive a recoverable update.

    Returns:
        Scalar mean loss in [0, 1].
    """
    n_hidden = hh_weights.shape[-1]
    n_blocks = n_hidden // block_size
    abs_w = jnp.abs(hh_weights)
    blocked = abs_w.reshape(*abs_w.shape[:-1], n_blocks, block_size)
    sorted_desc = -jnp.sort(-blocked, axis=-1)
    sorted_flat = sorted_desc.reshape(abs_w.shape)
    nonzero_score = 2.0 * jnn.sigmoid(k * sorted_flat) - 1.0
    loss = jnp.where(weight_targets == 0.0, nonzero_score, 1.0 - nonzero_score)

    # DEBUG: spot-check a few pairs. Within each block, sorted_flat and
    # weight_targets are descending, so columns near 0 in block 0 should pair
    # with the largest non-zero targets and columns near block_size-1 should
    # pair with the smallest (most likely zero) targets.
    debug_cols = [0, 1, 2, block_size - 3, block_size - 2, block_size - 1]
    for c in debug_cols:
        jax.debug.print(
            "[connectome_loss_nonzero] row=0 col={c} |w|={w:.4f} "
            "target={t:.4f} loss={l:.4f}",
            c=c,
            w=sorted_flat[0, c],
            t=weight_targets[0, c],
            l=loss[0, c],
        )

    return jnp.mean(loss)
