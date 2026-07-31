import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


def _sort_blocks_by_magnitude(mat, block_size):
    """Sort each (row, downstream-block) tile of a 2D matrix by descending
    magnitude, carrying each entry's sign along with it.

    Targets generated with transmitter signs are laid out largest-magnitude
    first with the zero padding at the end of the tile, so magnitude is the key
    that keeps zeros last. For a non-negative matrix this is identical to a
    plain descending sort, so unsigned targets are unaffected.
    """
    n_rows, n_cols = mat.shape
    n_blocks = n_cols // block_size
    blocked = mat.reshape(n_rows, n_blocks, block_size)
    order = np.argsort(-np.abs(blocked), axis=-1, kind='stable')
    sorted_blocks = np.take_along_axis(blocked, order, axis=-1)
    return sorted_blocks.reshape(mat.shape)


def randomize_target_matrix(targets, block_size, seed=0, sort_blocks=True):
    """Return a randomized target matrix preserving the zero fraction and the
    multiset of non-zero values from `targets`.

    The non-zero values are randomly redistributed across the matrix (shuffled
    without replacement), so the global fraction of zeros and the empirical
    distribution of non-zero entries are preserved exactly. Because the whole
    multiset is preserved, a signed target matrix keeps its exact
    excitatory/inhibitory split, just at scrambled positions. When
    ``sort_blocks`` is True each (row, downstream-block) tile is then sorted by
    descending magnitude to match the pre-sorted convention used by
    ``connectome_constraint_loss``, making the result a drop-in replacement for
    the original targets.

    Args:
        targets: ``(n_rows, n_cols)`` array of target weights, which may be
            signed (negative entries are inhibitory targets).
        block_size: number of columns per downstream block; must divide
            ``n_cols``. Unused when ``sort_blocks`` is False, but still
            validated so a bad value is caught either way.
        seed: integer seed for the NumPy RNG used to shuffle the values.
        sort_blocks: sort each tile by descending magnitude (the layout
            ``connectome_constraint_loss`` assumes). Pass False for
            ``connectome_constraint_loss_nonsorted``, whose per-weight targets
            are not block-sorted — sorting the control but not the experimental
            condition would make the two distributionally incomparable.

    Returns:
        ``np.ndarray`` with the same shape and dtype as ``targets``.
    """
    arr = np.asarray(targets)
    if arr.ndim != 2:
        raise ValueError(f"targets must be 2D, got shape {arr.shape}")
    n_rows, n_cols = arr.shape
    if n_cols % block_size != 0:
        raise ValueError(
            f"block_size={block_size} does not divide n_cols={n_cols}"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(arr.flatten()).reshape(arr.shape)
    if not sort_blocks:
        # The flat permutation already placed every value uniformly at random,
        # so the unsorted result is a valid control on its own.
        return shuffled.astype(arr.dtype, copy=False)

    # Sort each downstream block by descending magnitude so the result respects
    # the same block-sorted invariant assumed by connectome_constraint_loss.
    sorted_blocks = _sort_blocks_by_magnitude(shuffled, block_size)
    return sorted_blocks.astype(arr.dtype, copy=False)


def uniform_random_target_matrix(targets, block_size, seed=0, sort_blocks=True):
    """Return a target matrix whose non-zero entries are i.i.d. samples from
    Uniform(0, 1), placed at uniformly random positions, with the same total
    number of non-zero entries as ``targets``.

    The global zero fraction is preserved exactly, as is the number of negative
    entries when ``targets`` is signed: the freshly drawn magnitudes are given
    signs so that the excitatory/inhibitory split matches the matrix this is a
    control for, since a control that silently turned an E/I-balanced target
    into an all-excitatory one would confound sign structure with magnitude
    structure. Unlike :func:`randomize_target_matrix`, the multiset of non-zero
    magnitudes is *not* preserved — they are freshly drawn from U(0, 1) rather
    than shuffled from the original. When ``sort_blocks`` is True each (row,
    downstream-block) tile is then sorted by descending magnitude to match the
    convention used by :func:`connectome_constraint_loss`, making the result a
    drop-in replacement for the original targets.

    Args:
        targets: ``(n_rows, n_cols)`` array of target weights, which may be
            signed. Used only for shape, dtype, the count of non-zero entries,
            and the count of negative entries.
        block_size: number of columns per downstream block; must divide
            ``n_cols``. Unused when ``sort_blocks`` is False, but still
            validated so a bad value is caught either way.
        seed: integer seed for the NumPy RNG.
        sort_blocks: sort each tile by descending magnitude (the layout
            :func:`connectome_constraint_loss` assumes). Pass False for
            :func:`connectome_constraint_loss_nonsorted`, whose per-weight
            targets are not block-sorted — sorting the control but not the
            experimental condition would make the two distributionally
            incomparable.

    Returns:
        ``np.ndarray`` with the same shape and dtype as ``targets``.
    """
    arr = np.asarray(targets)
    if arr.ndim != 2:
        raise ValueError(f"targets must be 2D, got shape {arr.shape}")
    n_rows, n_cols = arr.shape
    if n_cols % block_size != 0:
        raise ValueError(
            f"block_size={block_size} does not divide n_cols={n_cols}"
        )

    rng = np.random.default_rng(seed)
    n_nonzero = int((arr != 0).sum())
    n_negative = int((arr < 0).sum())
    n_total = arr.size

    flat = np.zeros(n_total, dtype=arr.dtype)
    positions = rng.choice(n_total, size=n_nonzero, replace=False)
    # rng.random() samples [0, 1); re-draw any exact-zero samples so the
    # resulting positions remain genuinely non-zero (probability of a hit is
    # ~2**-23 per float32 draw, but we still want a hard guarantee).
    values = rng.random(n_nonzero).astype(arr.dtype)
    while np.any(values == 0):
        zero_idx = np.where(values == 0)[0]
        values[zero_idx] = rng.random(zero_idx.size).astype(arr.dtype)
    if n_negative:
        # Flip exactly as many entries negative as the original had, at
        # positions independent of the drawn magnitudes. Guarded so an unsigned
        # target matrix draws from the RNG exactly as it did before signs
        # existed, keeping old controls reproducible from their seed.
        negative_idx = rng.choice(n_nonzero, size=n_negative, replace=False)
        values[negative_idx] = -values[negative_idx]
    flat[positions] = values
    new_mat = flat.reshape(arr.shape)
    if not sort_blocks:
        # Positions were already drawn uniformly at random, so the unsorted
        # result is a valid control on its own.
        return new_mat.astype(arr.dtype, copy=False)

    sorted_blocks = _sort_blocks_by_magnitude(new_mat, block_size)
    return sorted_blocks.astype(arr.dtype, copy=False)


def connectome_constraint_loss(hh_weights, weight_targets, block_size,
                               exclude_self_weights=False,
                               signed_targets=False):
    """L1 loss between magnitude-sorted hh weights and block-sorted targets.

    Each RNN unit is assigned to a contiguous block along the downstream
    (column) axis: units [0, block_size) belong to block 0, [block_size,
    2*block_size) to block 1, and so on. Within each (row, downstream-block)
    tile, hh_weights is ranked by descending magnitude and compared elementwise
    to weight_targets, which is assumed to already be sorted that way.

    Args:
        hh_weights: (n_hidden, n_hidden) RNN hidden-to-hidden kernel.
        weight_targets: (n_hidden, n_hidden) pre-sorted target matrix.
        block_size: number of units per downstream block; must divide n_hidden.
        exclude_self_weights: when True, zero the diagonal of hh_weights (each
            unit's self connection) before sorting. Self weights then receive
            no constraint gradient and, ranking as zeros, no longer compete
            with other weights for the large target slots of their tile. Each
            zeroed diagonal entry still pairs with the smallest target of its
            tile, so a tile whose targets are all non-zero contributes a
            small parameter-independent offset to the reported loss value.
        signed_targets: when True, weight_targets carries a transmitter sign
            (negative entries are inhibitory) and each weight is compared with
            its own sign kept, so polarity is constrained as well as magnitude:
            a weight of the wrong sign costs |w| + |target| instead of the
            ||w| - |target|| it would cost if only magnitudes were compared.
            Ranking is still by magnitude, so which weight pairs with which
            target does not depend on this flag. When False (the default,
            correct for unsigned magnitude-only targets), only |hh_weights| is
            compared and the sign of each weight is left unconstrained.

    Returns:
        Scalar mean absolute error.
    """
    n_hidden = hh_weights.shape[-1]
    n_blocks = n_hidden // block_size
    if exclude_self_weights:
        self_mask = 1. - jnp.eye(n_hidden, dtype=hh_weights.dtype)
        hh_weights = hh_weights * self_mask
    abs_w = jnp.abs(hh_weights)

    blocked = abs_w.reshape(*abs_w.shape[:-1], n_blocks, block_size)
    if signed_targets:
        # Rank by magnitude exactly as the unsigned path does, but gather the
        # signed weights through that ranking rather than the magnitudes, so a
        # weight sitting at a rank whose target is inhibitory is penalised for
        # being excitatory. The indices are integers, so no gradient flows
        # through the ranking itself — only through the gather, as with sort.
        blocked_signed = hh_weights.reshape(*hh_weights.shape[:-1], n_blocks, block_size)
        order = jnp.argsort(-blocked, axis=-1)
        sorted_desc = jnp.take_along_axis(blocked_signed, order, axis=-1)
    else:
        sorted_desc = -jnp.sort(-blocked, axis=-1)
    sorted_flat = sorted_desc.reshape(abs_w.shape)
    return jnp.mean(jnp.abs(sorted_flat - weight_targets))

# Version of above that does not sort within blocks and instead has a fixed per-weight target value
# (weight [i, j] is always compared to target [i, j]). Also does not need block_size, since
# there is no within-block pairing. Selected via --fixed_connectome_targets.
# signed_targets has the same meaning as in connectome_constraint_loss: leave the
# weights signed so an inhibitory target constrains polarity, instead of comparing
# magnitudes only.
def connectome_constraint_loss_nonsorted(hh_weights, weight_targets, exclude_self_weights=False,
                                         signed_targets=False):
    n_hidden = hh_weights.shape[-1]
    if exclude_self_weights:
        self_mask = 1. - jnp.eye(n_hidden, dtype=hh_weights.dtype)
        hh_weights = hh_weights * self_mask
    if not signed_targets:
        hh_weights = jnp.abs(hh_weights)
    return jnp.mean(jnp.abs(hh_weights - weight_targets))


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

    Signed (transmitter-signed) targets need no special handling here: the loss
    only tests each target against zero and only ever sees |hh_weights|, and
    magnitude-descending targets still keep their zero padding last, so the
    pairing is the same as for unsigned targets. It therefore takes no
    signed_targets flag, unlike connectome_constraint_loss.

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
