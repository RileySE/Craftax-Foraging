import jax
import jax.numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact

# For utility functions - functions called more than once in meaningfully different parts of the codebase
# With the additional constraint that the functions make no reference (i.e. don't import from) any Craftax code


def get_distance_map(position, map_size):
    dist_x = jnp.abs(jnp.arange(0, map_size[0]) - position[0])
    dist_x = jnp.expand_dims(dist_x, axis=1)
    dist_x = jnp.tile(dist_x, (1, map_size[1]))

    dist_y = jnp.abs(jnp.arange(0, map_size[1]) - position[1])
    dist_y = jnp.expand_dims(dist_y, axis=0)
    dist_y = jnp.tile(dist_y, (map_size[0], 1))

    coords = jnp.stack([dist_x, dist_y], axis=-1)

    def _euclid_distance(x):
        return jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    dist = jax.vmap(jax.vmap(_euclid_distance))(coords)

    return dist


def weighted_choice(key, a, shape=(), p=None):
    """jax.random.choice(key, a, shape, p=p) for a 1-D array `a` (sampling
    with replacement), returning bit-for-bit the same samples.

    jax.random.choice inverts the CDF with searchsorted's default binary
    search, a lax.scan that runs on the GPU as a while loop costing a host
    round trip per iteration (14 for a 96x96 map). This runs the very same
    binary search unrolled, so it has no loop. (searchsorted's loop-free
    compare_all method would not be identical: XLA's GPU cumsum can step back
    by one ulp, and that method counts every entry below the draw.)
    """
    (p_arr,) = promote_dtypes_inexact(p)
    p_cuml = jnp.cumsum(p_arr)
    r = p_cuml[-1] * (1 - jax.random.uniform(key, shape, dtype=p_cuml.dtype))
    ind = jnp.searchsorted(p_cuml, r, method="scan_unrolled").astype(int)
    return jnp.take(jnp.asarray(a), ind, 0).reshape(shape)
