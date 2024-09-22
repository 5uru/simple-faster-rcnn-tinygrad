import jax
import jax.numpy as jnp


def _nms_gpu_post(mask, n_bbox, threads_per_block, col_blocks):
    """

    :param mask:
    :param n_bbox:
    :param threads_per_block:
    :param col_blocks:

    """
    def body_fun(carry, i):
        """

        :param carry:
        :param i:

        """
        selection, n_selection, remv = carry
        nblock = i // threads_per_block
        inblock = i % threads_per_block

        not_removed = jnp.logical_not(remv[nblock] & (1 << inblock))
        selection = jax.lax.cond(
            not_removed, lambda: selection.at[n_selection].set(i), lambda: selection
        )
        n_selection = n_selection + not_removed

        index = i * col_blocks
        new_remv = jax.lax.fori_loop(
            nblock,
            col_blocks,
            lambda j, remv: remv.at[j].set(remv[j] | mask[index + j]),
            remv,
        )

        return (selection, n_selection, new_remv), None

    selection = jnp.zeros((n_bbox,), dtype=jnp.int32)
    remv = jnp.zeros((col_blocks,), dtype=jnp.uint64)

    (selection, n_selection, _), _ = jax.lax.scan(
        body_fun, (selection, 0, remv), jnp.arange(n_bbox)
    )

    return selection, n_selection
