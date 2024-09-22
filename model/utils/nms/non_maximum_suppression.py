import jax
import jax.numpy as jnp


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs.

    This method checks each bounding box sequentially and selects the bounding
    box if the Intersection over Unions (IoUs) between the bounding box and the
    previously selected bounding boxes is less than :obj:`thresh`. This method
    is mainly used as postprocessing of object detection.
    The bounding boxes are selected from ones with higher scores.
    If :obj:`score` is not provided as an argument, the bounding box
    is ordered by its index in ascending order.

    :param bbox: Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
    :type bbox: array
    :param thresh: Threshold of IoUs.
    :type thresh: float
    :param score: An array of confidences whose shape is :math:`(R,)`. (Default value = None)
    :type score: array
    :param limit: The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible. (Default value = None)
    :type limit: int
    :rtype: array

    """
    return _non_maximum_suppression_gpu(bbox, thresh, score, limit)


@jax.jit
def _non_maximum_suppression_gpu(bbox, thresh, score=None, limit=None):
    """

    :param bbox:
    :param thresh:
    :param score:  (Default value = None)
    :param limit:  (Default value = None)

    """
    if len(bbox) == 0:
        return jnp.zeros((0, ), dtype=jnp.int32)

    n_bbox = bbox.shape[0]

    if score is not None:
        order = jnp.argsort(score)[::-1].astype(jnp.int32)
    else:
        order = jnp.arange(n_bbox, dtype=jnp.int32)

    sorted_bbox = bbox[order]

    threads_per_block = 64
    col_blocks = int(jnp.ceil(n_bbox / threads_per_block))

    mask = _compute_mask(sorted_bbox, thresh, n_bbox, threads_per_block,
                         col_blocks)
    selection, n_selection = _nms_gpu_post(mask, n_bbox, threads_per_block,
                                           col_blocks)

    selection = selection[:n_selection]
    selection = order[selection]

    if limit is not None:
        selection = selection[:limit]

    return selection


@jax.jit
def _compute_mask(bbox, thresh, n_bbox, threads_per_block, col_blocks):
    """

    :param bbox:
    :param thresh:
    :param n_bbox:
    :param threads_per_block:
    :param col_blocks:

    """

    def body_fun(i, mask):
        """

        :param i:
        :param mask:

        """
        cur_box = bbox[i]
        ious = jax.vmap(lambda x: calculate_iou(cur_box, x))(bbox[i + 1:])
        t = jnp.where(ious >= thresh, jnp.uint64(1), jnp.uint64(0))
        t = jnp.pad(t, (0, threads_per_block - len(t) % threads_per_block))
        t = t.reshape(-1, threads_per_block)
        t = jnp.bitwise_or.reduce(t << jnp.arange(threads_per_block), axis=1)
        mask = mask.at[i * col_blocks:(i + 1) * col_blocks].set(t)
        return mask

    mask = jnp.zeros((n_bbox * col_blocks, ), dtype=jnp.uint64)
    mask = jax.lax.fori_loop(0, n_bbox, body_fun, mask)
    return mask


@jax.jit
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
        selection = jax.lax.cond(not_removed,
                                 lambda: selection.at[n_selection].set(i),
                                 lambda: selection)
        n_selection = n_selection + not_removed

        index = i * col_blocks
        new_remv = jax.lax.fori_loop(
            nblock,
            col_blocks,
            lambda j, remv: remv.at[j].set(remv[j] | mask[index + j]),
            remv,
        )

        return (selection, n_selection, new_remv), None

    selection = jnp.zeros((n_bbox, ), dtype=jnp.int32)
    remv = jnp.zeros((col_blocks, ), dtype=jnp.uint64)

    (selection, n_selection, _), _ = jax.lax.scan(body_fun,
                                                  (selection, 0, remv),
                                                  jnp.arange(n_bbox))

    return selection, n_selection


@jax.jit
def calculate_iou(bbox_a, bbox_b):
    """

    :param bbox_a:
    :param bbox_b:

    """
    top = jnp.maximum(bbox_a[0], bbox_b[0])
    bottom = jnp.minimum(bbox_a[2], bbox_b[2])
    left = jnp.maximum(bbox_a[1], bbox_b[1])
    right = jnp.minimum(bbox_a[3], bbox_b[3])

    height = jnp.maximum(bottom - top, 0.0)
    width = jnp.maximum(right - left, 0.0)
    area_i = height * width

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    return area_i / (area_a + area_b - area_i)
