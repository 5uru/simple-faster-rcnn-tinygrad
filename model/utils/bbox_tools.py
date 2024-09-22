import jax
import jax.numpy as jnp
from jax import Array
from jax import jit


@jit
def loc2bbox(src_bbox: Array, loc: Array) -> Array:
    """

    :param src_bbox: Array:
    :param loc: Array:

    """
    if src_bbox.shape[0] == 0:
        return jnp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, jnp.newaxis] + src_ctr_y[:, jnp.newaxis]
    ctr_x = dx * src_width[:, jnp.newaxis] + src_ctr_x[:, jnp.newaxis]
    h = jnp.exp(dh) * src_height[:, jnp.newaxis]
    w = jnp.exp(dw) * src_width[:, jnp.newaxis]

    dst_bbox = jnp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox = dst_bbox.at[:, 0::4].set(ctr_y - 0.5 * h)
    dst_bbox = dst_bbox.at[:, 1::4].set(ctr_x - 0.5 * w)
    dst_bbox = dst_bbox.at[:, 2::4].set(ctr_y + 0.5 * h)
    dst_bbox = dst_bbox.at[:, 3::4].set(ctr_x + 0.5 * w)

    return dst_bbox


@jit
def bbox2loc(src_bbox: Array, dst_bbox: Array) -> Array:
    """

    :param src_bbox: Array:
    :param dst_bbox: Array:

    """
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = jnp.finfo(height.dtype).eps
    height = jnp.maximum(height, eps)
    width = jnp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = jnp.log(base_height / height)
    dw = jnp.log(base_width / width)

    return jnp.vstack((dy, dx, dh, dw)).transpose()


def bbox_iou(bbox_a: Array, bbox_b: Array) -> Array:
    """

    :param bbox_a: Array:
    :param bbox_b: Array:

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = jnp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = jnp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = jnp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = jnp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = jnp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


@jit
def generate_anchor_base(
    base_size=16, ratios=jnp.array([0.5, 1, 2]), anchor_scales=jnp.array([8, 16, 32])
):
    """

    :param base_size:  (Default value = 16)
    :param ratios:  (Default value = jnp.array([0.5)
    :param 1:
    :param 2]):
    :param anchor_scales:  (Default value = jnp.array([8)
    :param 16:
    :param 32]):

    """

    py = base_size / 2.0
    px = base_size / 2.0

    @jax.vmap
    def compute_anchor(ratio, scale):
        """

        :param ratio:
        :param scale:

        """
        h = base_size * scale * jnp.sqrt(ratio)
        w = base_size * scale * jnp.sqrt(1.0 / ratio)
        return jnp.array([py - h / 2.0, px - w / 2.0, py + h / 2.0, px + w / 2.0])

    ratios_mesh, scales_mesh = jnp.meshgrid(ratios, anchor_scales)
    ratios_flat = ratios_mesh.flatten()
    scales_flat = scales_mesh.flatten()

    anchor_base = compute_anchor(ratios_flat, scales_flat)

    return anchor_base
