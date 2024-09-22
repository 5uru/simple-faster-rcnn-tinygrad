import jax.numpy as jnp
import numpy.testing as npt
import pytest

from model.utils.bbox_tools import bbox2loc
from model.utils.bbox_tools import bbox_iou
from model.utils.bbox_tools import generate_anchor_base
from model.utils.bbox_tools import loc2bbox


def test_loc2bbox():
    """ """
    src_bbox = jnp.array([[0, 0, 10, 10], [10, 20, 30, 40]], dtype=jnp.float32)
    loc = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.float32)
    dst_bbox = loc2bbox(src_bbox, loc)
    assert jnp.all(dst_bbox == jnp.array([[0, 0, 10, 10], [10, 20, 30, 40]],
                                         dtype=jnp.float32))


def test_bbox2loc():
    """ """
    src_bbox = jnp.array([[0, 0, 10, 10], [10, 20, 30, 40]], dtype=jnp.float32)
    dst_bbox = jnp.array([[0, 0, 10, 10], [10, 20, 30, 40]], dtype=jnp.float32)
    loc = bbox2loc(src_bbox, dst_bbox)
    assert jnp.all(
        loc == jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.float32))


def test_bbox_iou():
    """ """
    bbox_a = jnp.array([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=jnp.float32)
    bbox_b = jnp.array([[1, 1, 2, 2], [0, 0, 3, 3]], dtype=jnp.float32)

    expected_iou = jnp.array([[0.25, 0.44444445], [0.25, 0.44444445]],
                             dtype=jnp.float32)
    result = bbox_iou(bbox_a, bbox_b)

    print("Actual IoU:", result)
    print("Expected IoU:", expected_iou)

    npt.assert_allclose(result, expected_iou, rtol=1e-5)


def test_generate_anchor_base_small():
    """ """
    expected_anchors = jnp.array(
        [
            [-0.62132025, -2.7426405, 3.6213202, 5.7426405],
            [-1.5, -1.5, 4.5, 4.5],
            [-2.7426405, -0.62132025, 5.7426405, 3.6213202],
            [-2.7426405, -6.985281, 5.7426405, 9.985281],
            [-4.5, -4.5, 7.5, 7.5],
            [-6.985281, -2.7426405, 9.985281, 5.7426405],
            [-6.985281, -15.470562, 9.985281, 18.470562],
            [-10.5, -10.5, 13.5, 13.5],
            [-15.470562, -6.985281, 18.470562, 9.985281],
        ],
        dtype=jnp.float32,
    )

    result = generate_anchor_base(base_size=3,
                                  ratios=jnp.array([0.5, 1, 2]),
                                  anchor_scales=jnp.array([2, 4, 8]))

    print("Actual Anchors:", result)
    print("Expected Anchors:", expected_anchors)
    assert jnp.allclose(result, expected_anchors, rtol=1e-5)
