import jax
import jax.numpy as jnp
import pytest

from model.utils.nms._nms_gpu_post_py import _nms_gpu_post


def test_nms_gpu_post():
    mask = jnp.array([0b101, 0b010, 0b001], dtype=jnp.uint64)
    n_bbox = 3
    threads_per_block = 1
    col_blocks = 3

    expected_selection = jnp.array([0, 1], dtype=jnp.int32)
    expected_n_selection = 2

    selection, n_selection = _nms_gpu_post(mask, n_bbox, threads_per_block, col_blocks)
    print("Actual IoU:", selection[:n_selection])
    print("Expected IoU:", expected_selection)
    print("Actual IoU:", n_selection)
    print("Expected IoU:", expected_n_selection)
    assert jnp.array_equal(selection[:n_selection], expected_selection)
    assert n_selection == expected_n_selection


if __name__ == "__main__":
    pytest.main()
