"""Content-only diagonal reuse preserves the literal GPy full-dot result."""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from basat import FixedPolynomialKernel

module = importlib.import_module("basat.models.FixedPolynomialKernel")


@pytest.fixture(autouse=True)
def empty_cache():
    module._gpy_diagonal_bytes.cache_clear()
    yield
    module._gpy_diagonal_bytes.cache_clear()


def points():
    return torch.tensor([[.1, .7, .3], [.8, .2, .5], [.4, .6, .9], [.35, .1, .25], [.23, .54, .91]], dtype=torch.float64)


def kernel(**changes):
    return FixedPolynomialKernel(input_dimension=3, scale=changes.pop("scale", 1.11),
        bias=changes.pop("bias", 2.84), variance=changes.pop("variance", 1.32),
        order=changes.pop("order", 6), **changes)


def literal(x, z=None, *, scale=1.11, bias=2.84, variance=1.32, order=6):
    left = x.numpy()[:, np.arange(x.shape[1])].astype("float")
    right = left if z is None else z.numpy()[:, np.arange(z.shape[1])].astype("float")
    return (((scale*np.dot(left, right.T))+bias)**order*variance).diagonal().copy()


def test_exact_literal_first_result_cache_hit_and_fresh_instance(monkeypatch):
    x = points()
    expected = literal(x)
    original, calls = np.dot, []
    def record(left, right):
        calls.append((left.shape, right.shape, np.shares_memory(left, right)))
        return original(left, right)
    monkeypatch.setattr(np, "dot", record)
    first = kernel()(x, diag=True)
    second = kernel()(x.clone(), diag=True)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert calls == [((5, 3), (3, 5), True)]
    assert module._gpy_diagonal_bytes.cache_info().hits == 1
    assert first.data_ptr() != second.data_ptr()


def test_return_mutation_and_input_mutation_cannot_poison_cache():
    x = points()
    expected = literal(x)
    first = kernel()(x, diag=True)
    first.fill_(-123.)
    np.testing.assert_array_equal(kernel()(x, diag=True), expected)
    x[0, 0] = torch.nextafter(x[0, 0], torch.tensor(float("inf"), dtype=torch.float64))
    np.testing.assert_array_equal(kernel()(x, diag=True), literal(x))
    assert module._gpy_diagonal_bytes.cache_info().misses == 2


@pytest.mark.parametrize("parameter", ["scale", "bias", "variance", "order"])
def test_every_parameter_bit_and_integer_kind_is_in_key(parameter):
    x, instance = points(), kernel()
    instance(x, diag=True)
    buffer = getattr(instance, "fixed_"+parameter)
    buffer.copy_(torch.nextafter(buffer, torch.tensor(float("inf"), dtype=torch.float64)))
    actual = instance(x, diag=True)
    expected = literal(x, scale=float(instance.fixed_scale), bias=float(instance.fixed_bias),
        variance=float(instance.fixed_variance), order=int(instance.fixed_order) if bool(instance.fixed_order_is_integer) else float(instance.fixed_order))
    np.testing.assert_array_equal(actual, expected)
    assert module._gpy_diagonal_bytes.cache_info().misses == 2
    kernel(order=6.0)(x, diag=True)
    assert module._gpy_diagonal_bytes.cache_info().misses == 3


def test_source_sharing_and_both_cross_inputs_are_in_key(monkeypatch):
    x, z = points(), points()
    original, sharing = np.dot, []
    def record(left, right):
        sharing.append(np.shares_memory(left, right))
        return original(left, right)
    monkeypatch.setattr(np, "dot", record)
    instance = kernel()
    a = instance(x, diag=True)
    b = instance(x, z, diag=True)
    z[0, 0] += .01
    c = instance(x, z, diag=True)
    np.testing.assert_array_equal(a, literal(x))
    np.testing.assert_array_equal(b, literal(x, points()))
    np.testing.assert_array_equal(c, literal(x, z))
    assert sharing[:3] == [True, False, False]
    assert module._gpy_diagonal_bytes.cache_info().misses == 3


def test_shape_and_bit_pattern_are_in_key():
    x, instance = points(), kernel()
    instance(x, diag=True)
    instance(x[:4], diag=True)
    signed_zero = x.clone()
    signed_zero[0, 0] = 0.
    instance(signed_zero, diag=True)
    signed_zero[0, 0] = -0.
    instance(signed_zero, diag=True)
    assert module._gpy_diagonal_bytes.cache_info().misses == 4


def test_shared_right_raw_bits_are_retained_even_when_torch_equal():
    x, instance = points(), kernel()
    x[0, 0] = 0.
    z = x.clone()
    z[0, 0] = -0.
    assert torch.equal(x, z) and x.numpy().tobytes() != z.numpy().tobytes()
    first = instance.forward(x, x, diag=True, _basat_self_covariance=True)
    second = instance.forward(x, z, diag=True, _basat_self_covariance=True)
    np.testing.assert_array_equal(first, literal(x))
    np.testing.assert_array_equal(second, literal(x))
    assert module._gpy_diagonal_bytes.cache_info().misses == 2


def test_equivalent_noncontiguous_input_uses_the_same_canonical_content():
    x, instance = points(), kernel()
    padded = torch.zeros((len(x), 6), dtype=torch.float64)
    padded[:, ::2] = x
    strided = padded[:, ::2]
    assert not strided.is_contiguous() and torch.equal(x, strided)
    np.testing.assert_array_equal(instance(x, diag=True), literal(x))
    np.testing.assert_array_equal(instance(strided, diag=True), literal(strided))
    assert module._gpy_diagonal_bytes.cache_info().misses == 1
    assert module._gpy_diagonal_bytes.cache_info().hits == 1


def test_cache_stores_no_full_gram_and_is_bounded():
    instance = kernel()
    for i in range(12):
        instance(points()+i*.001, diag=True)
    assert module._gpy_diagonal_bytes.cache_info().currsize == 8
    x = points()
    result = module._gpy_diagonal_bytes(tuple(x.shape), x.numpy().tobytes(), tuple(x.shape),
        x.numpy().tobytes(), True, instance.fixed_scale.numpy().tobytes(), instance.fixed_bias.numpy().tobytes(),
        instance.fixed_variance.numpy().tobytes(), instance.fixed_order.numpy().tobytes(), True)
    assert type(result) is bytes and len(result) == len(x)*8


def test_large_inputs_bypass_cache_without_changing_source_arithmetic(monkeypatch):
    monkeypatch.setattr(module, "_DIAGONAL_CACHE_MAX_INPUT_BYTES", 1)
    x, instance = points(), kernel()
    np.testing.assert_array_equal(instance(x, diag=True), literal(x))
    np.testing.assert_array_equal(instance(x, diag=True), literal(x))
    assert module._gpy_diagonal_bytes.cache_info().currsize == 0


def test_cache_retains_source_full_dot_not_rowwise_reduction():
    # These three-coordinate values expose last-bit full-BLAS vs rowwise sum
    # differences. Caching must retain literal Kdiag, not rewrite that formula.
    x = torch.arange(1, 301, dtype=torch.float64).reshape(100, 3)/301
    expected = literal(x)
    rowwise = ((1.11*np.sum(x.numpy()*x.numpy(), axis=1)+2.84)**6)*1.32
    assert np.any(expected != rowwise)
    np.testing.assert_array_equal(kernel()(x, diag=True), expected)
    np.testing.assert_array_equal(kernel()(x, diag=True), expected)
