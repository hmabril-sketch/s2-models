import numpy as np
import pytest

from s2models.linalg import (
    elementary_divisors,
    get_snf_backend,
    inverse_matrix,
    rank_mod_p,
    set_snf_backend,
    smith_normal_form,
)
from s2models.rings import Z, Z2, Zp

rng = np.random.default_rng(0)


def random_matrix(n, m, lo=-4, hi=5, density=0.6):
    A = rng.integers(lo, hi, size=(n, m))
    A[rng.random((n, m)) > density] = 0
    return A


def check_snf(A, ring):
    s = smith_normal_form(A, ring)
    n, m = A.shape
    PAQ = ring.reduce(ring.matmul(ring.matmul(s.P, ring.reduce(A)), s.Q))
    assert np.array_equal(PAQ, s.D)
    assert np.array_equal(ring.reduce(ring.matmul(s.P, s.P_inv)), np.eye(n, dtype=int))
    assert np.array_equal(ring.reduce(ring.matmul(s.Q, s.Q_inv)), np.eye(m, dtype=int))
    # diagonal and divisibility chain
    off = s.D.copy()
    np.fill_diagonal(off, 0)
    assert not np.any(off)
    d = list(s.divisors)
    assert all(x > 0 for x in d)
    for a, b in zip(d, d[1:]):
        assert b % a == 0
    if ring.is_field:
        assert all(x == 1 for x in d)
    return s


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (5, 3), (6, 6), (7, 4)])
def test_snf_over_z(shape):
    for _ in range(5):
        check_snf(random_matrix(*shape), Z)


@pytest.mark.parametrize("ring", [Z2, Zp(3), Zp(5)])
def test_snf_over_fields(ring):
    for _ in range(5):
        A = random_matrix(6, 8, 0, 7)
        s = check_snf(A, ring)
        assert s.rank == rank_mod_p(A, ring.characteristic)


def test_known_snf():
    A = np.array([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])
    assert smith_normal_form(A).divisors == (2, 6, 12)
    assert elementary_divisors(A) == (2, 6, 12)
    # unsigned incidence matrix of a triangle: non-bipartite -> torsion 2
    T = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    assert elementary_divisors(T) == (1, 1, 2)


def test_backends_agree():
    original = get_snf_backend()
    try:
        for _ in range(10):
            A = random_matrix(rng.integers(2, 9), rng.integers(2, 9))
            set_snf_backend("python")
            py = elementary_divisors(A)
            if original == "pari":
                set_snf_backend("pari")
                assert elementary_divisors(A) == py
            assert py == smith_normal_form(A).divisors
    finally:
        set_snf_backend(None)


def test_inverse():
    U = np.array([[1, 2, 0], [0, 1, 0], [3, 7, 1]])  # unimodular
    Ui = inverse_matrix(U, Z)
    assert np.array_equal(U @ Ui, np.eye(3, dtype=int))
    with pytest.raises(ValueError):
        inverse_matrix(np.array([[2, 0], [0, 1]]), Z)
    B = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    Bi = inverse_matrix(B, Z2)
    assert np.array_equal((B @ Bi) % 2, np.eye(3, dtype=int))


def test_rank_mod_p():
    A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert rank_mod_p(A, 2) == 2
    assert rank_mod_p(A, 3) == 3
