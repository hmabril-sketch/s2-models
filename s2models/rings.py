"""Coefficient rings used to build algebraic hypergraphs and s²-models.

The paper works with a commutative Euclidean ring F that is either the ring of
integers Z or a field.  In practice the experiments use F = Z/2Z for building
the s²-models and Z for the Smith Normal Form (SNF) of the resulting matrices.
This module provides a tiny abstraction over Z and the prime fields Z/pZ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_INT64_SAFE = 2**62


@dataclass(frozen=True)
class Ring:
    """A coefficient ring: Z (``characteristic == 0``) or Z/pZ with p prime."""

    characteristic: int = 0

    def __post_init__(self) -> None:
        p = self.characteristic
        if p < 0 or p == 1:
            raise ValueError("characteristic must be 0 (integers) or a prime")
        if p > 1 and any(p % q == 0 for q in range(2, int(p**0.5) + 1)):
            raise ValueError(f"Z/{p}Z is not a field: {p} is not prime")

    # ------------------------------------------------------------------ info
    @property
    def is_field(self) -> bool:
        return self.characteristic > 0

    @property
    def name(self) -> str:
        return "Z" if self.characteristic == 0 else f"Z{self.characteristic}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Ring({self.name})"

    # -------------------------------------------------------------- elements
    def reduce(self, A) -> np.ndarray:
        """Image of an integer matrix through the semiring map N -> F."""
        A = np.asarray(A)
        if self.characteristic == 0:
            return A.copy()
        return np.mod(A.astype(object) if A.dtype == object else A, self.characteristic).astype(np.int64)

    def eye(self, n: int) -> np.ndarray:
        return np.eye(n, dtype=np.int64)

    def zeros(self, shape) -> np.ndarray:
        return np.zeros(shape, dtype=np.int64)

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix product with coefficients in the ring.

        For Z we switch to arbitrary precision (``dtype=object``) whenever the
        product could overflow 64-bit integers.
        """
        A = np.asarray(A)
        B = np.asarray(B)
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"shape mismatch {A.shape} @ {B.shape}")
        p = self.characteristic
        if p > 0:
            # entries are in [0, p); the sum of products is bounded by k p^2
            if A.shape[1] * p * p < _INT64_SAFE:
                return np.mod(A.astype(np.int64) @ B.astype(np.int64), p)
            return np.mod(A.astype(object) @ B.astype(object), p).astype(np.int64)
        if A.dtype != object and B.dtype != object and A.size and B.size:
            bound = int(np.abs(A).max()) * int(np.abs(B).max()) * max(A.shape[1], 1)
            if bound < _INT64_SAFE:
                return A.astype(np.int64) @ B.astype(np.int64)
        R = A.astype(object) @ B.astype(object)
        return _shrink(R)

    def matpow(self, A: np.ndarray, k: int) -> np.ndarray:
        if k < 0:
            raise ValueError("negative powers are not supported")
        R = self.eye(A.shape[0])
        base = self.reduce(A)
        while k:
            if k & 1:
                R = self.matmul(R, base)
            k >>= 1
            if k:
                base = self.matmul(base, base)
        return R

    def inv(self, a: int) -> int:
        """Multiplicative inverse of a unit."""
        p = self.characteristic
        if p == 0:
            if a in (1, -1):
                return a
            raise ZeroDivisionError(f"{a} is not a unit of Z")
        a %= p
        if a == 0:
            raise ZeroDivisionError("0 is not invertible")
        return pow(a, -1, p)


def _shrink(R: np.ndarray) -> np.ndarray:
    """Convert an object array back to int64 when it fits."""
    if R.size == 0:
        return R.astype(np.int64)
    m = max(abs(int(x)) for x in R.flat)
    return R.astype(np.int64) if m < _INT64_SAFE else R


Z = Ring(0)
Z2 = Ring(2)


def Zp(p: int) -> Ring:
    """The prime field Z/pZ."""
    return Ring(p)


def as_ring(ring) -> Ring:
    """Accept ``Ring`` objects or the strings ``"Z"``, ``"Z2"``, ``"Z3"``..."""
    if isinstance(ring, Ring):
        return ring
    if isinstance(ring, int):
        return Ring(ring)
    s = str(ring).strip().upper()
    if s in ("Z", "ZZ", "INTEGERS"):
        return Z
    if s.startswith("Z") and s[1:].isdigit():
        return Ring(int(s[1:]))
    raise ValueError(f"unknown ring {ring!r}")
