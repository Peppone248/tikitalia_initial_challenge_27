"""Machine-independent arithmetic for the NN and the decoder (RULES 3: "the same file").

Why: a floating-point matrix product sums in an order that depends on the CPU and on the BLAS
build. The differences are ~1e-16, but SGD training is chaotic and amplifies them: on two PCs
the v0 pipeline gave files differing on 1.3% of the frames. Everything here uses only operations
whose result IEEE-754 fixes exactly (+ - * / sqrt, rounding, integer arithmetic):

    qmatmul   operands scaled by a power of two (from the exponent of their max, np.frexp) and
              rounded to int64 with BITS significant bits; the integer product is exact and
              independent of the summation order; the conversion back is correctly rounded.
              |a|,|b| < 2^BITS, sum over n terms < n * 2^(2 BITS) <= 2^62 for n <= 2048.
    det_exp   exp(x) by range reduction + polynomial: only + - * and ldexp, no libm / SIMD exp
    quantize  values rounded to a 2^-bits grid: differences below the grid vanish, and sums of
              grid values below 2^(53-bits) are exact, whatever the order
"""

import numpy as np

BITS = 26
LN2 = 0.6931471805599453
_EXP_COEF = [1.0 / np.prod(np.arange(1, k + 1), dtype=float) for k in range(14)]


def _to_int(a):
    a = np.asarray(a, dtype=np.float64)
    m = float(np.max(np.abs(a))) if a.size else 0.0
    if m == 0.0:
        return np.zeros(a.shape, dtype=np.int64), 1.0
    e = np.frexp(m)[1]                         # m < 2^e
    scale = np.ldexp(1.0, BITS - e)
    return np.rint(a * scale).astype(np.int64), scale


def qmatmul(a, b):
    """a @ b with int64 arithmetic (exact, order-independent), ~BITS bits of relative precision."""
    if a.shape[-1] > 2048:
        raise ValueError("qmatmul: inner dimension too large for int64")
    ai, sa = _to_int(a)
    bi, sb = _to_int(b)
    return (ai @ bi).astype(np.float64) / (sa * sb)


def det_exp(x):
    """exp(x) for x <= 0 (softmax after subtracting the max), machine-independent."""
    x = np.maximum(np.asarray(x, dtype=np.float64), -700.0)
    k = np.floor(x / LN2 + 0.5)
    r = x - k * LN2                             # |r| <= ln2 / 2 (+ rounding)
    p = np.full(r.shape, _EXP_COEF[-1])
    for c in reversed(_EXP_COEF[:-1]):          # Horner, 14 terms: error < 1e-16 on |r| < 0.35
        p = p * r + c
    return np.ldexp(p, k.astype(np.int64))


def quantize(a, bits):
    s = np.ldexp(1.0, int(bits))
    return np.rint(np.asarray(a, dtype=np.float64) * s) / s


def exact_mean_std(xq, bits):
    """Mean and std per column of values on a 2^-bits grid, via exact integer sums."""
    s = np.ldexp(1.0, int(bits))
    xi = np.rint(xq * s).astype(np.int64)
    n = xi.shape[0]
    tot = xi.sum(axis=0)
    sq = (xi * xi).sum(axis=0)
    mean = tot.astype(np.float64) / (n * s)
    var = (sq.astype(np.float64) / n - (tot.astype(np.float64) / n) ** 2) / (s * s)
    return mean, np.sqrt(np.maximum(var, 0.0))
