"""SHA-256-only random primitives for the research protocol.

The functions are counter based: a draw is a pure function of the episode seed,
namespace, and draw index. This makes common-random-number pairing explicit and
avoids Python, NumPy, and global RNG state in the demand process.
"""

from __future__ import annotations

import hashlib
import math


def digest(seed_hex: str, namespace: str, index: int = 0) -> bytes:
    if index < 0:
        raise ValueError("draw index must be non-negative")
    return hashlib.sha256(f"live-y-domain-randomized-grpo-v1|{seed_hex}|{namespace}|{index}".encode()).digest()


def uniform(seed_hex: str, namespace: str, index: int = 0) -> float:
    """Return a deterministic U[0,1) value from 53 digest bits."""

    return (int.from_bytes(digest(seed_hex, namespace, index)[:8], "big") >> 11) / 2**53


def poisson(seed_hex: str, namespace: str, lam: float, index: int = 0) -> int:
    """Sample Poisson(lam) using Knuth's algorithm and digest-derived uniforms."""

    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be finite and non-negative")
    if lam == 0:
        return 0
    threshold = math.exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        product *= uniform(seed_hex, f"{namespace}/poisson", count + index * 4096)
        count += 1
    return count - 1
