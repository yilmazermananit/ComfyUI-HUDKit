"""Determinism + caching primitives for ComfyUI-HUDKit.

This module is schema-agnostic: it imports only the Python standard library and
``yea_hud.types``. It contains **no** ComfyUI / torch / PIL imports and does no
I/O, networking, subprocess, ``eval``/``exec`` or any non-deterministic work. Every
public symbol here is pure, deterministic and safe to run inside Comfy Cloud.

The whole HUD look is *seed-stable*: a given ``(seed, track_id)`` must always pick
the same colour / primitive / asset, and a given ``(seed, track_id, frame_index)``
must always pick the same per-frame micro-variation, on every machine and every
run. Python's builtin :func:`hash` is randomised per-process (``PYTHONHASHSEED``)
and therefore **must not** be used for any of this. We hash with
:func:`hashlib.blake2b` over a canonical, type-tagged, ``'|'``-joined byte string so
that the result depends only on the *values* supplied, never on object identity,
dict ordering, or interpreter start-up state.

Coordinate / colour conventions are documented in ``docs/INTERFACES.md``; this
module does not deal with geometry, only with hashing, RNG and a tiny LRU cache.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import OrderedDict
from typing import Any, Hashable, Iterable, Optional

# NOTE: we import from .types purely to keep the dependency graph honest (this
# module is allowed to depend on types and nothing else in the package). We do
# not currently need a symbol from it, but importing the module validates that
# the package layout is intact wherever utils is used.
from . import types as _types  # noqa: F401  (intentional: package-graph anchor)


__all__ = [
    "stable_hash",
    "rng_for",
    "unit_float",
    "LRUCache",
]


# ---------------------------------------------------------------------------
# Canonical, deterministic serialisation of hash "parts"
# ---------------------------------------------------------------------------
#
# Every part is encoded as ``<TAG>:<payload>`` and the parts are joined with the
# byte ``|`` (0x7C). The leading tag guarantees that values of different types can
# never collide (e.g. the int ``1`` vs the str ``"1"`` vs the bool ``True``), and
# that the float ``1.0`` and the int ``1`` hash differently. This makes the hash a
# faithful function of *both* the type and the value of each part.

_SEP = b"|"

# Digest size: 8 bytes -> a full 64-bit unsigned integer, matching the contract
# ("64-bit deterministic hash"). blake2b supports any digest_size 1..64.
_DIGEST_BYTES = 8
_MASK64 = (1 << 64) - 1

# 53-bit mantissa scaling (same construction CPython uses for random()):
# draw a 53-bit integer and divide by 2**53 to get a uniform float in [0, 1).
_2_POW_53 = float(1 << 53)


def _canon_float(value: float) -> bytes:
    """Return a canonical, OS-/run-stable byte encoding of a float.

    ``repr(float)`` is the shortest string that round-trips and is identical on
    every CPython 3 build, which makes it deterministic. We additionally fold a
    few degenerate cases so that values which are *equal as numbers* hash the
    same and pathological values stay well-defined:

    * ``-0.0`` is normalised to ``0.0`` (they are numerically equal).
    * ``NaN`` collapses to the single token ``nan`` (all NaNs are treated alike;
      Python has many bit-patterns for NaN and ``repr`` of them is unstable).
    * ``+inf`` / ``-inf`` use their stable ``repr`` tokens ``inf`` / ``-inf``.
    """
    if value != value:  # NaN (NaN is the only value not equal to itself)
        return b"nan"
    if value == 0.0:     # collapses both +0.0 and -0.0
        return b"0.0"
    if math.isinf(value):
        return b"inf" if value > 0.0 else b"-inf"
    return repr(value).encode("ascii")


def _encode_part(part: Any) -> bytes:
    """Encode one hash part as ``<TAG>:<payload>`` bytes, deterministically.

    Supported, explicitly-typed parts (stable across runs / OS / process):

    * ``None``           -> ``N:``
    * ``bool``           -> ``B:0`` / ``B:1``   (tagged *before* int on purpose)
    * ``int``            -> ``I:<decimal>``      (arbitrary precision, sign kept)
    * ``float``          -> ``F:<canonical>``    (see :func:`_canon_float`)
    * ``str``            -> ``S:<utf-8 bytes>``
    * ``bytes``/``bytearray`` -> ``Y:<raw bytes>``

    Anything else falls back to ``S:`` over ``repr(part)``. This keeps the
    function total (it never raises on an exotic part) while still being a pure,
    deterministic function of the value. Builtins like ``hash()`` are never
    consulted, so the result is independent of ``PYTHONHASHSEED``.
    """
    # bool is a subclass of int -> it MUST be checked first, otherwise True/False
    # would be encoded as 1/0 and collide with the integers 1/0.
    if part is None:
        return b"N:"
    if isinstance(part, bool):
        return b"B:1" if part else b"B:0"
    if isinstance(part, int):
        return b"I:" + repr(part).encode("ascii")
    if isinstance(part, float):
        return b"F:" + _canon_float(part)
    if isinstance(part, str):
        return b"S:" + part.encode("utf-8")
    if isinstance(part, (bytes, bytearray)):
        return b"Y:" + bytes(part)
    # Defensive, total fallback for unusual inputs (e.g. a tuple key): use the
    # value's repr. Deterministic for the value types we actually pass around.
    return b"S:" + repr(part).encode("utf-8")


def _digest(*parts: Any) -> bytes:
    """Return the raw 8-byte blake2b digest of the canonicalised parts.

    The parts are individually encoded (type-tagged) and joined with ``|`` before
    hashing, so neither the count, the order, nor the types of the parts can be
    confused. With zero parts this hashes the empty string, which is itself a
    well-defined, stable value.
    """
    encoded = _SEP.join(_encode_part(p) for p in parts)
    return hashlib.blake2b(encoded, digest_size=_DIGEST_BYTES).digest()


def stable_hash(*parts: Any) -> int:
    """Deterministic 64-bit unsigned hash of ``parts``.

    Independent of ``PYTHONHASHSEED`` and stable across runs, processes and
    operating systems (it uses :func:`hashlib.blake2b`, **not** the builtin
    :func:`hash`). Each part is type-tagged before hashing, so ints, floats,
    strings, bools, bytes and ``None`` never collide with one another:
    ``stable_hash(1) != stable_hash("1") != stable_hash(1.0) != stable_hash(True)``.

    Args:
        *parts: Any mix of ``int``/``float``/``str``/``bool``/``bytes``/``None``
            (other types fall back to their ``repr``). Order is significant.

    Returns:
        int: A value in ``[0, 2**64)``.
    """
    return int.from_bytes(_digest(*parts), "big") & _MASK64


def rng_for(*parts: Any) -> random.Random:
    """Return a :class:`random.Random` seeded by :func:`stable_hash` of ``parts``.

    The returned generator is fully deterministic for the given parts and carries
    no shared/global state, so two calls with the same parts yield identical
    sequences and calls with different parts are independent. This is the seam
    used for *track-level identity* and *frame micro-variation*::

        rng_for(seed, track_id)                 # stable per-track choices
        rng_for(seed, track_id, frame_index)    # stable per-frame jitter

    Args:
        *parts: Hashing parts, forwarded verbatim to :func:`stable_hash`.

    Returns:
        random.Random: A freshly-seeded independent generator.
    """
    return random.Random(stable_hash(*parts))


def unit_float(*parts: Any) -> float:
    """Deterministic uniform float in ``[0.0, 1.0)`` from ``parts``.

    This is a single *direct* draw derived straight from the blake2b digest (it
    does not spin up a :class:`random.Random`), intended for probability gates
    such as ``unit_float(seed, track_id) < display_probability``. It is
    distribution-equivalent to ``rng_for(*parts).random()`` — uniform on
    ``[0, 1)`` with 53 bits of resolution — but is a single cheap draw with no
    generator allocation.

    The construction mirrors CPython's own ``random()``: take the top 53 bits of
    the 64-bit digest and divide by ``2**53``. The result is always ``>= 0.0`` and
    strictly ``< 1.0``.

    Args:
        *parts: Hashing parts, forwarded verbatim to :func:`stable_hash`.

    Returns:
        float: A value ``x`` with ``0.0 <= x < 1.0``.
    """
    bits64 = int.from_bytes(_digest(*parts), "big")
    # Take the high 53 bits to fill the IEEE-754 double mantissa cleanly.
    mantissa53 = bits64 >> (64 - 53)
    return mantissa53 / _2_POW_53


# ---------------------------------------------------------------------------
# LRUCache — small, insertion/usage-ordered cache with eviction
# ---------------------------------------------------------------------------


class LRUCache:
    """A tiny least-recently-used cache with a fixed maximum item count.

    Backed by :class:`collections.OrderedDict`; the front of the dict is the
    *least* recently used entry and the back is the *most* recently used. Both
    :meth:`get` (on a hit) and :meth:`put` mark a key as most-recently-used, so
    the entry evicted when the cache is full is always the genuinely
    least-recently-touched one. Used to cache rendered/recoloured asset layers,
    fonts, etc. (see ``text_render`` / ``assets``).

    The cache is deliberately not thread-safe and stores values by reference; it
    performs no copying. ``None`` is a legal value, but because :meth:`get`
    returns ``None`` for a miss, callers that need to store ``None`` should treat
    a ``None`` result as "absent" (the HUD code paths never cache ``None``).
    """

    def __init__(self, max_items: int = 512) -> None:
        """Create an LRU cache.

        Args:
            max_items: Maximum number of entries to retain. Values ``<= 0`` make
                the cache a no-op (it stores nothing and always misses), which is
                a safe way to disable caching without changing call sites.
        """
        self._max_items = int(max_items)
        self._store: "OrderedDict[Hashable, Any]" = OrderedDict()

    @property
    def max_items(self) -> int:
        """The configured maximum number of cached items."""
        return self._max_items

    def get(self, key: Hashable) -> Optional[Any]:
        """Return the value for ``key`` and mark it most-recently-used.

        Args:
            key: Any hashable cache key.

        Returns:
            The stored value, or ``None`` if the key is not present.
        """
        if key not in self._store:
            return None
        # Move to the MRU end so it is the last to be evicted.
        self._store.move_to_end(key, last=True)
        return self._store[key]

    def put(self, key: Hashable, value: Any) -> None:
        """Insert/update ``key`` -> ``value`` and evict beyond ``max_items``.

        Re-inserting an existing key refreshes both its value and its recency.
        After insertion, the oldest entries are dropped until the cache holds at
        most ``max_items`` items. If ``max_items <= 0`` the cache stays empty.

        Args:
            key: Any hashable cache key.
            value: The value to associate with ``key`` (stored by reference).
        """
        if self._max_items <= 0:
            # Caching disabled: ensure any stale entry for this key is gone.
            self._store.pop(key, None)
            return
        if key in self._store:
            self._store.move_to_end(key, last=True)
        self._store[key] = value
        # Evict from the LRU (front) end until within capacity.
        while len(self._store) > self._max_items:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    def __len__(self) -> int:
        """Return the number of entries currently held."""
        return len(self._store)

    def __contains__(self, key: Hashable) -> bool:
        """Return ``True`` if ``key`` is cached.

        This is a pure membership test and does **not** affect recency ordering
        (unlike :meth:`get`).
        """
        return key in self._store

    def keys(self) -> Iterable[Hashable]:
        """Return a view of the cached keys, ordered least- to most-recent."""
        return self._store.keys()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"LRUCache(max_items={self._max_items}, size={len(self._store)})"
