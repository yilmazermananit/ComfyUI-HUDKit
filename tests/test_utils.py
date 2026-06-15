"""Unit tests for ``yea_hud.utils`` (determinism + caching).

Runnable two ways:

* as pytest::

      /Users/differenterman/Documents/ComfyUI/.venv/bin/python -m pytest tests/test_utils.py -q

* as a plain script (no pytest required)::

      /Users/differenterman/Documents/ComfyUI/.venv/bin/python tests/test_utils.py

The script runner executes every ``test_*`` function, prints a summary and exits
non-zero on any failure. No ComfyUI, no network, no model files, no file writes.
The hardest guarantee under test is *seed-stable* hashing that is independent of
``PYTHONHASHSEED`` (brief §25 determinism cases), so a subset of those checks is
re-run in a fresh subprocess with a different hash seed.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Make the package importable when run as a bare script from anywhere.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import hashlib  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402

from yea_hud.utils import LRUCache, rng_for, stable_hash, unit_float  # noqa: E402


# ---------------------------------------------------------------------------
# stable_hash — determinism, range, type-separation
# ---------------------------------------------------------------------------


def test_stable_hash_is_int_in_64bit_range():
    for parts in [(), (0,), (1, 2, 3), ("person", 77, 3), (-5, 2.5, "x", True, None)]:
        h = stable_hash(*parts)
        assert isinstance(h, int)
        assert 0 <= h < (1 << 64), f"out of 64-bit range for {parts}: {h}"


def test_stable_hash_repeatable_same_process():
    a = stable_hash("person", 77, 3)
    b = stable_hash("person", 77, 3)
    assert a == b


def test_stable_hash_order_sensitive():
    assert stable_hash(1, 2) != stable_hash(2, 1)


def test_stable_hash_arity_sensitive():
    # An extra part must change the hash (no accidental flattening/concat collisions).
    assert stable_hash(1, 2) != stable_hash(1, 2, 3)
    assert stable_hash("ab") != stable_hash("a", "b")  # '|' separator prevents merge


def test_stable_hash_type_tags_prevent_collisions():
    # int 1 vs str "1" vs float 1.0 vs bool True must all differ.
    vals = {
        stable_hash(1),
        stable_hash("1"),
        stable_hash(1.0),
        stable_hash(True),
    }
    assert len(vals) == 4, f"type-tag collision: {vals}"
    # int 0 vs bool False vs float 0.0 vs None must differ.
    vals0 = {
        stable_hash(0),
        stable_hash(False),
        stable_hash(0.0),
        stable_hash(None),
    }
    assert len(vals0) == 4, f"zero-family collision: {vals0}"


def test_stable_hash_not_builtin_hash():
    # The contract forbids the randomised builtin hash. The builtin hash of a str
    # depends on PYTHONHASHSEED and (almost surely) differs from our stable value.
    assert stable_hash("person") != hash("person") or True  # not load-bearing alone
    # The load-bearing check is the cross-process test below; here we just assert
    # our value matches a hand-computed blake2b over the canonical encoding.
    expected = int.from_bytes(
        hashlib.blake2b(b"S:person", digest_size=8).digest(), "big"
    )
    assert stable_hash("person") == expected


def test_stable_hash_canonical_encoding_matches_blake2b():
    # Re-derive a multi-part hash by hand to lock the canonical wire format.
    encoded = b"|".join([b"S:p", b"I:77", b"I:3"])
    expected = int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "big")
    assert stable_hash("p", 77, 3) == expected


def test_stable_hash_float_normalises_signed_zero():
    # -0.0 and 0.0 are numerically equal and must hash the same.
    assert stable_hash(0.0) == stable_hash(-0.0)


def test_stable_hash_float_nan_is_stable():
    # All NaNs collapse to one stable token, so two independent NaNs agree.
    nan1 = float("nan")
    nan2 = math.nan
    assert stable_hash(nan1) == stable_hash(nan2)


def test_stable_hash_float_inf_distinct():
    assert stable_hash(float("inf")) != stable_hash(float("-inf"))
    assert stable_hash(float("inf")) != stable_hash(0.0)


def test_stable_hash_empty_is_defined_and_stable():
    assert stable_hash() == stable_hash()
    assert isinstance(stable_hash(), int)


def test_stable_hash_accepts_negative_and_big_ints():
    # Arbitrary-precision ints must not raise and must be distinct.
    big = 10 ** 40
    assert stable_hash(big) != stable_hash(big + 1)
    assert stable_hash(-1) != stable_hash(1)


def test_stable_hash_string_track_id():
    # source_track_id can be int OR str (see types.Detection); both must work.
    assert isinstance(stable_hash(77, "track_a"), int)
    assert stable_hash(77, "track_a") != stable_hash(77, "track_b")


# ---------------------------------------------------------------------------
# rng_for — deterministic generators
# ---------------------------------------------------------------------------


def test_rng_for_returns_random_Random():
    assert isinstance(rng_for(1, 2), random.Random)


def test_rng_for_same_parts_same_sequence():
    r1 = rng_for(77, 5)
    r2 = rng_for(77, 5)
    seq1 = [r1.random() for _ in range(8)]
    seq2 = [r2.random() for _ in range(8)]
    assert seq1 == seq2


def test_rng_for_different_parts_differ():
    s1 = [rng_for(77, 5).random() for _ in range(4)]
    s2 = [rng_for(77, 6).random() for _ in range(4)]
    assert s1 != s2


def test_rng_for_seeded_by_stable_hash():
    # The generator must be seeded by exactly stable_hash(*parts).
    parts = ("person", 77, 3)
    expected = random.Random(stable_hash(*parts)).random()
    assert rng_for(*parts).random() == expected


def test_rng_for_independent_state():
    # Drawing from one generator must not disturb a sibling created identically.
    r1 = rng_for(1, 2)
    _ = [r1.random() for _ in range(100)]
    r2 = rng_for(1, 2)
    # r2 is fresh -> its first draw equals the first draw of a brand new gen.
    assert r2.random() == random.Random(stable_hash(1, 2)).random()


def test_rng_for_frame_micro_variation_distinct_per_frame():
    seed, tid = 77, 3
    draws = [rng_for(seed, tid, f).random() for f in range(5)]
    assert len(set(draws)) == len(draws), "per-frame draws should differ"


# ---------------------------------------------------------------------------
# unit_float — direct uniform draw in [0,1)
# ---------------------------------------------------------------------------


def test_unit_float_in_unit_interval():
    for parts in [(0,), (77, 3), ("a", "b", "c"), (-1, 2.5), ()]:
        u = unit_float(*parts)
        assert isinstance(u, float)
        assert 0.0 <= u < 1.0, f"{parts} -> {u} not in [0,1)"


def test_unit_float_deterministic():
    assert unit_float(77, 3) == unit_float(77, 3)


def test_unit_float_varies_with_parts():
    assert unit_float(77, 3) != unit_float(77, 4)


def test_unit_float_matches_documented_construction():
    # Top 53 bits of the 64-bit digest / 2**53.
    parts = ("person", 77, 3)
    encoded = b"|".join([b"S:person", b"I:77", b"I:3"])
    bits = int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "big")
    expected = (bits >> (64 - 53)) / float(1 << 53)
    assert unit_float(*parts) == expected


def test_unit_float_distribution_roughly_uniform():
    # Smoke check that a probability gate behaves sanely: ~display_probability
    # fraction of tracks pass. Not a statistical test, just a sanity band.
    n = 4000
    p = 0.85
    passes = sum(1 for tid in range(n) if unit_float(77, tid) < p)
    frac = passes / n
    assert 0.80 <= frac <= 0.90, f"gate fraction {frac} far from {p}"


def test_unit_float_consistent_with_stable_hash():
    # unit_float must derive from the SAME digest as stable_hash (same parts ->
    # the high bits of stable_hash reconstruct unit_float).
    parts = (12, "z", 4.5)
    h = stable_hash(*parts)
    expected = (h >> (64 - 53)) / float(1 << 53)
    assert unit_float(*parts) == expected


# ---------------------------------------------------------------------------
# LRUCache — basic behaviour + eviction + recency
# ---------------------------------------------------------------------------


def test_lru_put_get_roundtrip():
    c = LRUCache(max_items=4)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    assert c.get("b") == 2
    assert len(c) == 2


def test_lru_miss_returns_none():
    c = LRUCache(max_items=4)
    assert c.get("nope") is None


def test_lru_overwrite_updates_value():
    c = LRUCache(max_items=4)
    c.put("a", 1)
    c.put("a", 99)
    assert c.get("a") == 99
    assert len(c) == 1


def test_lru_evicts_oldest_beyond_capacity():
    c = LRUCache(max_items=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.put("d", 4)  # should evict "a" (least recently used)
    assert len(c) == 3
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3
    assert c.get("d") == 4


def test_lru_get_refreshes_recency():
    c = LRUCache(max_items=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    # Touch "a" so it is now most-recently-used; "b" becomes the LRU.
    assert c.get("a") == 1
    c.put("d", 4)  # evicts "b"
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3
    assert c.get("d") == 4


def test_lru_put_refreshes_recency():
    c = LRUCache(max_items=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.put("a", 11)  # re-insert refreshes recency -> "b" is now LRU
    c.put("d", 4)   # evicts "b"
    assert c.get("b") is None
    assert c.get("a") == 11
    assert c.get("c") == 3
    assert c.get("d") == 4


def test_lru_clear():
    c = LRUCache(max_items=3)
    c.put("a", 1)
    c.put("b", 2)
    c.clear()
    assert len(c) == 0
    assert c.get("a") is None


def test_lru_len_and_contains():
    c = LRUCache(max_items=3)
    assert len(c) == 0
    c.put("a", 1)
    assert len(c) == 1
    assert ("a" in c) is True
    assert ("z" in c) is False
    # __contains__ must NOT mutate recency (unlike get): "a" stays the LRU here.
    c.put("b", 2)
    c.put("c", 3)
    _ = "a" in c  # membership only, no refresh
    c.put("d", 4)  # evicts the true LRU, which is still "a"
    assert c.get("a") is None


def test_lru_default_max_items_is_512():
    c = LRUCache()
    assert c.max_items == 512
    for i in range(600):
        c.put(i, i)
    assert len(c) == 512
    # Oldest 88 (0..87) evicted; newest retained.
    assert c.get(0) is None
    assert c.get(87) is None
    assert c.get(88) == 88
    assert c.get(599) == 599


def test_lru_zero_capacity_is_noop():
    c = LRUCache(max_items=0)
    c.put("a", 1)
    assert len(c) == 0
    assert c.get("a") is None


def test_lru_negative_capacity_is_noop():
    c = LRUCache(max_items=-5)
    c.put("a", 1)
    assert len(c) == 0


def test_lru_tuple_and_int_keys():
    # Keys used by callers include (id, w_bucket, h_bucket, color_id) tuples.
    c = LRUCache(max_items=4)
    c.put((1, 64, 64, 0), "x")
    c.put(42, "y")
    assert c.get((1, 64, 64, 0)) == "x"
    assert c.get(42) == "y"


def test_lru_capacity_one():
    c = LRUCache(max_items=1)
    c.put("a", 1)
    c.put("b", 2)  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert len(c) == 1


# ---------------------------------------------------------------------------
# Cross-process determinism: the single most important property.
# Re-run a hash in a fresh interpreter with a *different* PYTHONHASHSEED and
# assert the value is byte-for-byte identical to the in-process value.
# ---------------------------------------------------------------------------


def test_stable_hash_independent_of_pythonhashseed():
    expected_hash = stable_hash("person", 77, 3)
    expected_uf = unit_float("person", 77, 3)
    snippet = (
        "from yea_hud.utils import stable_hash, unit_float;"
        "print(stable_hash('person',77,3));"
        "print(repr(unit_float('person',77,3)))"
    )
    seen_hashes = set()
    seen_ufs = set()
    for seed in ("0", "1", "987654321", "random"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0, f"subprocess failed (seed={seed}): {out.stderr}"
        lines = out.stdout.strip().splitlines()
        assert len(lines) == 2, f"unexpected output: {out.stdout!r}"
        h = int(lines[0])
        uf = float(lines[1])
        seen_hashes.add(h)
        seen_ufs.add(uf)
        assert h == expected_hash, f"hash drift at PYTHONHASHSEED={seed}"
        assert uf == expected_uf, f"unit_float drift at PYTHONHASHSEED={seed}"
    assert len(seen_hashes) == 1, f"hash not stable across seeds: {seen_hashes}"
    assert len(seen_ufs) == 1, f"unit_float not stable across seeds: {seen_ufs}"


# ---------------------------------------------------------------------------
# Script runner (no pytest required)
# ---------------------------------------------------------------------------


def _run_all() -> int:
    """Execute every ``test_*`` in this module; print a summary; return exit code."""
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = 0
    failed = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - we want to report everything
            failed += 1
            failures.append((name, repr(exc)))
            print(f"FAIL  {name}: {exc!r}")
        else:
            passed += 1
            print(f"ok    {name}")
    print("-" * 60)
    print(f"{passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
