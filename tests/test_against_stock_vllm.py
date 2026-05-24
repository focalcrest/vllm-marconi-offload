"""Integration test: runs the patcher against the actually-installed vLLM.

Skips automatically if vLLM is not importable. This is the test that
proves the AST matcher still finds the right assert in upstream vLLM —
unlike the synthetic tests in test_patches.py, which only exercise the
matcher predicate against hand-written source strings.

Run with::

    pytest tests/test_against_stock_vllm.py -v

The expected outcomes by environment:

* Stock vLLM whose ``Scheduler._mamba_block_aligned_split`` still
  contains the ``assert num_external_computed_tokens == 0`` guard:
  ``PATCH_STATUS == "applied"``.
* vLLM that has already removed the guard (e.g. via this project's
  upstream PR or your own manual edit): ``PATCH_STATUS == "not-needed"``.
* vLLM old enough that the hybrid scheduler method doesn't exist:
  ``PATCH_STATUS == "not-needed-no-hybrid-method"``.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

vllm = pytest.importorskip("vllm")
sched_mod = pytest.importorskip("vllm.v1.core.sched.scheduler")


def _has_external_tokens_assert() -> bool:
    """True if the live Scheduler._mamba_block_aligned_split still asserts."""
    method = getattr(sched_mod.Scheduler, "_mamba_block_aligned_split", None)
    if method is None:
        return False
    try:
        src = inspect.getsource(method)
    except (OSError, TypeError):
        return False
    return "assert num_external_computed_tokens == 0" in src


def test_patch_status_reflects_environment(tmp_path, monkeypatch):
    # Force a fresh import so the patcher runs every time, regardless of
    # whether other tests / earlier sessions already imported this package.
    import sys
    for mod_name in list(sys.modules):
        if mod_name.startswith("vllm_marconi_offload"):
            del sys.modules[mod_name]

    had_assert = _has_external_tokens_assert()

    import vllm_marconi_offload as pkg

    if had_assert:
        assert pkg.PATCH_STATUS == "applied", (
            f"Expected patcher to remove the assert, got {pkg.PATCH_STATUS!r}"
        )
        # Confirm the assert is gone after import.
        assert not _has_external_tokens_assert(), (
            "Source-level check says the assert is still present after "
            "PATCH_STATUS=applied — the AST rewrite did not stick."
        )
    else:
        assert pkg.PATCH_STATUS in {
            "not-needed",
            "not-needed-no-hybrid-method",
        }, f"Unexpected PATCH_STATUS for assert-free env: {pkg.PATCH_STATUS!r}"


def test_factory_registration(monkeypatch):
    """SimpleCPUOffloadConnector ends up in KVConnectorFactory's registry."""
    import sys
    for mod_name in list(sys.modules):
        if mod_name.startswith("vllm_marconi_offload"):
            del sys.modules[mod_name]
    importlib.import_module("vllm_marconi_offload")

    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    assert "SimpleCPUOffloadConnector" in getattr(
        KVConnectorFactory, "_registry", {}
    )


def test_patched_method_bytecode_has_no_assert():
    """After patching, the method's bytecode must contain no AssertionError
    raise and no copy of the upstream assert message. ``inspect.getsource``
    cannot be used here because the patched method is built via ``exec`` and
    has no on-disk source, so we go one level lower.

    Skipped if the live vLLM never had the assert in the first place.
    """
    if not _has_external_tokens_assert():
        pytest.skip(
            "live vLLM has no `assert num_external_computed_tokens == 0` "
            "to remove; nothing to verify here."
        )

    import vllm_marconi_offload  # noqa: F401 — applies the patch

    method = sched_mod.Scheduler._mamba_block_aligned_split
    code = method.__code__

    msg = "External KV connector is not verified yet"
    has_msg = any(
        isinstance(c, str) and msg in c
        for c in code.co_consts
        if c is not None
    )
    assert not has_msg, (
        "Patcher reported success but the assert message string is still "
        "in the method's co_consts — the AST rewrite did not remove the assert."
    )

    import dis
    import io
    buf = io.StringIO()
    dis.dis(code, file=buf)
    asm = buf.getvalue()
    assert "LOAD_ASSERTION_ERROR" not in asm, (
        "Patched method still has a LOAD_ASSERTION_ERROR opcode."
    )
    assert "RAISE_VARARGS" not in asm, (
        "Patched method still has a RAISE_VARARGS opcode."
    )
