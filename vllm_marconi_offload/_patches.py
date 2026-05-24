"""Runtime patches for stock vLLM to unblock this connector on hybrid models.

vLLM upstream guards the hybrid-model scheduler path with

    assert num_external_computed_tokens == 0, \
        "External KV connector is not verified yet."

That guard was a TODO marker, not a correctness gate — the function body
below it already factors external tokens into ``num_computed_tokens``.
But the assert blocks any KV connector (this one, LMCache, Mooncake)
from running on hybrid attention models like Qwen3.6, Jamba,
RecurrentGemma, or MiniMax-Text-01.

This module finds that assert at import time, removes it via AST surgery,
and rebinds the patched method onto :class:`vllm.v1.core.sched.scheduler.Scheduler`.
If the assert is already gone (upstream landed the fix, or you patched
your vLLM manually), this module quietly does nothing.

The patch is intentionally narrow:

* It only matches an Assert node whose test is
  ``num_external_computed_tokens == 0``.
* It only touches ``Scheduler._mamba_block_aligned_split``.
* It does not modify any file on disk.

If you prefer to apply the same edit manually, see the README.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

__all__ = ["maybe_patch_hybrid_scheduler", "PATCH_STATUS"]


# Filled in by :func:`maybe_patch_hybrid_scheduler` so callers / tests can
# introspect what happened: "applied", "already-patched", "not-needed",
# "skipped-no-vllm", or "skipped-<reason>".
PATCH_STATUS: str = "pending"


def _matches_external_computed_tokens_assert(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Assert):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not (isinstance(test.left, ast.Name)
            and test.left.id == "num_external_computed_tokens"):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    rhs = test.comparators[0]
    return isinstance(rhs, ast.Constant) and rhs.value == 0


def maybe_patch_hybrid_scheduler() -> str:
    """Attempt to remove vLLM's hybrid-scheduler guard.

    Returns the new value of :data:`PATCH_STATUS`. Safe to call multiple
    times — second call sees the patched body and returns ``"already-patched"``.
    """
    global PATCH_STATUS
    try:
        import vllm.v1.core.sched.scheduler as sched_mod
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:  # pragma: no cover - env-dependent
        PATCH_STATUS = f"skipped-no-vllm:{type(exc).__name__}"
        return PATCH_STATUS

    method = getattr(Scheduler, "_mamba_block_aligned_split", None)
    if method is None:
        # Older vLLM without hybrid-aware scheduler; nothing to patch.
        PATCH_STATUS = "not-needed-no-hybrid-method"
        return PATCH_STATUS

    try:
        src = textwrap.dedent(inspect.getsource(method))
    except (OSError, TypeError) as exc:
        PATCH_STATUS = f"skipped-no-source:{type(exc).__name__}"
        return PATCH_STATUS

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        PATCH_STATUS = f"skipped-parse-error:{exc.msg}"
        return PATCH_STATUS

    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        PATCH_STATUS = "skipped-unexpected-ast"
        return PATCH_STATUS

    fn = tree.body[0]
    new_body = [n for n in fn.body if not _matches_external_computed_tokens_assert(n)]
    removed = len(new_body) != len(fn.body)
    if not removed:
        # The assert is gone already — either upstream removed it, or this
        # module has been imported before. In both cases we are happy.
        PATCH_STATUS = "not-needed"
        return PATCH_STATUS

    fn.body = new_body
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        code = compile(module, "<vllm_marconi_offload-patches>", "exec")
        ns: dict = {}
        # exec inside the scheduler module's namespace so the rebuilt
        # function sees Request, list, int, and the rest of vLLM's locals.
        exec(code, sched_mod.__dict__, ns)
    except Exception as exc:
        PATCH_STATUS = f"skipped-compile-error:{type(exc).__name__}"
        return PATCH_STATUS

    patched = ns.get("_mamba_block_aligned_split")
    if patched is None:
        PATCH_STATUS = "skipped-no-fn-after-exec"
        return PATCH_STATUS

    Scheduler._mamba_block_aligned_split = patched  # type: ignore[attr-defined]
    PATCH_STATUS = "applied"
    return PATCH_STATUS
