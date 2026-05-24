"""Unit tests for the hybrid-scheduler runtime patcher.

These tests do not require vLLM to be importable; they exercise the AST
predicate against synthetic functions so the matcher contract stays
locked down regardless of what upstream vLLM looks like today.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from vllm_marconi_offload._patches import (
    _matches_external_computed_tokens_assert,
)


def _stmts(src: str) -> list[ast.stmt]:
    return ast.parse(textwrap.dedent(src)).body


# -------------------- matcher contract --------------------


def test_matches_canonical_assert():
    [node] = _stmts(
        '''
        assert num_external_computed_tokens == 0, \
            "External KV connector is not verified yet."
        '''
    )
    assert _matches_external_computed_tokens_assert(node)


def test_matches_assert_without_message():
    [node] = _stmts("assert num_external_computed_tokens == 0\n")
    assert _matches_external_computed_tokens_assert(node)


def test_matches_assert_with_parenthesized_message():
    [node] = _stmts(
        '''
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet."
        )
        '''
    )
    assert _matches_external_computed_tokens_assert(node)


def test_does_not_match_unrelated_assert():
    [node] = _stmts("assert num_new_tokens > 0\n")
    assert not _matches_external_computed_tokens_assert(node)


def test_does_not_match_inverted_comparison():
    # `0 == num_external_computed_tokens` — different left/right; we
    # match on `Name == 0`, not the reverse, on purpose.
    [node] = _stmts("assert 0 == num_external_computed_tokens\n")
    assert not _matches_external_computed_tokens_assert(node)


def test_does_not_match_different_constant():
    [node] = _stmts("assert num_external_computed_tokens == 1\n")
    assert not _matches_external_computed_tokens_assert(node)


def test_does_not_match_non_assert_statement():
    [node] = _stmts("num_external_computed_tokens = 0\n")
    assert not _matches_external_computed_tokens_assert(node)


# -------------------- end-to-end: rewriting a function --------------------


def test_strip_assert_keeps_other_statements():
    """Rebuild a function with the assert removed, leave other code intact."""
    original_src = textwrap.dedent(
        '''
        def m(self, n_ext):
            num_external_computed_tokens = n_ext
            assert num_external_computed_tokens == 0, (
                "External KV connector is not verified yet.")
            return num_external_computed_tokens + 1
        '''
    ).strip("\n") + "\n"

    # Sanity: original raises on non-zero input.
    ns_before: dict = {}
    exec(original_src, ns_before)
    with pytest.raises(AssertionError):
        ns_before["m"](None, 5)

    # Apply the patcher logic against the source string directly.
    tree = ast.parse(original_src)
    fn = tree.body[0]
    new_body = [
        n for n in fn.body
        if not _matches_external_computed_tokens_assert(n)
    ]
    assert len(new_body) == len(fn.body) - 1, (
        f"expected to drop exactly one statement, "
        f"got {len(fn.body)} -> {len(new_body)}"
    )
    fn.body = new_body
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns_after: dict = {}
    exec(compile(module, "<test>", "exec"), {}, ns_after)

    # After patching, the non-zero input no longer raises and the return
    # value is the post-assert expression.
    assert ns_after["m"](None, 5) == 6
