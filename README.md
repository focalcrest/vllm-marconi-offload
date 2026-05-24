# vllm-marconi-offload

Two-tier (RAM + SSD) CPU KV cache offload for **stock vLLM**, with
reuse-aware **Marconi-style eviction**, packaged as a single
`pip install` that registers itself with vLLM at import time.

* **L1**: pinned-or-mmap CPU pool (typical: a few GB of `/dev/shm`) —
  microsecond-latency tier that absorbs HBM eviction overflow.
* **L2**: optional mmap pool on a slower / larger medium (typical:
  an XFS-on-NVMe partition with hundreds of GB) — millisecond-latency
  cold tier that survives across server restarts.
* **Marconi eviction** ([NSDI '25][marconi-paper]): scores cached
  blocks by `reuse_count × exp(-decay × age) / (depth + 1)` so hot
  shared prefixes (system prompts, RAG contexts, multi-turn agent
  histories) outlive the LRU cold tail under burst load.

If your workload reuses prefixes — same system prompt across many
agents, multi-turn conversations that idle and resume, RAG with hot
contexts — this connector keeps that state addressable across HBM
eviction events instead of forcing re-prefill.

[marconi-paper]: https://www.usenix.org/conference/nsdi25/presentation/zhang-yujie

---

## Install

```bash
pip install vllm                  # any supported version (see matrix)
pip install vllm-marconi-offload
```

That's it. No vLLM source edits. Import-time hooks register the
connector with `KVConnectorFactory` and apply a small in-memory
patch to vLLM's scheduler (see [Hybrid models](#hybrid-models)).

---

## Quickstart

```bash
vllm serve <model> \
    --enable-prefix-caching \
    --kv-transfer-config '{
      "kv_connector": "SimpleCPUOffloadConnector",
      "kv_connector_module_path": "vllm_marconi_offload",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "cpu_pool_path": "/dev/shm/vllm_kv_l1",
        "cpu_bytes_to_use": 6442450944,
        "l2_pool_path": "/kvcache/vllm_kv_l2",
        "l2_bytes_to_use": 322122547200,
        "eviction_policy": "marconi",
        "eviction_decay": 0.005,
        "eviction_reuse_mode": "linear",
        "admission_threshold": 1
      }
    }'
```

* `cpu_pool_path` — directory holding the L1 mmap files. On tmpfs
  (`/dev/shm`) for low-latency RAM-backed storage, or any regular
  filesystem if you want pageable memory.
* `cpu_bytes_to_use` — total L1 budget across all engines. The
  connector divides by `data_parallel_size` and the underlying
  manager divides by `tensor_parallel_size` to compute per-rank
  capacity.
* `l2_pool_path` / `l2_bytes_to_use` — same shape, for the optional
  L2 tier. Omit both fields to run single-tier.
* `eviction_policy` — `"marconi"` (default) or `"lru"`. Marconi
  scores `reuse_count × exp(-decay × age) / (depth + 1)`.
* `eviction_decay` — exponential decay over scheduler ticks
  (default `0.0` — pure reuse-and-depth ranking).
* `eviction_reuse_mode` — `"linear"` (concentrates cache on the
  single hottest prefix), `"log"` (recommended for Zipfian /
  long-tail workloads), or `"none"` (depth-only).
* `admission_threshold` — number of times a hash must be seen
  before being admitted to L1 (default `1` matches vLLM's "admit
  immediately" behaviour; raise to filter one-shot prompts out of
  the cache).

Observability: grep your server log for `[CPU-OFFLOAD HIT]` (per-N
requests hit-rate breakdown) and `Marconi eviction: stats=`
(tracked / l1 / l2 / demotes / promotes / hoisted).

---

## Hybrid models

vLLM's hybrid-attention scheduler (Qwen3.6, Jamba, RecurrentGemma,
MiniMax-Text-01, ...) currently guards its prefill path with:

```python
assert num_external_computed_tokens == 0, \
    "External KV connector is not verified yet."
```

That assert was a TODO marker, not a correctness gate — the
function body below it already factors external tokens into
`num_computed_tokens` correctly. But it stops any external KV
connector (this one, LMCache, Mooncake, NIXL) from running on
hybrid models.

On import, `vllm_marconi_offload` removes that assert via AST
surgery on the live `Scheduler._mamba_block_aligned_split` method.
The patch is:

* **Narrow** — only matches `Assert` nodes whose test is
  `num_external_computed_tokens == 0`.
* **In-memory** — no file on disk is modified.
* **Idempotent** — re-importing does nothing.
* **Self-disabling** — if upstream vLLM has already removed the
  assert (e.g., a future release), or you patched manually, the
  module sees no matching node and is a no-op.

You can inspect what happened via:

```python
import vllm_marconi_offload
print(vllm_marconi_offload.PATCH_STATUS)
# -> "applied" / "not-needed" / "skipped-<reason>"
```

If you would rather **apply the patch manually**, the equivalent
edit to `vllm/v1/core/sched/scheduler.py` (line numbers may vary)
is to delete the lines:

```python
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet.")
```

inside `Scheduler._mamba_block_aligned_split`. We are working on
upstreaming this one-line removal so the patch becomes unnecessary
on future vLLM releases.

If your model is a **pure-attention transformer** (Llama, Mistral,
plain Qwen2/3, ...), the guard is never reached and the patch has
no observable effect.

---

## Compatibility matrix

| `vllm-marconi-offload` | Tested vLLM versions | Notes |
|:-----------------------|:---------------------|:------|
| `0.1.x`                | `0.7.x` `0.8.x`      | Initial release. Hybrid-model patch enabled. |

When vLLM ships an incompatible refactor of `Scheduler._mamba_block_aligned_split`,
we publish a new release with an updated AST matcher and bump the matrix.
The patcher fails closed (no-op + warning), never produces wrong output.

---

## Design notes

* **No CUDA, no kernels.** The connector is pure Python plus
  `mmap()` and `cudaMemcpy`. Works on any GPU vLLM supports.
* **Two tiers, demote-on-store, promote-on-load.** When HBM evicts
  a block, the manager demotes it to L1 instead of dropping it;
  when L1 fills, L1 victims demote to L2. A future request whose
  prefix matches a demoted block walks L1 → L2, promotes the
  matched blocks back through the tiers, and the prefill skips
  recomputation entirely.
* **Hybrid models supported** — implements `SupportsHMA` mixin
  from vLLM's connector base.
* **Per-engine isolation** — engine_id is suffixed into mmap file
  names, so DP > 1 deployments don't collide on shared pool paths.
* **Marconi admission tracker is decoupled from eviction policy.**
  You can enable Marconi admission filtering (skip caching one-shot
  prompts) with `admission_threshold > 1` even if you keep
  `eviction_policy = "lru"`.

---

## Caveats and limits

* This is a **single-host** connector. KV state in L1/L2 does not
  migrate across machines. For multi-host KV pooling, look at
  LMCache or Mooncake.
* L2 promote-on-load is **synchronous** — a request that has to
  pull blocks from L2 stalls until the read completes. With XFS on
  NVMe at ~3-7 GB/s, a 4K-token block (~16-64 KB depending on
  arch) loads in a few microseconds; long contexts of tens of
  thousands of tokens are still measured in tens of milliseconds.
* vLLM's DP router is **not prefix-aware** as of 0.7.x — two
  replicas hold independent connector pools. Use a sticky-prefix
  proxy in front of `--data-parallel-size > 1` if you want
  cross-replica prefix sharing.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
