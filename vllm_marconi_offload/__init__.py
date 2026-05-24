"""vLLM Marconi-eviction L1+L2 KV cache offload connector.

Importing this module:

1. Tries to apply the hybrid-scheduler runtime patch (see ``_patches.py``).
   If your vLLM already lacks the guard — or if your model is not hybrid —
   the patch is a no-op.
2. Registers ``SimpleCPUOffloadConnector`` with vLLM's
   ``KVConnectorFactory`` under that same name, so this string is usable
   directly in ``--kv-transfer-config``::

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

The connector implements a two-tier KV offload (``L1`` = pinned-or-mmap
CPU pool; ``L2`` = optional mmap pool on a slower / larger medium like
an XFS NVMe partition) with reuse-aware Marconi-style eviction. See the
README for design notes and tuning advice.
"""

from __future__ import annotations

import logging

from vllm_marconi_offload._patches import maybe_patch_hybrid_scheduler

__version__ = "0.1.0"
__all__ = [
    "SimpleCPUOffloadConnector",
    "PATCH_STATUS",
    "__version__",
]

_logger = logging.getLogger("vllm_marconi_offload")


# --- Stage 1: apply the hybrid-scheduler patch (no-op if not needed) ---

PATCH_STATUS = maybe_patch_hybrid_scheduler()
if PATCH_STATUS == "applied":
    _logger.info(
        "vllm_marconi_offload: removed `assert num_external_computed_tokens "
        "== 0` from vllm.v1.core.sched.scheduler.Scheduler._mamba_block_aligned_split "
        "to unblock hybrid-model + external-KV-connector usage."
    )
elif PATCH_STATUS.startswith("skipped"):
    _logger.warning(
        "vllm_marconi_offload: did not patch the hybrid scheduler guard "
        "(status=%s). Hybrid models (Qwen3.6, Jamba, RecurrentGemma, ...) "
        "may hit an AssertionError. See the README for the manual patch.",
        PATCH_STATUS,
    )

# --- Stage 2: import the connector and register it with the factory ---

# Re-exported for direct `from vllm_marconi_offload import SimpleCPUOffloadConnector`.
from vllm_marconi_offload.connector import SimpleCPUOffloadConnector  # noqa: E402


def _register() -> None:
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        _logger.warning(
            "vllm_marconi_offload: vLLM not importable, skipping connector "
            "registration (%s).",
            exc,
        )
        return

    # Idempotent: if the connector is already registered (e.g. duplicate
    # import), KVConnectorFactory will raise; swallow and move on.
    try:
        KVConnectorFactory.register_connector(
            "SimpleCPUOffloadConnector",
            "vllm_marconi_offload.connector",
            "SimpleCPUOffloadConnector",
        )
        _logger.info(
            "vllm_marconi_offload: registered SimpleCPUOffloadConnector with "
            "KVConnectorFactory."
        )
    except Exception as exc:
        # Most likely: already registered. Don't spam at warning level.
        _logger.debug(
            "vllm_marconi_offload: register_connector skipped (%s).", exc
        )


_register()
