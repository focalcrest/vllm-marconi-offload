#!/usr/bin/env bash
# Minimal launch example using vllm-marconi-offload's dual-tier KV cache.
#
# Prereqs:
#   pip install vllm vllm-marconi-offload
#
# Notes:
# * /dev/shm/vllm_kv_l1 = tmpfs (RAM-backed) L1 pool
# * /kvcache/vllm_kv_l2 = a regular filesystem path (XFS-on-NVMe in our
#   reference deployment) for the L2 cold tier
# * cpu_bytes_to_use and l2_bytes_to_use are SERVER-WIDE totals;
#   the connector divides by data_parallel_size and the underlying
#   manager further divides by tensor_parallel_size to compute per-rank
#   capacity.
# * Adjust the byte counts to match your machine's /dev/shm budget and
#   your /kvcache partition size.

set -e

MODEL=${MODEL:-Qwen/Qwen3-32B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
L1_PATH=${L1_PATH:-/dev/shm/vllm_kv_l1}
L2_PATH=${L2_PATH:-/kvcache/vllm_kv_l2}
L1_BYTES=${L1_BYTES:-$((6 * 1024 * 1024 * 1024))}     # 6 GB tmpfs
L2_BYTES=${L2_BYTES:-$((300 * 1024 * 1024 * 1024))}   # 300 GB SSD

# Clean the pools on a fresh launch; remove these lines if you want
# warm-cache survival across restarts.
rm -rf "$L1_PATH" "$L2_PATH"

KV_TRANSFER_JSON=$(cat <<EOF
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_connector_module_path": "vllm_marconi_offload",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_pool_path": "$L1_PATH",
    "cpu_bytes_to_use": $L1_BYTES,
    "l2_pool_path": "$L2_PATH",
    "l2_bytes_to_use": $L2_BYTES,
    "eviction_policy": "marconi",
    "eviction_decay": 0.005,
    "eviction_reuse_mode": "linear",
    "admission_threshold": 1
  }
}
EOF
)

exec vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --enable-prefix-caching \
    --kv-transfer-config "$KV_TRANSFER_JSON"
