# Publishing checklist

The repo is set up so that everything can be published independently
of the upstream vLLM project. This doc captures the remaining steps
the maintainer (you) has to run once, with their credentials.

## 1. Pre-flight (already verified locally)

* `uv build` produces both sdist and wheel into `dist/`.
* `pytest tests/` passes 10/10 (1 skip is intentional — see
  `tests/test_against_stock_vllm.py`).
* Wheel install was verified in two environments:
  * stock-vLLM venv → `PATCH_STATUS = "applied"`
  * no-vLLM venv → import does not crash; clean warnings; `PATCH_STATUS
    = "skipped-no-vllm:ModuleNotFoundError"`.

## 2. GitHub repo

```bash
# Create the public repo first on github.com under your account / org.
# Then:
cd /workspace/vllm-marconi-offload
git remote add origin git@github.com:<your-org>/vllm-marconi-offload.git
git push -u origin main

# Tag the first release.
git tag -a v0.1.0 -m "v0.1.0 — initial release"
git push origin v0.1.0
```

## 3. PyPI

You'll need a PyPI account and an API token scoped to this project.
First-time publish: register the project name via TestPyPI to make
sure nothing breaks.

```bash
# TestPyPI dry-run (recommended for the very first publish):
uv publish --publish-url https://test.pypi.org/legacy/ \
    --token <your-testpypi-token> \
    dist/vllm_marconi_offload-0.1.0*

# Real PyPI:
uv publish \
    --token <your-pypi-token> \
    dist/vllm_marconi_offload-0.1.0*
```

After this lands, anyone with stock vLLM 0.20.x can do

```bash
pip install vllm-marconi-offload
```

and use the connector via `--kv-transfer-config` as shown in the README.

## 4. Verify the published artifact

In a clean venv:

```bash
uv venv /tmp/marconi-check --python 3.12
/tmp/marconi-check/bin/python -m pip install vllm==0.20.2 vllm-marconi-offload
/tmp/marconi-check/bin/python -c "
import vllm_marconi_offload as v
print('version:', v.__version__)
print('PATCH_STATUS:', v.PATCH_STATUS)
print('connector:', v.SimpleCPUOffloadConnector.__name__)
"
```

Expected output:

```
version: 0.1.0
PATCH_STATUS: applied
connector: SimpleCPUOffloadConnector
```

## 5. Bump cadence

* On every vLLM release that lands incompatible scheduler changes,
  re-run the integration test against the new version.
* If the AST matcher misses the assert, update the matcher and ship
  a `0.1.x` patch release; bump the compat matrix in the README.
* When the upstream PR removes the assert in vLLM, the patcher
  cleanly degrades to `PATCH_STATUS = "not-needed"` with no changes
  here — release a `0.2.x` to reflect "now works on upstream
  unmodified" if you want.
