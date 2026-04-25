# Runpod Startup Checklist (Cursor + Remote Kernel)

Use this every time you launch a new pod.

## 1) Start pod and open Jupyter URL

1. In Runpod, wait until pod status is `Ready`.
2. Open the `Jupyter Lab` link (port `8888`).
3. If prompted for token, get it from pod terminal:

```bash
jupyter server list
```

Copy the URL/token shown there.

## 2) Connect Cursor to the remote kernel

In Cursor notebook UI:

1. Open `my_work/notebooks/runpod_baseline.ipynb`.
2. `Select Kernel` -> `Select Another Kernel...` -> `Existing Jupyter Server...`
3. Paste:

```text
https://<your-runpod-host>/?token=<your-token>
```

4. Select the remote `Python 3 (ipykernel)` kernel.

## 3) Pod terminal bootstrap (repo + sync)

Run once per fresh pod (or after re-creating pod):

```bash
cd /workspace
if [ ! -d thesis_circuit_breaker ]; then
  git clone https://github.com/hollolaszlo88/thesis_circuit_breaker.git
fi
cd /workspace/thesis_circuit_breaker
git pull
```

## 4) Environment variables to verify

In Runpod pod config, ensure:

- `HF_TOKEN` is set (valid Hugging Face token with Gemma access)
- optional: `CT_REPO_DIR=/workspace/thesis_circuit_breaker`

After editing env vars, restart kernel (or pod).

## 5) Notebook run order

In `runpod_baseline.ipynb`, run in order:

1. `Runtime + Hugging Face Setup`
2. `Imports`
3. `Runtime sanity checks`
4. Model load cell

If any setup cell installs/pins packages, restart kernel once and rerun from top.

## 6) Sanity checks (must pass)

Run:

```python
import os, torch
print("cwd:", os.getcwd())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("HF token set:", bool(os.environ.get("HF_TOKEN")))
```

Expected:

- `cuda available: True`
- GPU name shown (A4500/A5000/etc.)
- `HF token set: True`

## 7) Verify persistent cache is active

Run:

```python
import os
for k in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "CT_OUTPUT_DIR"]:
    print(k, "=", os.environ.get(k))
```

Expected paths under `/workspace/...`.

Optional cache size check:

```bash
du -sh /workspace/hf/hub
du -sh /workspace/hf/transformers
```

## 8) If imports fail

In pod terminal:

```bash
cd /workspace/thesis_circuit_breaker
pip install -e .
```

Restart kernel, rerun from top.

## 9) If CUDA is False

In notebook cell:

```python
%pip uninstall -y torch torchvision torchaudio
%pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
```

Restart kernel and re-run sanity checks.

## 10) Before stopping pod

- Confirm important notebook/code changes are in local git.
- If outputs matter, save/copy from `/workspace/results/...`.
- Stop pod to avoid charges.
- Stop dialog should not say "no volume configured" if network volume is attached.
