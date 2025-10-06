# FedGAN prototype (src/fedgan_gpt5_mini)

This is a minimal single-process FedGAN prototype based on the FedGAN paper summary and the TensorFlow DCGAN tutorial. It implements multiple Agent instances (each with a local Generator and Discriminator), an Intermediary (server) that aggregates parameters by weighted average, and a CLI training loop that saves per-epoch generated images, loss CSV/PNG, and a GIF summarizing progress.

Important: This prototype is intended for implementation-only delivery. It has not been fully run or validated in this workspace. The code targets Python 3.12 and TensorFlow 2.16.x as pinned in `requirements.txt`.

Files of interest
- `requirements.txt` - pinned packages to install in a venv (under this folder)
- `models/generator.py` - generator model (MNIST / DCGAN-like)
- `models/discriminator.py` - discriminator model
- `utils/data_loader.py` - loads MNIST and splits data per-agent
- `utils/visuals.py` - image grid saving, GIF, and loss plotting
- `agents/agent.py` - Agent class: local training and weight access
- `server/intermediary.py` - Aggregation (weighted average by data size)
- `main.py` - CLI entrypoint to run single-process FedGAN training loop

Quick setup (macOS, zsh)

1. Create and activate a Python 3.12 venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r src/fedgan_gpt5_mini/requirements.txt
```

2. Run the prototype

From the repository root run (module mode ensures package-relative imports work):

```bash
python -m src.fedgan_gpt5_mini.main --agents 3 --epochs 50 --sync_k 5 --batch_size 32 --outdir tmp_outputs
```

Notes and caveats
- The code uses `tf.keras` APIs; some static analyzers in this environment flagged tf.keras as unknown — this is a static lint issue and does not affect runtime if TensorFlow is installed.
- The implementation is single-process and does not implement network communication between Agents and the Intermediary. Synchronization is performed by direct method calls in the Intermediary.
- The prototype saves generated images per agent per epoch, per-agent loss CSV/PNG files, and a GIF assembled from agent 0's images.

Recommended next steps
- Run a short smoke test (e.g., `--epochs 2 --agents 2 --batch_size 64`) to validate the runtime and adjust memory/device settings (tensorflow-metal is included for macOS acceleration).
- Add unit tests for Agent weight get/set and Intermediary aggregation.
