"""Smoke-test NOMAD Projection on an 8xH100 Modal container.

Run from the repository root:

    modal run examples/modal_smoke_8xh100.py

Notes on the environment pins:
- torch==2.9.0 is required for multi-GPU on Modal. NOMAD shares CUDA tensors
  between the parent process and mp.spawn workers via CUDA IPC; Modal's gVisor
  sandbox blocks the pidfd_getfd syscall that other torch versions (2.7.1,
  2.12) rely on for this. 2.9.0 carries a working fallback.
- n_cells should be >= (ideally a multiple of) the GPU count. Cells are
  assigned to GPUs round-robin, and this fork falls back to fewer GPUs when
  there are fewer cells than devices.
"""

import modal

app = modal.App("nomad-projection-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.0",
        "numpy",
        "scikit-learn",
        "scipy==1.13.1",
        "matplotlib",
        "tqdm",
    )
    .add_local_python_source("nomad_projection")
)


@app.function(gpu="H100:8", image=image, timeout=60 * 60, memory=65536)
def smoke(n_points: int, dim: int, n_cells: int, epochs: int, batch_size: int) -> dict:
    import time

    import numpy as np
    import torch

    from nomad_projection import NomadProjection

    print(f"visible GPUs: {torch.cuda.device_count()}, torch {torch.__version__}")

    rng = np.random.default_rng(0)
    n_blobs = 32
    centers = (rng.normal(size=(n_blobs, dim)) * 5).astype(np.float32)
    blob_ids = rng.integers(0, n_blobs, size=n_points)
    X = centers[blob_ids] + rng.normal(size=(n_points, dim)).astype(np.float32)

    start = time.time()
    p = NomadProjection()
    coords = p.fit_transform(X=X, epochs=epochs, batch_size=batch_size, n_cells=n_cells)
    elapsed = time.time() - start

    assert coords.shape == (n_points, 2), coords.shape
    finite_frac = float(np.isfinite(coords).all(axis=1).mean())

    # Blob separation: spread of blob centroids vs. spread within blobs.
    centroids = np.stack([coords[blob_ids == b].mean(axis=0) for b in range(n_blobs)])
    within = float(np.mean([coords[blob_ids == b].std() for b in range(n_blobs)]))
    between = float(centroids.std())

    return {
        "n_points": n_points,
        "n_cells": n_cells,
        "world_size": p.world_size,
        "seconds": round(elapsed, 1),
        "finite_frac": finite_frac,
        "coord_std": [float(coords[:, 0].std()), float(coords[:, 1].std())],
        "between_blob_std": between,
        "within_blob_std": within,
    }


@app.local_entrypoint()
def main(n_points: int = 2_000_000, dim: int = 64, epochs: int = 10, batch_size: int = 80000):
    # n_cells=16: even split across 8 GPUs (2 cells each).
    # n_cells=5 (the README default): fewer cells than GPUs, exercises the
    # world_size fallback that used to crash upstream.
    for n_cells in (16, 5):
        result = smoke.remote(n_points, dim, n_cells, epochs, batch_size)
        print(f"n_cells={n_cells}: {result}")
        assert result["finite_frac"] == 1.0, "non-finite coordinates"
        assert min(result["coord_std"]) > 0, "degenerate embedding"
    print("SMOKE PASSED")
