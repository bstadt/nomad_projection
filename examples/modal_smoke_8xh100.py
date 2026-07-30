"""Smoke-test NOMAD Projection on an 8xH100 Modal container.

Run from the repository root:

    modal run examples/modal_smoke_8xh100.py

Covers four scenarios:
1. Feature mode, n_cells=16 (2 cells/GPU, balanced partition).
2. Feature mode, n_cells=5 — fewer cells than GPUs, exercises the
   world_size fallback that used to crash upstream.
3. Feature mode on pathologically skewed data (75% of points in one tight
   blob) — the case where LSH k-means produced one mass cell; the balanced
   partition must still shard evenly.
4. Graph mode — neighbor tables read straight off a synthetic community
   graph, no feature matrix and no kNN search at all.

Notes on the environment pins:
- torch==2.9.0 is required for multi-GPU on Modal. NOMAD shares CUDA tensors
  between the parent process and mp.spawn workers via CUDA IPC; Modal's gVisor
  sandbox blocks the pidfd_getfd syscall that other torch versions (2.7.1,
  2.12) rely on for this. 2.9.0 carries a working fallback.
- n_cells should be >= (ideally a multiple of) the GPU count. Cells are
  assigned to GPUs round-robin, and this fork falls back to fewer GPUs when
  there are fewer cells than devices.
- Per-step GPU memory scales with batch_size * n_noise; the upstream README
  defaults (80000 x 10000) OOM an 80GiB H100. 40000 x 2000 peaks well under
  10GiB.
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

with image.imports():
    import numpy as np


def _blobs(n_points, dim, n_blobs, skewed=False, seed=0):
    rng = np.random.default_rng(seed)
    centers = (rng.normal(size=(n_blobs, dim)) * 5).astype(np.float32)
    if skewed:
        # 75% of points in blob 0, the rest spread over the others
        blob_ids = np.where(
            rng.random(n_points) < 0.75,
            0,
            rng.integers(1, n_blobs, size=n_points),
        )
        scale = np.where(blob_ids == 0, 0.05, 1.0).astype(np.float32)[:, None]
    else:
        blob_ids = rng.integers(0, n_blobs, size=n_points)
        scale = np.float32(1.0)
    X = centers[blob_ids] + rng.normal(size=(n_points, dim)).astype(np.float32) * scale
    return X, blob_ids


def _community_graph(n_points, k, n_comm, cross_frac=0.05, seed=0):
    """(n, k) neighbor table: mostly same-community neighbors, some rewired."""
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, n_comm, size=n_points)
    neighbors = np.empty((n_points, k), dtype=np.int64)
    for c in range(n_comm):
        ids = np.flatnonzero(comm == c)
        neighbors[ids] = ids[rng.integers(0, len(ids), size=(len(ids), k))]
    cross = rng.random((n_points, k)) < cross_frac
    neighbors[cross] = rng.integers(0, n_points, size=int(cross.sum()))
    return neighbors, comm


def _summarize(p, coords, group_ids, elapsed):
    n_groups = int(group_ids.max()) + 1
    centroids = np.stack(
        [coords[group_ids == b].mean(axis=0) for b in range(n_groups)]
    )
    sizes = np.bincount(p.cluster_assignments)
    return {
        "world_size": p.world_size,
        "seconds": round(elapsed, 1),
        "finite_frac": float(np.isfinite(coords).all(axis=1).mean()),
        "coord_std": [float(coords[:, 0].std()), float(coords[:, 1].std())],
        "between_group_std": float(centroids.std()),
        "within_group_std": float(
            np.mean([coords[group_ids == b].std() for b in range(n_groups)])
        ),
        "cell_sizes_min_max": [int(sizes.min()), int(sizes.max())],
    }


@app.function(gpu="H100:8", image=image, timeout=60 * 60, memory=65536)
def smoke_features(
    n_points: int, dim: int, n_cells: int, epochs: int, batch_size: int,
    n_noise: int, skewed: bool,
) -> dict:
    import time
    import torch
    from nomad_projection import NomadProjection

    print(f"visible GPUs: {torch.cuda.device_count()}, torch {torch.__version__}")
    X, blob_ids = _blobs(n_points, dim, n_blobs=32, skewed=skewed)

    start = time.time()
    p = NomadProjection()
    coords = p.fit_transform(
        X=X, epochs=epochs, batch_size=batch_size, n_cells=n_cells, n_noise=n_noise
    )
    return _summarize(p, coords, blob_ids, time.time() - start)


@app.function(gpu="H100:8", image=image, timeout=60 * 60, memory=65536)
def smoke_graph(
    n_points: int, k: int, n_cells: int, epochs: int, batch_size: int, n_noise: int
) -> dict:
    import time
    import torch
    from nomad_projection import NomadProjection

    print(f"visible GPUs: {torch.cuda.device_count()}, torch {torch.__version__}")
    neighbors, comm = _community_graph(n_points, k, n_comm=32)

    start = time.time()
    p = NomadProjection()
    coords = p.fit_transform(
        neighbors=neighbors,
        epochs=epochs,
        batch_size=batch_size,
        n_cells=n_cells,
        n_neighbors=8,
        n_noise=n_noise,
    )
    out = _summarize(p, coords, comm, time.time() - start)
    labels = p.cluster_assignments
    out["intra_cell_edge_frac"] = float(
        (labels[neighbors.ravel()] == np.repeat(labels, k)).mean()
    )
    return out


@app.local_entrypoint()
def main(
    n_points: int = 2_000_000,
    dim: int = 64,
    epochs: int = 10,
    batch_size: int = 40000,
    n_noise: int = 2000,
    only: str = "",
):
    """Run all scenarios, or a comma-separated subset via --only."""
    wanted = set(only.split(",")) if only else None
    results = {}

    for name, n_cells, skewed in (
        ("features_16cells", 16, False),
        ("features_5cells_fallback", 5, False),
        ("features_skewed", 16, True),
    ):
        if wanted is not None and name not in wanted:
            continue
        r = smoke_features.remote(n_points, dim, n_cells, epochs, batch_size, n_noise, skewed)
        print(f"{name}: {r}")
        assert r["finite_frac"] == 1.0, name
        assert min(r["coord_std"]) > 0, name
        lo, hi = r["cell_sizes_min_max"]
        assert hi - lo <= n_cells, f"{name}: unbalanced cells {lo}..{hi}"
        results[name] = r

    if wanted is None or "graph_mode" in wanted:
        r = smoke_graph.remote(n_points, 16, 16, epochs, batch_size, n_noise)
        print(f"graph_mode: {r}")
        assert r["finite_frac"] == 1.0
        assert min(r["coord_std"]) > 0
        assert r["world_size"] == 8
        lo, hi = r["cell_sizes_min_max"]
        assert hi - lo <= 16, f"graph: unbalanced cells {lo}..{hi}"
        results["graph_mode"] = r

    print("SMOKE PASSED")
