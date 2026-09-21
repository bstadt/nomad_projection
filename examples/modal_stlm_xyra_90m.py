"""NOMAD projection of the Xyra 2022-2026 mutual-interaction graph, on Modal.

97,814,228 accounts / 514,006,155 undirected mutual edges — the graph behind
xiq's NP512/NP1024 embeddings, mirrored to
s3://calcifer-hot/stlm-1/xyra-2022-2026/kernel/full-630991582a6b7b64-host-v1/.

Five stages, each resumable off a Modal Volume, deliberately split so the two
expensive-but-different resources are never bought at the same time: the METIS
partition is a long single-threaded CPU job and has no use for a GPU, so it
runs on a CPU box and hands `cell_labels` to the 8xH100 run.

    modal run examples/modal_stlm_xyra_90m.py::fetch
    modal run examples/modal_stlm_xyra_90m.py::build_neighbors
    modal run --detach examples/modal_stlm_xyra_90m.py::partition
    modal run --detach examples/modal_stlm_xyra_90m.py::project
    modal run examples/modal_stlm_xyra_90m.py::render

Input layout (see stlm-1/kernelize/pack_arrays.py, which wrote it):
  mutual_edges_u.bin    uint32 x E, edge source as a compact id
  mutual_edges_v.bin    uint32 x E, edge target
  pos_weights_recip.bin float32 x 2E, cat(u,v),cat(v,u) aligned; both halves
                        hold the same per-edge kernel weight 1/max(rank_ab, rank_ba)
  nodes.npz             uids (uint64, compact id -> raw X uid), deg (int32)
"""

import modal

app = modal.App("nomad-stlm-xyra-90m")
vol = modal.Volume.from_name("nomad-stlm-xyra", create_if_missing=True)
aws = modal.Secret.from_name("calcifer-aws")

BUCKET = "calcifer-hot"
PREFIX = ("stlm-1/xyra-2022-2026/kernel/full-630991582a6b7b64-host-v1"
          "/packed-observations-v1/arrays")
N_NODES = 97_814_228
N_EDGES = 514_006_155
FILES = ["mutual_edges_u.bin", "mutual_edges_v.bin", "pos_weights_recip.bin", "nodes.npz"]

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==1.26.4", "boto3")

# partition.py imports torch at module scope, so the CPU box needs the (cpu-only)
# wheel to call the fork's own METIS partitioner rather than a reimplementation.
metis_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy==1.26.4", "scikit-learn", "matplotlib", "tqdm", "pymetis")
    .add_local_python_source("nomad_projection")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.0", "numpy", "scikit-learn", "scipy==1.13.1",
                 "matplotlib", "tqdm")
    .add_local_python_source("nomad_projection")
)

viz_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==1.26.4", "pandas", "datashader", "colorcet", "pillow", "boto3")


@app.function(image=cpu_image, volumes={"/vol": vol}, secrets=[aws],
              cpu=8, memory=16384, timeout=3 * 3600)
def fetch():
    """Mirror the packed arrays from S3 onto the volume. Idempotent."""
    import os
    import time

    import boto3
    from boto3.s3.transfer import TransferConfig

    s3 = boto3.client("s3", region_name="us-east-2")
    cfg = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=16)
    os.makedirs("/vol/raw", exist_ok=True)

    out = {}
    for name in FILES:
        key, dst = f"{PREFIX}/{name}", f"/vol/raw/{name}"
        size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            out[name] = f"cached {size}"
            continue
        t0 = time.time()
        s3.download_file(BUCKET, key, dst, Config=cfg)
        got = os.path.getsize(dst)
        assert got == size, f"{name}: {got} != {size}"
        out[name] = f"{size} in {time.time() - t0:.0f}s"
        print(f"{name}: {out[name]}", flush=True)
    vol.commit()
    return out


@app.function(image=cpu_image, volumes={"/vol": vol}, cpu=16, memory=98304,
              timeout=6 * 3600)
def build_neighbors(k: int = 16, n_buckets: int = 16):
    """(n, k) neighbor table, each row a node's mutuals by descending kernel weight.

    Done bucket-by-bucket over contiguous source-id ranges rather than as one
    lexsort over all 1.028B directed edges: a single sort of that many keys
    needs ~35GB of index arrays alone, while 16 buckets peak under 15GB and
    cost only a few extra linear scans of the (memory-bandwidth-bound) edge
    arrays.
    """
    import time

    import numpy as np

    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    u = np.fromfile("/vol/raw/mutual_edges_u.bin", dtype=np.uint32)
    v = np.fromfile("/vol/raw/mutual_edges_v.bin", dtype=np.uint32)
    E = u.size
    assert (E, v.size) == (N_EDGES, N_EDGES), (E, v.size)
    w_both = np.fromfile("/vol/raw/pos_weights_recip.bin", dtype=np.float32)
    assert w_both.size == 2 * E

    # pack_arrays.py writes the same per-edge weight into both halves; confirm
    # rather than assume, then keep only one.
    probe = np.random.default_rng(0).integers(0, E, 200_000)
    assert np.array_equal(w_both[probe], w_both[E + probe]), "weight halves differ"
    w = np.ascontiguousarray(w_both[:E])
    del w_both
    log(f"loaded E={E}, w in [{w.min():.3g}, {w.max():.3g}]")

    n = N_NODES
    assert int(max(u.max(), v.max())) < n
    out = np.full((n, k), -1, dtype=np.int32)

    bounds = np.linspace(0, n, n_buckets + 1).astype(np.int64)
    for bi in range(n_buckets):
        lo, hi = int(bounds[bi]), int(bounds[bi + 1])
        mu = (u >= lo) & (u < hi)
        mv = (v >= lo) & (v < hi)
        src = np.concatenate([u[mu], v[mv]]).astype(np.int64)
        dst = np.concatenate([v[mu], u[mv]])
        wt = np.concatenate([w[mu], w[mv]])
        del mu, mv

        order = np.lexsort((-wt, src))
        src, dst = src[order], dst[order]
        del order, wt

        counts = np.bincount(src - lo, minlength=hi - lo)
        indptr = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        rank = np.arange(src.size, dtype=np.int64) - indptr[src - lo]
        keep = rank < k
        out[src[keep], rank[keep]] = dst[keep]
        log(f"bucket {bi + 1}/{n_buckets}: nodes {lo}..{hi}, {src.size} directed edges")
        del src, dst, rank, keep, counts, indptr

    np.save("/vol/neighbors_k%d.npy" % k, out)
    deg = np.load("/vol/raw/nodes.npz")["deg"]
    np.save("/vol/degrees.npy", deg)

    filled = (out >= 0).sum(axis=1)
    stats = {
        "n": int(n),
        "k": int(k),
        "directed_edges": int(2 * E),
        "edges_kept": int(filled.sum()),
        "edge_keep_frac": float(filled.sum() / (2 * E)),
        "isolated_nodes": int((filled == 0).sum()),
        "deg_median": float(np.median(deg)),
        "deg_max": int(deg.max()),
        "deg_ge_k_frac": float((deg >= k).mean()),
        "seconds": round(time.time() - t0, 1),
    }
    log(str(stats))
    vol.commit()
    return stats


@app.function(image=metis_image, volumes={"/vol": vol}, cpu=16, memory=196608,
              timeout=12 * 3600)
def partition(n_cells: int = 16, k_part: int = 6, k: int = 16, method: str = "metis"):
    """Cells for graph mode, as a standalone CPU job.

    GraphKNN drops every edge that crosses a cell boundary, so the partition
    sets a hard ceiling on how much of the graph the optimizer ever sees. METIS
    is run on the k_part strongest neighbors per node rather than all k: the cut
    it finds is driven by the heavy edges anyway, and the smaller graph keeps
    METIS's working set manageable at 10^8 nodes. The labels are then applied to
    the full-width table.
    """
    import time

    import numpy as np
    from nomad_projection.partition import (graph_partition_labels,
                                            graph_partition_labels_metis)

    t0 = time.time()
    neighbors = np.load(f"/vol/neighbors_k{k}.npy")
    sub = np.ascontiguousarray(neighbors[:, :k_part])
    del neighbors
    print(f"partitioning on k={k_part} table, {int((sub >= 0).sum())} entries",
          flush=True)

    used = method
    if method == "metis":
        try:
            labels = graph_partition_labels_metis(sub, n_cells)
        except Exception as e:  # noqa: BLE001 - want the fallback on any METIS failure
            print(f"METIS failed ({type(e).__name__}: {e}); falling back to chop",
                  flush=True)
            used, labels = "chop", graph_partition_labels(sub, n_cells)
    else:
        labels = graph_partition_labels(sub, n_cells)

    name = f"labels_{used}_c{n_cells}.npy"
    np.save(f"/vol/{name}", labels.astype(np.int64))

    # How much of the graph survives this partition, measured on the full table.
    neighbors = np.load(f"/vol/neighbors_k{k}.npy", mmap_mode="r")
    kept = 0
    total = 0
    no_same_cell = 0
    for i in range(0, N_NODES, 4_000_000):
        nb = np.asarray(neighbors[i:i + 4_000_000])
        valid = nb >= 0
        own = np.broadcast_to(labels[i:i + nb.shape[0]][:, None], nb.shape)
        same = np.zeros_like(valid)
        same[valid] = labels[nb[valid]] == own[valid]
        kept += int(same.sum())
        total += int(valid.sum())
        no_same_cell += int((same.sum(axis=1) == 0).sum())

    sizes = np.bincount(labels, minlength=n_cells)
    stats = {
        "partitioner": used,
        "labels_file": name,
        "n_cells": int(n_cells),
        "cell_sizes_min_max": [int(sizes.min()), int(sizes.max())],
        "edge_keep_frac": kept / max(total, 1),
        "nodes_with_no_same_cell_neighbor": no_same_cell,
        "no_same_cell_frac": no_same_cell / N_NODES,
        "seconds": round(time.time() - t0, 1),
    }
    print(stats, flush=True)
    vol.commit()
    return stats


@app.function(image=gpu_image, gpu="H100:8", volumes={"/vol": vol}, cpu=32,
              memory=262144, timeout=16 * 3600)
def project(labels_file: str, tag: str, k: int = 16, epochs: int = 200,
            batch_size: int = 40000, n_noise: int = 2000, n_neighbors: int = 8,
            lr_scale: float = 0.015, late_exaggeration_time: float = 0.6,
            late_exaggeration_scale: float = 4.0):
    import time

    import numpy as np
    import torch
    from nomad_projection import NomadProjection

    print(f"visible GPUs: {torch.cuda.device_count()}, torch {torch.__version__}",
          flush=True)
    neighbors = np.load(f"/vol/neighbors_k{k}.npy")
    labels = np.load(f"/vol/{labels_file}")
    print(f"neighbors {neighbors.shape} {neighbors.dtype}, labels {labels.shape}",
          flush=True)

    t0 = time.time()
    p = NomadProjection()
    coords = p.fit_transform(
        neighbors=neighbors,
        cell_labels=labels,
        epochs=epochs,
        batch_size=batch_size,
        n_neighbors=n_neighbors,
        n_noise=n_noise,
        lr_scale=lr_scale,
        late_exaggeration_time=late_exaggeration_time,
        late_exaggeration_scale=late_exaggeration_scale,
    )
    elapsed = time.time() - t0
    coords = np.asarray(coords, dtype=np.float32)
    np.save(f"/vol/coords_{tag}.npy", coords)

    # Neighbours must end up closer than random pairs or the layout ignored the
    # graph — every other sanity check passes on a layout that did.
    rng = np.random.default_rng(0)
    idx = rng.choice(N_NODES, 200_000, replace=False)
    nb = neighbors[idx, 0]
    ok = nb >= 0
    d_nb = np.linalg.norm(coords[idx[ok]] - coords[nb[ok]], axis=1)
    d_rand = np.linalg.norm(
        coords[idx[ok]] - coords[rng.integers(0, N_NODES, ok.sum())], axis=1)

    stats = {
        "tag": tag,
        "seconds": round(elapsed, 1),
        "world_size": p.world_size,
        "epochs": epochs,
        "finite_frac": float(np.isfinite(coords).all(axis=1).mean()),
        "coord_std": [float(coords[:, 0].std()), float(coords[:, 1].std())],
        "neighbor_vs_random": float(np.median(d_nb) / max(np.median(d_rand), 1e-9)),
    }
    print(stats, flush=True)
    vol.commit()
    return stats


@app.function(image=viz_image, volumes={"/vol": vol}, secrets=[aws], cpu=16,
              memory=131072, timeout=2 * 3600)
def render(tag: str, width: int = 4000, height: int = 4000, pct: float = 0.02,
           color_by_degree: bool = True):
    """Datashader renders of the finished layout."""
    import time

    import colorcet
    import datashader as ds
    import datashader.transfer_functions as tf
    import numpy as np
    import pandas as pd

    t0 = time.time()
    coords = np.load(f"/vol/coords_{tag}.npy")
    good = np.isfinite(coords).all(axis=1)
    coords = coords[good]

    # Percentile bounds, not min/max: a handful of runaway points otherwise
    # compress the whole layout into a few pixels.
    sample = coords[np.random.default_rng(0).choice(len(coords),
                                                    min(5_000_000, len(coords)),
                                                    replace=False)]
    x_range = tuple(np.percentile(sample[:, 0], [pct, 100 - pct]))
    y_range = tuple(np.percentile(sample[:, 1], [pct, 100 - pct]))
    print(f"x {x_range} y {y_range}", flush=True)

    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    if color_by_degree:
        deg = np.load("/vol/degrees.npy")[good]
        df["logdeg"] = np.log10(np.maximum(deg, 1)).astype(np.float32)

    cvs = ds.Canvas(plot_width=width, plot_height=height,
                    x_range=x_range, y_range=y_range)
    out = {}

    agg = cvs.points(df, "x", "y")
    img = tf.set_background(tf.shade(agg, cmap=colorcet.fire, how="eq_hist"), "black")
    path = f"/vol/render_{tag}_density_{width}.png"
    img.to_pil().save(path)
    out["density"] = path
    print(f"density done {time.time() - t0:.0f}s", flush=True)

    if color_by_degree:
        agg_d = cvs.points(df, "x", "y", ds.mean("logdeg"))
        img_d = tf.set_background(
            tf.shade(agg_d, cmap=colorcet.bmy, how="linear"), "black")
        path = f"/vol/render_{tag}_degree_{width}.png"
        img_d.to_pil().save(path)
        out["degree"] = path
        print(f"degree done {time.time() - t0:.0f}s", flush=True)

    vol.commit()
    blobs = {}
    for name, path in out.items():
        with open(path, "rb") as f:
            blobs[name] = f.read()
        print(f"{name}: {len(blobs[name]) / 1e6:.1f} MB", flush=True)
    return blobs


@app.local_entrypoint()
def main(stage: str = "all", tag: str = "c16metis", labels_file: str = "",
         epochs: int = 200, n_cells: int = 16):
    import os

    if stage in ("all", "fetch"):
        print("fetch:", fetch.remote())
    if stage in ("all", "build"):
        print("build_neighbors:", build_neighbors.remote())
    if stage in ("all", "partition"):
        st = partition.remote(n_cells=n_cells)
        print("partition:", st)
        labels_file = st["labels_file"]
    if stage in ("all", "project"):
        assert labels_file, "pass --labels-file"
        print("project:", project.remote(labels_file=labels_file, tag=tag,
                                         epochs=epochs))
    if stage in ("all", "render"):
        blobs = render.remote(tag=tag)
        os.makedirs("out", exist_ok=True)
        for name, data in blobs.items():
            p = f"out/xyra90m_{tag}_{name}.png"
            with open(p, "wb") as f:
                f.write(data)
            print(f"wrote {p} ({len(data) / 1e6:.1f} MB)")
