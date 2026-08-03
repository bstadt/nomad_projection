# NOMAD Projection
<img src="wiki_transparent.png" alt="NOMAD Projection" width="512">

Negative Or Mean Affinity Discrimination (NOMAD) Projection is a massively scalable method for nonlinear dimensionality reduction.
It is the fastest and easiest way to compute t-SNE or UMAP style visualizations of multi-million point datasets.

Please contact brandon@nomic.ai with inquiries!

## Installation
You can install NOMAD Projection with pip:
```bash
pip install nomad-projection
```

## Usage
```python
from nomad_projection import NomadProjection

p = NomadProjection()

#Required Parameters
lowd = p.fit_transform(X=x,
                       epochs=100,
                       batch_size=80000)

#All Parameters
lowd = p.fit_transform(X=x,
                       epochs=100,
                       batch_size=80000,
                       n_neighbors=8,
                       n_noise=10000,
                       n_cells=5,
                       cluster_subset_size=5000000,
                       momentum=0.8,
                       lr_scale=0.1,
                       learning_rate_decay_start_time=0.3,
                       late_exaggeration_time=1.7,
                       late_exaggeration_scale=1.2,
                       late_exaggeration_n_noise=10000,
                       )
```

## Projecting graphs directly

When your data *is* a large graph, the kNN is already sitting in the adjacency structure — there is no reason to run the (expensive, OOM-prone) per-cell similarity search. Pass a `(n, k)` neighbor table instead of a feature matrix and the search is skipped entirely:

```python
from nomad_projection import NomadProjection
from nomad_projection.partition import topk_neighbors_from_csr

neighbors = topk_neighbors_from_csr(adjacency_csr, k=16)  # (n, k) ids, -1 padded
# ...or build (n, k) yourself: each row = a node's neighbor ids by descending weight

p = NomadProjection()
lowd = p.fit_transform(neighbors=neighbors, epochs=100, batch_size=40000,
                       n_cells=16, n_neighbors=8, n_noise=2000)
```

- With only `neighbors`, cells come from the graph itself — see **Cells decide how much graph you keep** below. Init is random.
- Pass `X` *alongside* `neighbors` and the features are used for the balanced partition and PCA init, while the kNN still comes straight off the graph — the recommended setup when you have both (e.g. SVD vectors + the interaction graph they came from).
- Per-node neighbor lists shorter than `n_neighbors` are repeated cyclically; isolated nodes get a self-loop (a no-op positive force).

## Cells decide how much graph you keep

In graph mode each node keeps only the neighbors that land in its **own cell**; every
crossing edge is dropped before training, and a node left with no same-cell neighbor gets
a self-loop, which is a no-op force. The partition therefore sets a hard ceiling on how
much of the graph the optimizer can see, and no learning rate or epoch count recovers what
it discarded.

`graph_partition` selects how those cells are formed:

| value | cells from |
|-------|------------|
| `'auto'` (default) | metis when pymetis is installed, else `'chop'` |
| `'metis'` | balanced minimum k-way cut (`pip install pymetis`) |
| `'chop'` | label propagation, then an equal contiguous chop |

`'chop'` constrains balance exactly but never counts crossing edges, which is fine when
communities are small and clean and expensive when they are not. Measured on a 630k-node /
5.7M-edge TikTok repost graph at `n_cells=8`:

| partition | edges kept | nodes with no same-cell neighbor | reciprocal-pair distance in the layout |
|-----------|-----------:|---------------------------------:|---------------------------------------:|
| `chop` | 0.293 | 34.8% | 0.625 |
| `metis` | **0.647** | **0.1%** | **0.021** |

The last column is the median distance between reciprocal pairs divided by the median
random-pair distance, so 1.0 would mean the layout ignored the graph entirely. The METIS
partition itself took 6s on that graph. Note that even a minimum cut keeps only ~65% of
edges here — large social graphs have no good balanced cuts — so cells still bound quality;
min-cut raises the ceiling rather than removing it.

## Late exaggeration

Attraction can be boosted for the tail of training, the way t-SNE's early exaggeration works but
inverted: let the layout spread first, then pull it onto its structure. `pos_weight` becomes
`late_exaggeration_scale` once `t = epoch/epochs` passes `late_exaggeration_time`.

**This did not work before.** `late_exaggeration_time` defaulted to `1.1` while `t` never exceeds
`1.0`, so the branch was unreachable and `late_exaggeration_scale` and `late_exaggeration_n_noise`
were silently ignored in every run. The default is now `0.6`; with `late_exaggeration_scale=1`
(unchanged) and `late_exaggeration_n_noise` following `n_noise`, the defaults still behave exactly
as before — the parameters simply do something when you set them. A time `>= 1.0` now warns.

The two knobs are not independent. What predicts the result is **`(1 - t) x scale`** — how long the
boost runs times how hard it pulls. On a 630k-node repost graph (k=8, 800 epochs, 8 GPUs):

| t | scale | (1-t)&times;scale | extent | occupied bins | result |
|---|-------|------------|--------|---------------|--------|
| — | 1 (off) | 0 | 28.0 | 39% | uniform grain, no structure legible |
| 0.3 | 2 | 1.4 | 21.1 | 18% | soft clumping |
| **0.6** | **4** | **1.6** | 21.1 | 14% | **cleanly separated clusters with filaments** |
| 0.3 | 3 | 2.1 | 14.4 | 11% | filaments radiating from a dense core |
| 0.6 | 8 | 3.2 | 14.5 | 6% | sharpest separation, sparser field |
| 0.3 | 8 | 5.6 | 2.4 | 2% | over-collapsed |
| 0.3 | 32 | 22.4 | 0.72 | 2% | crushed to a point cloud |

Useful structure lives around 1.5-2; past ~3 the layout implodes.

Note that raising attraction is not interchangeable with lowering repulsion. Cutting `n_noise`
reduces the runaway growth of the embedding but produces no cluster structure, and taken far enough
(`n_noise=16`) it yields a ring: with random negatives that sparse, the mean-affinity term — each
point repelled from the other cells' centroids — dominates and evacuates the middle.

## Balanced partitioning

Centroid-based clustering (upstream's LSH k-means) can put nearly all points into one cell on heavy-tailed data — one mass cell OOMs the per-cell kNN and shards terribly across GPUs. This fork's default partition (`partition='balanced'`) is recursive bisection along principal directions with the cut at the *rank* quantile, so cell sizes are exact (within ±1 point per split) by construction, on any data distribution. Every GPU gets an identically-sized shard. The upstream behavior remains available with `partition='lsh'`.

## Running on Modal (8xH100)

This fork is validated on an 8xH100 [Modal](https://modal.com) container. To smoke-test it end to end, run this from the repository root:

```bash
pip install modal
modal setup   # first time only: authenticates your Modal account
modal run examples/modal_smoke_8xh100.py
```

No other local dependencies are needed — the container image defined in the example pins everything (`torch==2.9.0`, `scipy==1.13.1`, numpy/scikit-learn/matplotlib/tqdm) and mounts the local `nomad_projection` package, so the code you edit locally is the code that runs remotely.

**Validated configuration** (2026-07-30, 2M points, 10 epochs, `batch_size=40000`, `n_noise=2000`, `gpu="H100:8"`):

| scenario | n_cells | GPUs used | wall clock | result |
|----------|---------|-----------|------------|--------|
| features (64d blobs) | 16 | 8 (2 cells/GPU) | 156s | cells exactly 125,000 each; all coords finite |
| features | 5 (upstream README default) | 5 via fallback | 132s | cells exactly 400,000 each; all coords finite |
| features, 75% of mass in one blob | 16 | 8 | 164s | cells still exactly 125,000 each (LSH k-means collapses here) |
| graph mode (community graph, no X) | 16 | 8 | 235s | kNN search skipped; cells exactly even; 92% of edges intra-cell |

Run a subset of scenarios with e.g. `modal run examples/modal_smoke_8xh100.py --only graph_mode`. For long runs prefer `modal run --detach` — an ephemeral app is otherwise stopped server-side if the client connection blips.

To project your own data instead of the synthetic blobs, replace the array construction at the top of `smoke()` with a load from a [Modal Volume](https://modal.com/docs/guide/volumes) (generate or upload your `.npy` there first; passing hundreds of MB through `.remote()` arguments works but is slow). For large real datasets, start from the parameter notes at the bottom of this section.

Things this fork fixes/pins that upstream does not, all discovered the hard way:

- **`torch==2.9.0` is required for multi-GPU on Modal.** NOMAD's distributed training shares CUDA tensors between the parent process and its `mp.spawn` workers via CUDA IPC. Modal's gVisor sandbox blocks the `pidfd_getfd` syscall that other torch versions use for this (2.7.1 and 2.12 both fail in testing), which breaks worker startup; 2.9.0 carries a working fallback. Single-GPU runs work on any torch version.
- **`n_cells` must be at least the GPU count** (cells are assigned to GPUs round-robin). Upstream crashes with `torch.cat` on an empty list when a rank gets zero cells — e.g. the README-default `n_cells=5` on 8 GPUs. This fork falls back to using `n_cells` GPUs instead. Prefer `n_cells` as a multiple of the GPU count so ranks hold even cluster counts.
- Missing `nullcontext` import that crashed the CPU / pre-Ampere autocast path.
- The GPU-side clusterer is released before workers are spawned, so its CUDA tensors are not needlessly IPC-shared with (and re-materialized in) every worker.

- **Per-step GPU memory scales with `batch_size * n_noise`.** Each training step materializes several `(batch_size, n_noise)`-shaped tensors (the int64 noise-index tensor alone is `batch_size * n_noise * 8` bytes) plus their autograd graph. The usage-example values above (`batch_size=80000, n_noise=10000`) need more than 80GiB and OOM an H100 after the first step; `batch_size=40000, n_noise=2000` peaks well under 10GiB per GPU.

Practical parameter notes from projecting a 28.8M x 128 dataset on 8xH100 (~large-run defaults): `lr_scale=0.015`, `late_exaggeration_scale=1.0`, `n_neighbors=64`, `n_cells=16`. Beware very skewed cluster size distributions: a single dominant cell OOMs the per-cell kNN, and many small cells (64+) make it drastically slower.

## Paper Replication

### Environment Setup
Due to the heterogeneous nature of python package management and cuda configuration, replicating the paper requires managing 3 different environments.

#### Nomad Projection Environment:
The nomad projection environment is the managed with venv.
Simply run the following commands from the root of the repository to create it:
```bash
python3 -m venv nomad_projection_env
source nomad_projection_env/bin/activate
pip install .
```

#### t-SNE-CUDA Environment:
The t-SNE-CUDA environment requires miniconda to be installed.
First, follow the instructions [here](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) to install miniconda.
Then, follow the setup on the t-SNE-CUDA [repository](https://github.com/CannyLab/tsne-cuda).
Finally, run the following commands from the root of the repository in your conda environment:
```bash
conda install pytorch click scikit-learn pandas
pip install -e .
```

#### RAPIDS UMAP Environment:
The RAPIDS UMAP environment reuires a custom conda environment which is generated from the [RAPIDS installation selector](https://docs.rapids.ai/install/).
For the paper, the following command was used:
```bash
conda create -n rapids-24.10 -c rapidsai -c conda-forge -c nvidia  \
    rapids=24.10 python=3.12 'cuda-version>=12.0,<=12.5'
```
Once this command executes, run the following commands from the root of the repository in your rapisd conda environment:
```bash
conda install pytorch click scikit-learn pandas
pip install -e .
```

### Input Data
Input data can be downloaded from R2.
To gain access, run `aws configure` with the following credentials:

Access Key ID: `94bef7d178281190c5ca48f483b6504b`

Secret Access Key: `ac885a46694e8e1a073375b8da1961be42371a9f4434bd58ffd3e5c46a3be67b`

Then run the following command from the root of the repository to download the data (please note that this will download nearly a terabyte of data):
```bash
aws s3 sync --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://nomad-projection-input-data ./data
```

### Figures
NOMAD Projection uses the figures submodule to manage generation of the figures in the paper.

#### ArXiv and Imagenet

Reproducing the arXiv and Imagenet figures requires two steps:
1. Run commands to generate results from each algorithm for each dataset.
2. Assemble the results into the final plot.


From the nomad projection environment, run the following command:
```python
python nomad_project/figures/arxiv.py --nomad
```
```python
python nomad_project/figures/imagenet.py --nomad
```

From the t-SNE-CUDA environment, run the following commands:
```python
python nomad_project/figures/arxiv.py --tsnecuda
```
```python
python nomad_project/figures/imagenet.py --tsnecuda
```

From the RAPIDS UMAP environment, run the following commands:
```python
python nomad_project/figures/arxiv.py --rapids-umap
```
```python
python nomad_project/figures/imagenet.py --rapids-umap
```

Finally, assemble the results into the final plot in the nomad projection environment
```python
python nomad_project/figures/arxiv.py --plot
```
```python
python nomad_project/figures/imagenet.py --plot
```
The output will be stored in the results directory in the root of the repository.

### PubMed
From the nomad projection environment, run the following command:
```python
python nomad_project/figures/pubmed.py --nomad
```
The output will be stored in the results directory in the root of the repository.

### Multilingual Wikipedia
```python
python nomad_project/figures/wiki.py --nomad
```
The output will be stored in the results directory in the root of the repository.
