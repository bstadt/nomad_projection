import numpy as np
import torch


def _principal_direction(X, idx, device, sample_cap=262144):
    """Unit-norm top principal direction of X[idx], estimated on a subsample."""
    if len(idx) > sample_cap:
        sub = idx[np.random.permutation(len(idx))[:sample_cap]]
    else:
        sub = idx
    S = torch.from_numpy(np.ascontiguousarray(X[sub])).to(device=device, dtype=torch.float32)
    S = S - S.mean(dim=0, keepdim=True)
    try:
        _, _, V = torch.pca_lowrank(S, q=1, niter=4)
        v = V[:, 0]
    except Exception:
        v = torch.randn(S.size(1), device=device)
    nrm = v.norm()
    if not torch.isfinite(nrm) or nrm == 0:
        v = torch.randn(S.size(1), device=device)
        nrm = v.norm()
    return v / nrm


def _project(X, idx, v, device, chunk=2_000_000):
    """X[idx] @ v computed in chunks, returned on CPU."""
    out = torch.empty(len(idx), dtype=torch.float32)
    for i in range(0, len(idx), chunk):
        c = idx[i:i + chunk]
        Xc = torch.from_numpy(np.ascontiguousarray(X[c])).to(device=device, dtype=torch.float32)
        out[i:i + len(c)] = (Xc @ v).cpu()
    return out


def balanced_partition_labels(X, n_cells, device=None):
    """Partition rows of X into n_cells cells whose sizes differ by at most 1.

    Recursive bisection: each subset is split along its top principal
    direction at the *rank* quantile matching the target cell counts, so
    balance is exact by construction regardless of the data distribution
    (unlike centroid-based clustering, which can put nearly all points in
    one cell on heavy-tailed data).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = X.shape[0]
    if n_cells < 1:
        raise ValueError('n_cells must be >= 1')
    labels = np.empty(n, dtype=np.int64)
    stack = [(np.arange(n, dtype=np.int64), 0, n_cells)]
    while stack:
        idx, first_cell, cells = stack.pop()
        if cells == 1:
            labels[idx] = first_cell
            continue
        left_cells = cells // 2
        n_left = int(round(len(idx) * left_cells / cells))
        v = _principal_direction(X, idx, device)
        proj = _project(X, idx, v, device)
        order = torch.argsort(proj).numpy()
        stack.append((idx[order[:n_left]], first_cell, left_cells))
        stack.append((idx[order[n_left:]], first_cell + left_cells, cells - left_cells))
    return labels


def graph_partition_labels(neighbors, n_cells, n_iter=10, device=None, chunk=4_000_000):
    """Balanced cells over a graph given a (n, k) neighbor table (-1 = missing).

    Two stages:
    1. Label propagation to find communities — each node repeatedly takes the
       mode of its neighbors' (plus its own) labels. Fully vectorized via
       row-wise torch.mode over the neighbor table, chunked so arbitrarily
       large graphs fit in memory.
    2. Balanced chop — nodes are grouped by community (largest communities
       first) and the grouped ordering is cut into n_cells equal contiguous
       chunks. Cell sizes differ by at most 1 by construction; only the few
       communities straddling a chunk boundary get split.

    (A BFS-style ordering like reverse Cuthill-McKee is not used because
    small-diameter graphs — e.g. social graphs — interleave communities
    across its level sets, destroying locality.)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neighbors = np.asarray(neighbors)
    n, k = neighbors.shape
    self_col = np.arange(n, dtype=np.int64)
    nb = np.where(neighbors >= 0, neighbors, self_col[:, None]).astype(np.int64)

    labels = np.arange(n, dtype=np.int64)
    for it in range(n_iter):
        new = np.empty(n, dtype=np.int64)
        lab_t = torch.from_numpy(labels).to(device)
        for i in range(0, n, chunk):
            nb_t = torch.from_numpy(nb[i:i + chunk]).to(device)
            L = torch.cat([lab_t[nb_t], lab_t[i:i + nb_t.size(0)].unsqueeze(1)], dim=1)
            new[i:i + nb_t.size(0)] = torch.mode(L, dim=1).values.cpu().numpy()
        converged = (new == labels).all()
        labels = new
        if converged:
            print(f'label propagation converged after {it + 1} iterations')
            break

    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    comm_rank = np.empty(len(counts), dtype=np.int64)
    comm_rank[np.argsort(-counts, kind='stable')] = np.arange(len(counts))
    perm = np.argsort(comm_rank[inv], kind='stable')
    out = np.empty(n, dtype=np.int64)
    out[perm] = np.arange(n, dtype=np.int64) * n_cells // n
    return out


def topk_neighbors_from_csr(A, k):
    """Read a (n, k) neighbor table straight off a sparse adjacency matrix.

    Rows are each node's neighbors sorted by descending edge weight, padded
    with -1 past the node's degree. Fully vectorized (lexsort over the nnz
    entries) — no per-row Python loop.
    """
    A = A.tocsr()
    n = A.shape[0]
    row = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr))
    order = np.lexsort((-A.data, row))
    rank = np.arange(A.nnz, dtype=np.int64) - A.indptr[row]
    out = np.full((n, k), -1, dtype=np.int64)
    keep = rank < k
    out[row[keep], rank[keep]] = A.indices[order][keep]
    return out
