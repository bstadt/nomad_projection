import gc
import time
import torch
import numpy as np
from tqdm import tqdm

class LSHKMeans:
    def __init__(self, n_clusters, max_iter=1000, tol=1e-4, lsh_bits=16, chunk_size=2000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.lsh_bits = lsh_bits
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('LSHMeans Device: ', self.device)

    def _lsh_hash(self, X, random_vectors):
        projections = torch.matmul(X, random_vectors)
        return (projections > 0).int()

    def _initialize_centroids_lsh(self, X):
        random_vectors = torch.randn(X.size(1), self.lsh_bits, dtype=X.dtype).to(self.device)
        hashes = self._lsh_hash(X, random_vectors)
        unique_hashes, counts = torch.unique(hashes, return_counts=True, dim=0)
        sorted_indices = torch.argsort(counts, descending=True)
        unique_hashes = unique_hashes[sorted_indices]
        centroids = []

        for i in range(min(self.n_clusters, unique_hashes.size(0))):
            mask = torch.all(hashes == unique_hashes[i], dim=1)
            centroid = X[mask].mean(dim=0)
            centroids.append(centroid)

        centroids = torch.stack(centroids)
        if centroids.size(0) < self.n_clusters:
            additional_centroids = X[torch.randperm(X.size(0))[:self.n_clusters - centroids.size(0)]]
            centroids = torch.cat((centroids, additional_centroids))

        return centroids

    def fit(self, X):
        X = X.to(self.device)
        centroids = self._initialize_centroids_lsh(X)
        chunk_size = self.chunk_size
        for _ in tqdm(range(self.max_iter)):
            labels = torch.empty(X.size(0), dtype=torch.long, device=self.device)
            min_distances = torch.full((X.size(0),), float('inf'), device=self.device, dtype=X.dtype)

            for i in range(0, centroids.size(0), chunk_size):
                centroid_chunk = centroids[i:i+chunk_size]
                chunk_distances = torch.cdist(X, centroid_chunk)
                
                chunk_min_distances, chunk_labels = torch.min(chunk_distances, dim=1)
                
                update_mask = chunk_min_distances < min_distances
                min_distances[update_mask] = chunk_min_distances[update_mask]
                labels[update_mask] = chunk_labels[update_mask] + i

            del min_distances, chunk_distances

            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(self.n_clusters)])
            
            if torch.all(torch.abs(new_centroids - centroids) < self.tol):
                break
            
            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

    def predict(self, X):
        X = X.to(self.device)
        labels = torch.empty(X.size(0), dtype=torch.long, device=self.device)
        min_distances = torch.full((X.size(0),), float('inf'), device=self.device, dtype=X.dtype)
        chunk_size = self.chunk_size

        for i in range(0, self.centroids.size(0), chunk_size):
            centroid_chunk = self.centroids[i:i+chunk_size]
            chunk_distances = torch.cdist(X, centroid_chunk)
            
            chunk_min_distances, chunk_labels = torch.min(chunk_distances, dim=1)
            
            update_mask = chunk_min_distances < min_distances
            min_distances[update_mask] = chunk_min_distances[update_mask]
            labels[update_mask] = chunk_labels[update_mask] + i

        return labels

class PartitionANN:
    def __init__(self, X, n_neighbors, n_cells, cluster_subset_size, cluster_chunk_size=None, method='balanced'):
        if cluster_chunk_size is None:
            cluster_chunk_size = n_cells
        total_s = time.time()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)

        # cluster
        s = time.time()
        if method == 'balanced':
            print('Balanced partition (recursive principal-direction bisection)')
            from nomad_projection.partition import balanced_partition_labels
            cluster_ids = balanced_partition_labels(X, n_cells, device=self.device)
            self.n_cells = n_cells
            self.labels = cluster_ids
        elif X.shape[0] < 5000000:
            print('Clustering on All Data')
            X = torch.from_numpy(X)
            print('clustering')
            self.n_cells = n_cells
            self._cluster_obj = LSHKMeans(n_clusters=self.n_cells, chunk_size=cluster_chunk_size)
            self._cluster_obj.fit(X)
            cluster_ids = self._cluster_obj.labels.cpu().numpy()
            self.labels = cluster_ids
            X = X.cpu().numpy()
        else:
            print('Clustering on Subset')
            indices = np.random.permutation(X.shape[0])[:cluster_subset_size]
            sub_X = torch.from_numpy(X[indices])
            self.n_cells = n_cells
            self._cluster_obj = LSHKMeans(n_clusters=self.n_cells, chunk_size=cluster_chunk_size)
            self._cluster_obj.fit(sub_X)

            cluster_ids = []
            for i in range(0, X.shape[0], cluster_subset_size):
                chunk = torch.from_numpy(X[i:i+cluster_subset_size])
                cluster_ids.append(self._cluster_obj.predict(chunk))
            cluster_ids = torch.cat(cluster_ids, dim=0).cpu().numpy()
            self.labels = cluster_ids
            del self._cluster_obj
            
        e = time.time()
        print('clustering took: ', e-s)

        # get knns
        print('computing neighbors')
        s = time.time()
        
        X_ids = np.arange(X.shape[0])
        '''
        clusterwise_X_ids = [X_ids[cluster_ids==cluster_id]
                             for cluster_id in range(np.max(cluster_ids) + 1)]
        self.clusterwise_X_ids = clusterwise_X_ids
        '''

        self.clusterwise_X_ids = []
        clusterwise_topk = {}
        #for cluster_id, cur_X_ids in enumerate(tqdm(clusterwise_X_ids)):
        for cluster_id in tqdm(range(np.max(cluster_ids)+1)):
            cur_X_ids = X_ids[cluster_ids==cluster_id]
            self.clusterwise_X_ids.append(cur_X_ids)
            x = torch.from_numpy(X[cur_X_ids]).to(self.device)
            chunk_size = 1000
            num_vectors = x.shape[0]
            topk_indices = []
            
            for i in range(0, num_vectors, chunk_size):
                chunk = x[i:i+chunk_size]
                chunk_similarities = chunk @ x.T
                chunk_topk = torch.topk(chunk_similarities, n_neighbors).indices.cpu()
                topk_indices.append(chunk_topk)
            
            clusterwise_topk[cluster_id] = torch.cat(topk_indices, dim=0).numpy()

        e = time.time()
        self._clusterwise_topk = clusterwise_topk
        print('neighborfinding took: ', e-s)
        
        total_e = time.time()
        print('total time: ', total_e-total_s)



    def knn(self, target, k):
        return self._topk[target, :k]


class GraphKNN:
    """Per-cell kNN tables read directly off a (n, k) neighbor table.

    Replaces PartitionANN's dense per-cell similarity search when the data is
    a graph whose neighbor lists are already known (e.g. an adjacency matrix).
    Each node keeps its neighbors that landed in the same cell, cyclically
    repeated up to n_neighbors; a node with no same-cell neighbors gets itself
    (a self-positive is a no-op force). Output format matches PartitionANN:
    _clusterwise_topk[c] is (m, n_neighbors+1) of cell-local indices with
    column 0 = self.
    """

    def __init__(self, neighbors, labels, n_neighbors):
        s = time.time()
        neighbors = np.asarray(neighbors)
        labels = np.asarray(labels, dtype=np.int64)
        n, k_in = neighbors.shape
        if n_neighbors > k_in:
            print(f'GraphKNN: n_neighbors={n_neighbors} > neighbor table width '
                  f'{k_in}; entries will repeat cyclically')
        k = n_neighbors
        n_cells = int(labels.max()) + 1
        self.labels = labels

        local_idx = np.empty(n, dtype=np.int64)
        self.clusterwise_X_ids = []
        for c in range(n_cells):
            ids = np.flatnonzero(labels == c)
            self.clusterwise_X_ids.append(ids)
            local_idx[ids] = np.arange(len(ids), dtype=np.int64)

        self._clusterwise_topk = {}
        for c, ids in enumerate(tqdm(self.clusterwise_X_ids)):
            nb = neighbors[ids]  # (m, k_in) global ids, -1 padded
            valid = nb >= 0
            same = np.zeros_like(valid)
            same[valid] = labels[nb[valid]] == c
            # stable-sort same-cell neighbors to the front of each row
            order = np.argsort(~same, axis=1, kind='stable')
            nb_sorted = np.take_along_axis(nb, order, axis=1)
            counts = same.sum(axis=1)
            cols = np.arange(k, dtype=np.int64)[None, :]
            src = np.where(counts[:, None] > 0,
                           cols % np.maximum(counts[:, None], 1), 0)
            gathered = np.take_along_axis(nb_sorted, src, axis=1)
            gathered = np.where(counts[:, None] > 0, gathered, ids[:, None])
            out = np.empty((len(ids), k + 1), dtype=np.int64)
            out[:, 0] = np.arange(len(ids), dtype=np.int64)
            out[:, 1:] = local_idx[gathered]
            self._clusterwise_topk[c] = out
        print('graph knn extraction took: ', time.time() - s)