import os
import gc
import time
import torch
import numpy as np
from datetime import timedelta
from contextlib import nullcontext
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from nomad_projection.neighbors import PartitionANN, GraphKNN
from nomad_projection.partition import (balanced_partition_labels, graph_partition_labels,
                                        graph_partition_labels_metis)

def get_gpu_memory_usage():
    import gc
    total_memory = 0
    tensor_shapes = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    memory = obj.element_size() * obj.nelement()
                    total_memory += memory
                    tensor_shapes.append((obj.shape, memory))
        except:
            pass
    
    print(f"GPU memory usage: {total_memory / 1024**2:.2f} MB")
    print("Tensor shapes:")
    for shape, mem in tensor_shapes:
        print(f"  Shape: {shape}, Memory: {mem / 1024**2:.2f} MB")


def _graph_labels(neighbors, n_cells, method):
    """Cells for graph mode, honouring `graph_partition` and pymetis availability."""
    if method not in ('auto', 'metis', 'chop'):
        raise ValueError(f"graph_partition must be 'auto', 'metis' or 'chop', got {method!r}")
    if method == 'chop':
        return graph_partition_labels(neighbors, n_cells)
    try:
        return graph_partition_labels_metis(neighbors, n_cells)
    except ImportError:
        if method == 'metis':
            raise ImportError(
                "graph_partition='metis' requires pymetis (pip install pymetis)") from None
        print('pymetis not installed; falling back to the label-propagation partition. '
              'pip install pymetis for a min-cut partition, which keeps substantially '
              'more edges inside cells.')
        return graph_partition_labels(neighbors, n_cells)


class NomadProjection:
    def __init__(self):
        self._knn = None
        self._model = None
        self._optim = None

        self.world_size = torch.cuda.device_count()
                    
        self.gpu_cluster_map = {}
        self._model_sizes = []
        self.cluster_assignments = None


    def _autocast_context(self):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            if props.major >= 8:  # Ampere GPUs (compute capability >= 8.0) support BF16
                print(f"Using autocast for BF16 on GPU: {props.name}")
                return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            else:
                print(f"GPU {props.name} does not support BF16. Running without autocast.")
        else:
            print("CUDA is not available. Running without autocast.")

        return nullcontext() 


    @torch.compile
    def _step(self,
              model_idxs,
              knn,
              rank,
              batch_size,
              n_neighbors,
              n_noise,
              pos_weight,
              neg_weight,
              do_gather,
              context,
              cell_of_row=None,
              mu_cell_ids=None,
              cell_repulsion_weight=1.0,
              local_cell_ids=None,
              cell_separation_weight=0.0):

            x = torch.cat([self._model[model_num] for model_num in model_idxs], axis=0)
            mus = torch.stack([self._model[model_num].mean(dim=0) for model_num in model_idxs], axis=0)

            n_cells_total = len(self._model)
            if do_gather:
                mu_container = []
                for i in range(self.world_size):
                    models_on_rank = len([ k for k, gpu in self.gpu_cluster_map.items() if gpu == i])
                    mu_container.append(torch.zeros(models_on_rank, mus.size(1), device=f'cuda:{rank}'))
                torch.distributed.all_gather(mu_container, mus)
                # Every cell's centroid. This previously dropped all cells living
                # on the local rank, so with n_cells > world_size a point was never
                # repelled from the other cells sharing its GPU (2 of 16 at
                # n_cells=16 on 8 GPUs). Rows are rank-major; mu_cell_ids carries
                # the matching global cell id so a point can mask its own.
                all_mus = torch.cat(mu_container, dim=0)
            else:
                all_mus = mus

            n = x.size(0) 
            with context:
                target_idxs = torch.randint(low=0, high=n, size=(batch_size,), device=f'cuda:{rank}')
                noise_idxs = torch.randint(low=0, high=batch_size*n_neighbors, size=(batch_size, n_noise), device=f'cuda:{rank}')

                cur_batch_size = target_idxs.size(0)

                #Gather target and neighbor embeddings
                neighbor_idxs = knn[target_idxs, 1:n_neighbors+1]
                target_embs = x[target_idxs].reshape(cur_batch_size, 2)
                neighbor_embs = x[neighbor_idxs.reshape(cur_batch_size*n_neighbors, )]
                noise_embs = (neighbor_embs[noise_idxs]).reshape(cur_batch_size, n_noise, 2)
                
                #Compute kernel distances
                positives = ((target_embs.reshape(cur_batch_size, 1, 2) - neighbor_embs.reshape(cur_batch_size, n_neighbors, 2))**2).sum(axis=-1)
                negatives = ((target_embs.reshape(cur_batch_size, 1, 2) - noise_embs)**2).sum(axis=-1)

                poskerns = 1/(1+positives) #[Batch, Neighbors]
                negkerns_single = 1 / (1 + negatives)  # Shape: [B, L]
                negkerns = negkerns_single.sum(dim=1, keepdim=True).expand(-1, n_neighbors)  # Shape: [B, K]

                #Compute loss
                ranks = torch.arange(1, n_neighbors+ 1, dtype=torch.float32).cuda()
                exp_ranks = torch.exp(1 / ranks)
                sum_exp_ranks = exp_ranks.sum()
                pji = exp_ranks / sum_exp_ranks
                pji = pji.reshape(1, -1)

                if n_cells_total > 1 and cell_of_row is not None and mu_cell_ids is not None:
                    dist_negatives = ((target_embs.reshape(cur_batch_size, 1, 2) - all_mus)**2).sum(axis=-1)
                    dist_negkerns_single = 1 / (1 + dist_negatives)
                    # Mask each point's OWN cell rather than its whole rank, so
                    # every cell supplies negatives to every other cell.
                    own_cell = cell_of_row[target_idxs]
                    dist_negkerns_single = dist_negkerns_single.masked_fill(
                        own_cell.unsqueeze(1) == mu_cell_ids.unsqueeze(0), 0.0)
                    dist_negkerns = (cell_repulsion_weight
                                     * dist_negkerns_single.sum(dim=1, keepdim=True)
                                     ).expand(-1, n_neighbors)
                else:
                    dist_negkerns = torch.zeros_like(negkerns)

                losses = -1 * pos_weight * (torch.log(poskerns) * pji).sum(axis=-1) + neg_weight * (torch.log(poskerns + negkerns + dist_negkerns) * pji).sum(axis=-1)

                loss = losses.mean()

                # Centroid-to-centroid separation, at layout scale.
                #
                # dist_negkerns repels every POINT from centroids that all sit
                # near the origin, which is an isotropic push that cannot
                # separate anything, and with 1/(1+d^2) its force decays as
                # 1/d^3 -- negligible once the layout is wider than a few units.
                # Here the repulsion acts between the centroids themselves, and
                # mu_i is the differentiable mean of cell i's points, so the
                # gradient translates the whole cell as a body and leaves its
                # internal structure alone. Distances are measured in units of
                # the current layout radius, so the force stays meaningful as
                # the embedding grows instead of vanishing.
                if cell_separation_weight > 0 and local_cell_ids is not None:
                    sigma = x.detach().pow(2).sum(dim=1).mean().sqrt().clamp(min=1e-6)
                    dmu2 = ((mus[:, None, :] - all_mus[None, :, :].detach()) ** 2).sum(-1)
                    kmu = 1.0 / (1.0 + dmu2 / (sigma ** 2))
                    kmu = kmu.masked_fill(
                        local_cell_ids.unsqueeze(1) == mu_cell_ids.unsqueeze(0), 0.0)
                    loss = loss + cell_separation_weight * kmu.sum(dim=1).mean()
                
            loss.backward()

            self._optim.step()
            return loss.item()

    def train_on_gpu(self,
                     rank,
                     n,
                     batch_size,
                     epochs,
                     n_neighbors,
                     n_noise,
                     late_exaggeration_time,
                     late_exaggeration_scale,
                     late_exaggeration_n_noise,
                     lr_scale,
                     learning_rate_decay_start_time,
                     distributed,
                     cell_repulsion_weight='auto',
                     cell_separation_weight=0.0):

        # derive schedules
        def n_noise_schedule(t):
            if t > late_exaggeration_time:
                return late_exaggeration_n_noise
            else:
                return n_noise

        def lr_schedule(t):
            if t > learning_rate_decay_start_time:
                tprime = (t-learning_rate_decay_start_time)/(1-learning_rate_decay_start_time)
                return n*lr_scale*(1-tprime) + n*1e-8*lr_scale*(tprime)
            else:
                return n*lr_scale

        def pos_weight_schedule(t):
            if t > late_exaggeration_time:
                return late_exaggeration_scale
            else:
                return 1

        if distributed:
            # setup distributed training
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'

            torch.cuda.set_device(rank)
            dist.init_process_group(backend="nccl", rank=rank, world_size=self.world_size, timeout=timedelta(seconds=12000))

            print('Initialized Process Group: {}'.format(torch.cuda.current_device()))
            print(torch.cuda.get_device_name(torch.cuda.current_device()))

        context = self._autocast_context()

        model_idxs = [i for i in range(len(self._model)) if self.gpu_cluster_map[i] == rank]
        local_knn = []
        cell_rows = []
        offset = 0
        for i in model_idxs:
            local_knn.append(torch.tensor(self._knn[i] + offset))
            cell_rows.append(torch.full((self._knn[i].shape[0],), i, dtype=torch.long))
            offset += self._knn[i].shape[0]
        local_knn = torch.cat(local_knn, axis=0).to(f'cuda:{rank}')
        # Global cell id per row of x, so _step can mask a point's own cell.
        cell_of_row = torch.cat(cell_rows, axis=0).to(f'cuda:{rank}')

        n_cells_total = len(self._model)
        # Global cell id for each row of torch.cat(mu_container): rank-major,
        # matching how all_gather lays out the per-rank centroid blocks.
        local_cell_ids = torch.tensor(model_idxs, dtype=torch.long,
                                      device=f'cuda:{rank}')
        mu_cell_ids = torch.tensor(
            [c for i in range(self.world_size) for c in range(n_cells_total)
             if self.gpu_cluster_map[c] == i] if distributed else list(model_idxs),
            dtype=torch.long, device=f'cuda:{rank}')

        n_neighbors = torch.tensor(n_neighbors, device=f'cuda:{rank}')
        # _step already returns loss.item(), so accumulating the trajectory is free.
        # It is otherwise unrecoverable: training runs in spawned processes, so a
        # caller can only scrape stdout, and only on even epochs.
        loss_log = []
        for epoch in range(epochs):

            epoch_loss_sum, epoch_steps = 0.0, 0
            t = epoch/epochs
            cur_n_noise = n_noise_schedule(t)
            cur_pos_weight = pos_weight_schedule(t)
            cur_lr = lr_schedule(t)
            # 'auto' puts one centroid on the same footing as one random
            # negative: the centroid sum has n_cells-1 terms against n_noise,
            # so unweighted it is ~1% of the negative mass and only acts within
            # ~1 unit of a centroid -- it punches a hole rather than separating
            # cells. See examples/modal_stlm_xyra_90m.py::force_balance.
            if cell_repulsion_weight == 'auto':
                cur_cell_w = cur_n_noise / max(n_cells_total - 1, 1)
            else:
                cur_cell_w = float(cell_repulsion_weight)
            for step in range(n//(batch_size * self.world_size)):
                self._optim.zero_grad()
                loss = self._step(model_idxs=model_idxs,
                                  knn=local_knn,
                                  rank=rank,
                                  batch_size=batch_size,
                                  n_neighbors=n_neighbors,
                                  n_noise=cur_n_noise,
                                  pos_weight=cur_pos_weight,
                                  do_gather=distributed,
                                  neg_weight=1,
                                  context=context,
                                  cell_of_row=cell_of_row,
                                  mu_cell_ids=mu_cell_ids,
                                  cell_repulsion_weight=cur_cell_w,
                                  local_cell_ids=local_cell_ids,
                                  cell_separation_weight=cell_separation_weight)

                epoch_loss_sum += loss
                epoch_steps += 1

                if not epoch % 2 and not rank:
                    print('t: {:.4f}'.format(t),
                          '\tdevice:{}'.format(rank),
                          '\tloss: {:.4f}'.format(loss),
                          '\tcur_lr: {}'.format(cur_lr),
                          '\tcell_w: {:.2f}'.format(cur_cell_w))

                # Update learning rate and momentum
                for param_group in self._optim.param_groups:
                    param_group['lr'] = cur_lr

            if not rank and epoch_steps:
                loss_log.append({'epoch': epoch, 't': t,
                                 'loss': epoch_loss_sum / epoch_steps,
                                 'pos_weight': cur_pos_weight, 'lr': cur_lr})

        loss_path = os.environ.get('NOMAD_LOSS_PATH')
        if not rank and loss_path:
            import json
            with open(loss_path, 'w') as f:
                json.dump(loss_log, f)
            print(f'wrote {len(loss_log)} epoch losses to {loss_path}', flush=True)


    def fit_transform(self,
            X=None,
            batch_size=None,
            epochs=None,
            neighbors=None,
            partition='balanced',
            graph_partition='auto',
            cell_labels=None,
            n_cells=5,
            cluster_chunk_size=2000,
            n_neighbors=8,
            n_noise=10000,
            late_exaggeration_time=0.6,
            late_exaggeration_scale=1,
            late_exaggeration_n_noise=None,
            momentum=0.8,
            learning_rate_decay_start_time=0.3,
            lr_scale=0.1,
            cluster_subset_size=5000000,
            cell_repulsion_weight=1.0,
            cell_separation_weight=0.0,
            debug_plot=False,
           ):
        """Project to 2D.

        Two input modes:
        - Feature mode: pass X (n, d). Cells come from `partition`
          ('balanced' = exactly even recursive-bisection cells, 'lsh' =
          upstream LSH k-means), and kNN is computed per cell.
        - Graph mode: pass neighbors (n, k) of global neighbor ids per node
          (-1 = missing; e.g. from partition.topk_neighbors_from_csr). The
          kNN search is skipped entirely — neighbor tables are read off the
          graph. Cells come from X (balanced partition) when X is also
          given, otherwise from `graph_partition`. X, when present alongside
          neighbors, is used only for partitioning and PCA init.

        cell_separation_weight adds a repulsion between the cell CENTROIDS,
        measured in units of the current layout radius. cell_repulsion_weight
        cannot separate cells at any magnitude: it pushes every point away from
        centroids that all sit near the origin, which is an isotropic force, and
        1/(1+d^2) decays as 1/d^3 so it is ~1e-6 once the layout is wide. This
        term instead acts centroid-to-centroid; since mu_i is the differentiable
        mean of cell i's points, its gradient translates the cell as a body and
        leaves the within-cell layout intact. 0 disables it.

        cell_repulsion_weight scales the repulsion of every point from the
        other cells' centroids. Cells are otherwise coupled by nothing else, so
        this is the only force that can keep them from being laid out on top of
        one another. Unweighted the term contributes n_cells-1 summands against
        n_noise (2000), i.e. ~1% of the negative mass, and since the kernel is
        1/(1+d^2) it only bites within ~1 unit -- measured on a 97.8M-node
        graph it carved a hole at each centroid while the 16 cells stayed fully
        superimposed (median centroid separation 15 against within-cell spread
        119).

        Raising it does NOT fix that, and measurably makes it worse. Every
        centroid already sits near the origin, so "repel from the other cells'
        centroids" is, for every point of every cell, the same isotropic
        outward push; it drives each cell toward a symmetric annulus, whose
        centroid is the origin. Measured at n_cells=16, 97.8M nodes, 150
        epochs, separation ratio = median centroid distance / median within-cell
        spread:

            weight   1     10    n_noise/(n_cells-1)=133
            sep      0.105 0.104 0.074
            centroid 7.36  7.21  5.16   (spread flat at ~70)

        so the knob is kept for measurement but defaults to 1.0. Separating the
        cells needs the symmetry broken before training -- initialising each
        cell around its own position in a coarse layout of the cell graph --
        not a bigger coefficient on a centrally-symmetric force.

        cell_labels, when given, overrides both partitioners with a
        precomputed (n,) cell assignment and sets n_cells from it. Partitioning
        a large graph (METIS on 10^8 nodes) is a long CPU job with no use for a
        GPU, so it is worth running separately and passing the result in rather
        than paying for idle accelerators while it runs.

        graph_partition selects how graph-mode cells are formed:
          'auto'   metis when pymetis is installed, else 'chop' (default)
          'metis'  balanced minimum k-way cut; requires pymetis
          'chop'   label propagation then an equal contiguous chop

        Cells matter more than they look: GraphKNN keeps only the neighbors
        that land in the same cell, so a partition that cuts many edges
        silently deletes most of the attractive force. On a 630k-node repost
        graph, metis kept 64.7% of edges against chop's 29.3%, and left 0.1%
        of nodes with no same-cell neighbor against 34.8%.
        """
        # late_exaggeration_time is a FRACTION of training: t = epoch/epochs, so it only
        # has an effect below 1.0. It shipped defaulting to 1.1, which meant the whole
        # late phase — both the attraction boost and its noise count — was unreachable.
        if late_exaggeration_time >= 1.0 and (late_exaggeration_scale != 1
                                              or late_exaggeration_n_noise is not None):
            print(f'WARNING: late_exaggeration_time={late_exaggeration_time} >= 1.0, but t '
                  f'never exceeds 1.0 — the late phase will not run and '
                  f'late_exaggeration_scale/n_noise are ignored. Pass a fraction (e.g. 0.6).')
        if late_exaggeration_n_noise is None:
            late_exaggeration_n_noise = n_noise

        if X is None and neighbors is None:
            raise ValueError('pass X (feature mode) and/or neighbors (graph mode)')
        if batch_size is None or epochs is None:
            raise ValueError('batch_size and epochs are required')

        #Setup Params
        n = X.shape[0] if X is not None else neighbors.shape[0]

        if neighbors is not None:
            neighbors = np.asarray(neighbors)
            if X is not None and X.shape[0] != neighbors.shape[0]:
                raise ValueError('X and neighbors disagree on n')
            if cell_labels is not None:
                labels = np.asarray(cell_labels, dtype=np.int64)
                if labels.shape != (n,):
                    raise ValueError(
                        f'cell_labels must be shape ({n},), got {labels.shape}')
                n_cells = int(labels.max()) + 1
                print(f'using precomputed cell_labels: {n_cells} cells, sizes '
                      f'{np.bincount(labels).min()}..{np.bincount(labels).max()}')
            elif X is not None:
                labels = balanced_partition_labels(X, n_cells)
            else:
                labels = _graph_labels(neighbors, n_cells, graph_partition)
            self._knn_obj = GraphKNN(neighbors, labels, n_neighbors)
        else:
            self._knn_obj = PartitionANN(X, n_neighbors+1, n_cells, cluster_subset_size, cluster_chunk_size, method=partition)
        self._knn = self._knn_obj._clusterwise_topk
        self._clusterwise_X_ids = self._knn_obj.clusterwise_X_ids
        self.cluster_assignments = self._knn_obj.labels
        # Drop the clusterer before mp.spawn pickles self: it can hold CUDA tensors,
        # which would be needlessly IPC-shared with every worker process.
        self._knn_obj = None
        torch.cuda.empty_cache()

        init_lr = n * lr_scale

        if X is not None:
            # Convert input data to a PyTorch tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)

            # Perform PCA using PyTorch
            U, S, V = torch.pca_lowrank(X_tensor, q=2)
            init = U[:, :2].numpy()  # Get the first two principal components
            del X_tensor
        else:
            # No features to seed from — random init at the usual scale
            init = np.random.randn(n, 2).astype(np.float32)
        init /= (init[:, 0].std()) / 1e-4

        num_clusters = len(self._knn)
        cluster_ids = np.arange(num_clusters)

        # A rank with zero clusters crashes in _step (torch.cat of an empty list),
        # so never use more GPUs than cells.
        if self.world_size > num_clusters:
            print(f'n_cells={num_clusters} < {self.world_size} visible GPUs; '
                  f'training on {num_clusters} GPUs')
            self.world_size = num_clusters
        if self.world_size > 1 and num_clusters % self.world_size != 0:
            print(f'WARNING: n_cells={num_clusters} is not divisible by world_size='
                  f'{self.world_size}; ranks will hold uneven cluster counts. '
                  f'Prefer n_cells as a multiple of the GPU count.')

        # Initialize the embedding models on each GPU
        self._model = torch.nn.ParameterList()
        for cluster_id in cluster_ids:
            gpu = cluster_id % self.world_size
            init_idxs = self._clusterwise_X_ids[cluster_id]
            cluster_init_data = torch.tensor(init[init_idxs], dtype=torch.float32, device=f'cuda:{gpu}')
            cluster_model = torch.nn.Parameter(cluster_init_data)
            self._model_sizes.append(cluster_init_data.size(0))

            self._model.append(cluster_model)
            self.gpu_cluster_map[cluster_id] = gpu
        del self._clusterwise_X_ids

        self._optim = torch.optim.SGD([
            {'params': self._model.parameters()},
        ], lr=init_lr, momentum=momentum)

        distributed = self.world_size > 1
        if distributed:
            # Launch parallel training on all GPUs
            mp.spawn(self.train_on_gpu,
                    nprocs=self.world_size,
                    args=(n,
                        batch_size,
                        epochs,
                        n_neighbors,
                        n_noise,
                        late_exaggeration_time,
                        late_exaggeration_scale,
                        late_exaggeration_n_noise,
                        lr_scale,
                        learning_rate_decay_start_time,
                        distributed,
                        cell_repulsion_weight,
                        cell_separation_weight),
                    join=True)
        else:
            self.train_on_gpu(rank=0,
                              n=n,
                              batch_size=batch_size,
                              epochs=epochs,
                              n_neighbors=n_neighbors,
                              n_noise=n_noise,
                              late_exaggeration_time=late_exaggeration_time,
                              late_exaggeration_scale=late_exaggeration_scale,
                              late_exaggeration_n_noise=late_exaggeration_n_noise,
                              lr_scale=lr_scale,
                              learning_rate_decay_start_time=learning_rate_decay_start_time,
                              distributed=distributed,
                              cell_repulsion_weight=cell_repulsion_weight,
                              cell_separation_weight=cell_separation_weight)

        # Collect final embeddings in the order of input data
        final_embeddings = torch.zeros((len(self.cluster_assignments), 2), dtype=torch.float32)
        
        for cluster_id in range(len(self._model)):
            cluster_indices = torch.where(torch.tensor(self.cluster_assignments) == cluster_id)[0]
            cluster_embeddings = self._model[cluster_id].data.detach().cpu()
            final_embeddings[cluster_indices] = cluster_embeddings

        # Convert to numpy array before returning
        return final_embeddings.numpy()