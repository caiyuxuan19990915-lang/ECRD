import math
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss


class FactualCascadeReasoning(nn.Module):
    """Encode the behavior chain with a shared transition (Section IV-A)."""

    def __init__(self, layer_nums, hidden_dim, dropout, behaviors):
        super().__init__()
        self.behaviors = behaviors
        self.layer_nums = layer_nums
        self.behavior_gnn_layers = nn.ModuleDict({
            behavior: nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim, add_self_loops=False, bias=False)
                for _ in range(depth)
            ])
            for behavior, depth in zip(behaviors, layer_nums)
        })
        self.transition = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, init_embeddings, edge_index_dict, device, degree_by_behavior=None):
        current_embeddings = init_embeddings
        all_behavior_embeddings = []

        for index, behavior in enumerate(self.behaviors):
            edges = edge_index_dict[behavior].to(device)
            degree = None if degree_by_behavior is None else degree_by_behavior[behavior].to(device)
            x = current_embeddings
            aggregated = x
            for gnn_layer in self.behavior_gnn_layers[behavior]:
                # Degree-normalized propagation (Eq. 1).
                x = gnn_layer(x=x, edge_index=edges, degree=degree)
                x = self.dropout(x)
                aggregated = aggregated + x

            # Sum layers 0 through L and transfer to the next behavior (Eq. 2).
            all_behavior_embeddings.append(aggregated)
            if index + 1 < len(self.behaviors):
                current_embeddings = self.transition(aggregated)

        return all_behavior_embeddings, all_behavior_embeddings[-1]


class CounterfactualAttention(nn.Module):
    """Aggregate candidate embeddings for CRD (Section IV-C.3)."""

    def __init__(self, emb_dim):
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.scale = math.sqrt(emb_dim)

    def forward(self, factual, candidates, valid=None):
        query = self.q_proj(factual).unsqueeze(1)
        keys = self.k_proj(candidates)
        values = self.v_proj(candidates)
        scores = (query @ keys.transpose(-1, -2)).squeeze(1) / self.scale
        if valid is not None:
            scores = scores.masked_fill(~valid, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return (weights.unsqueeze(1) @ values).squeeze(1)


class _NeighborIndex:
    """Index a static edge list by source node, retaining edge multiplicity."""

    def __init__(self, edge_index, num_nodes):
        source, target = edge_index.detach().cpu().numpy()
        order = np.lexsort((target, source))
        self.targets = target[order]
        self.offsets = np.concatenate(([0], np.bincount(source, minlength=num_nodes).cumsum()))

    def neighbors_of(self, node):
        return self.targets[self.offsets[node]:self.offsets[node + 1]]

    def has_edge(self, source, target):
        neighbors = self.neighbors_of(source)
        position = np.searchsorted(neighbors, target)
        return position < len(neighbors) and neighbors[position] == target

    def edge_count(self, source, target):
        neighbors = self.neighbors_of(source)
        return int(np.searchsorted(neighbors, target, side='right') - np.searchsorted(neighbors, target))

    def gather(self, nodes):
        starts = self.offsets[nodes]
        counts = self.offsets[nodes + 1] - starts
        ends = counts.cumsum()
        positions = np.arange(counts.sum()) + np.repeat(starts - ends + counts, counts)
        return self.targets[positions], counts


class DualCounterfactualReasoning(nn.Module):
    """Construct structural and semantic counterfactuals (Section IV-B)."""

    def __init__(self, args, cascade_encoder):
        super().__init__()
        self.aux_behaviors = args.behaviors[:-1]
        self.cascade_encoder = cascade_encoder
        self.num_hops = sum(cascade_encoder.layer_nums)
        self.k_semantic = getattr(args, 'k_semantic', 2)
        self.gamma = getattr(args, 'gamma', 0.3)
        if self.k_semantic < 0:
            raise ValueError('k_semantic must be nonnegative.')
        self.struct_attention = CounterfactualAttention(args.embedding_size)
        self.sem_attention = CounterfactualAttention(args.embedding_size)
        self.cached_base_candidates = None
        self.cached_out_candidates = None
        self._indexed_dataset = None

    def _index(self, dataset):
        if dataset is not self._indexed_dataset:
            self.num_nodes = dataset.item_offset + dataset.item_count + 1
            self._union = _NeighborIndex(dataset.all_edge_index, self.num_nodes)
            self._behavior_graphs = {
                behavior: _NeighborIndex(edges, self.num_nodes)
                for behavior, edges in dataset.edge_index.items()
            }
            self._indexed_dataset = dataset

    def _observed(self, user, item_node):
        return tuple(behavior for behavior in self.aux_behaviors
                     if self._behavior_graphs[behavior].has_edge(user, item_node))

    def auxiliary_catalog_pairs(self, users, dataset):
        """Return the batch rows and catalog columns with observed auxiliary edges."""
        self._index(dataset)
        if not self.aux_behaviors:
            empty = users.new_empty((0,))
            return empty, empty

        user_ids = np.asarray(users.tolist(), dtype=np.int64)
        batch_rows = np.arange(user_ids.size)
        width = dataset.item_count + 1
        pair_keys = []
        for behavior in self.aux_behaviors:
            neighbors, counts = self._behavior_graphs[behavior].gather(user_ids)
            items = neighbors - dataset.item_offset
            in_catalog = (items > 0) & (items < width)
            keys = np.repeat(batch_rows, counts) * width + items
            pair_keys.append(keys[in_catalog])

        keys = torch.as_tensor(np.unique(np.concatenate(pair_keys)), device=users.device)
        return keys // width, keys % width

    def _structural_edits(self, observed, item_node):
        if len(observed) >= 2:
            return {behavior: (item_node, None) for behavior in observed}
        if len(observed) == 1:
            return {behavior: (None, item_node) for behavior in self.aux_behaviors
                    if behavior not in observed}
        return {}

    def _semantic_items(self, users, catalog_items, item_nodes, E_base, E_factual, dataset):
        # Apply both eligibility filters before the Eq. (4) ranking.
        pool = dataset.popular_items_tensor.to(item_nodes.device)
        count = min(self.k_semantic, pool.numel())
        if count == 0:
            return item_nodes.new_empty((len(users), 0))

        if self.cached_base_candidates is None or self.cached_out_candidates is None:
            pool_nodes = pool + dataset.item_offset
            base_pool = E_base[pool_nodes]
            factual_pool = E_factual[pool_nodes]
        else:
            base_pool = self.cached_base_candidates
            factual_pool = self.cached_out_candidates

        base_items = F.normalize(E_base[item_nodes], dim=-1)
        base_pool = F.normalize(base_pool, dim=-1)
        eligible = base_items @ base_pool.T <= self.gamma
        positions = {item: col for col, item in enumerate(dataset.popular_items)}
        history = [dataset.train_behavior_dict[behavior] for behavior in dataset.edge_index]
        offset = dataset.item_offset
        blocked = []
        for row, (user, item) in enumerate(zip(users, catalog_items)):
            seen = {item}
            seen.update(self._union.neighbors_of(user) - offset)
            for behavior_history in history:
                seen.update(behavior_history.get(str(user)) or ())
            blocked.extend((row, positions[seen_item]) for seen_item in seen if seen_item in positions)
        if blocked:
            rows, cols = zip(*blocked)
            eligible[torch.tensor(rows, device=pool.device), torch.tensor(cols, device=pool.device)] = False

        scores = E_factual[item_nodes] @ factual_pool.T
        indices = scores.masked_fill(~eligible, float('-inf')).topk(count, dim=-1).indices
        return torch.where(eligible.gather(1, indices), pool[indices], 0)

    def _local_nodes(self, roots):
        frontier = np.unique(roots)
        visited = np.zeros(self.num_nodes, dtype=bool)
        visited[frontier] = True
        for _ in range(self.num_hops):
            neighbors, _ = self._union.gather(frontier)
            discovered = np.zeros_like(visited)
            discovered[neighbors] = True
            frontier = np.flatnonzero(discovered & ~visited)
            if not frontier.size:
                break
            visited[frontier] = True
        return np.flatnonzero(visited)

    def _local_candidate(self, user, item_node, candidate_item, edits, E_base, dataset, device):
        # The edit endpoints are roots of the h-hop neighborhood.
        nodes = self._local_nodes([user, item_node, candidate_item])
        local_id = np.full(self.num_nodes, -1, dtype=np.int64)
        local_id[nodes] = np.arange(len(nodes))
        node_ids = torch.from_numpy(nodes)
        local_user = local_id[user]
        local_graphs = {}
        local_degrees = {}

        for behavior, adjacency in self._behavior_graphs.items():
            target, counts = adjacency.gather(nodes)
            target = local_id[target]
            source = np.repeat(np.arange(len(nodes)), counts)
            kept = target >= 0
            source, target = source[kept], target[kept]
            # Use global degrees, with the same changes as the local edge list.
            degree = dataset.degree_by_behavior[behavior][node_ids]
            if behavior in edits:
                old, new = edits[behavior]
                copies = 1
                if old is not None:
                    local_old = local_id[old]
                    forward = (source == local_user) & (target == local_old)
                    reverse = (source == local_old) & (target == local_user)
                    copies = int(forward.sum())
                    degree[local_old] -= copies
                    degree[local_user] -= int(reverse.sum())
                    keep = ~(forward | reverse)
                    source, target = source[keep], target[keep]
                if new is not None:
                    local_new = local_id[new]
                    source = np.concatenate((source, np.tile([local_user, local_new], copies)))
                    target = np.concatenate((target, np.tile([local_new, local_user], copies)))
                    degree[local_user] += copies
                    degree[local_new] += copies

            local_graphs[behavior] = torch.from_numpy(np.stack((source, target)))
            local_degrees[behavior] = degree

        local_input = E_base[node_ids.to(device)]
        _, encoded = self.cascade_encoder(local_input, local_graphs, device, degree_by_behavior=local_degrees)
        return torch.stack((encoded[local_user], encoded[local_id[candidate_item]]))

    def _aggregate_training(self, candidates, factual_u, factual_i, attention, structural):
        indices = [row for row, pairs in enumerate(candidates) if pairs]
        zeros = torch.zeros_like(factual_u)
        valid_rows = torch.zeros(len(candidates), dtype=torch.bool, device=factual_u.device)
        if not indices:
            return zeros, zeros, zeros, valid_rows

        rows = torch.tensor(indices, device=factual_u.device)
        pairs = [torch.stack(candidates[row]) for row in indices]
        padded = pad_sequence(pairs, batch_first=True)
        lengths = torch.tensor([len(row) for row in pairs], device=factual_u.device)
        valid = torch.arange(padded.size(1), device=factual_u.device) < lengths[:, None]
        candidates_u = padded[:, :, 0]
        candidates_i = padded[:, :, 1]
        scores = (candidates_u * candidates_i).sum(dim=-1)
        if structural:
            # Lowest structural score gives the user negative (Eq. 3).
            selection = scores.masked_fill(~valid, float('inf')).argmin(dim=1)
            hard = candidates_u[torch.arange(len(rows), device=rows.device), selection]
        else:
            # Highest semantic score gives the item negative (Eq. 5).
            selection = scores.masked_fill(~valid, float('-inf')).argmax(dim=1)
            hard = candidates_i[torch.arange(len(rows), device=rows.device), selection]

        direction_u = attention(factual_u[rows], candidates_u, valid)
        direction_i = attention(factual_i[rows], candidates_i, valid)
        valid_rows[rows] = True
        return (zeros.index_copy(0, rows, direction_u), zeros.index_copy(0, rows, direction_i),
                zeros.index_copy(0, rows, hard), valid_rows)

    def training_directions(self, users, catalog_items, E_base, E_factual_all, dataset, device):
        self._index(dataset)
        offset = dataset.item_offset
        item_nodes = catalog_items + offset
        user_ids = users.tolist()
        catalog_ids = catalog_items.tolist()
        observed = [self._observed(user, item) for user, item in zip(user_ids, item_nodes.tolist())]
        active_rows = [row for row, behaviors in enumerate(observed) if behaviors]
        has_auxiliary = torch.tensor([bool(behaviors) for behaviors in observed],
                                     dtype=torch.bool, device=device)
        structural, semantic = [], []
        with torch.no_grad():
            replacements = {}
            if active_rows:
                alternatives = self._semantic_items(
                    [user_ids[row] for row in active_rows],
                    [catalog_ids[row] for row in active_rows],
                    item_nodes[active_rows], E_base, E_factual_all, dataset
                ).tolist()
                replacements = dict(zip(active_rows, alternatives))

            for row, (user, item, behaviors) in enumerate(zip(user_ids, catalog_ids, observed)):
                item_node = item + offset
                struct_edits = self._structural_edits(behaviors, item_node)
                structural.append([
                    self._local_candidate(user, item_node, item_node, {behavior: edit}, E_base, dataset, device)
                    for behavior, edit in struct_edits.items()
                ])
                semantic.append([
                    self._local_candidate(user, item_node, alternative + offset,
                                          {behavior: (item_node, alternative + offset) for behavior in behaviors},
                                          E_base, dataset, device)
                    for alternative in replacements.get(row, ()) if alternative
                ])

        factual_u = E_factual_all[users]
        factual_i = E_factual_all[item_nodes]
        str_u, str_i, hard_u, str_valid = self._aggregate_training(
            structural, factual_u, factual_i, self.struct_attention, structural=True)
        sem_u, sem_i, hard_i, sem_valid = self._aggregate_training(
            semantic, factual_u, factual_i, self.sem_attention, structural=False)
        return str_u, str_i, sem_u, sem_i, hard_u, hard_i, str_valid, sem_valid, has_auxiliary

    def _edit_full_edges(self, edges, edits, behavior):
        # Deduplicate edits when the same user is present more than once.
        removed = set()
        added = []
        for user, old, new in set(edits):
            copies = 1
            if old is not None:
                removed.update(((user, old), (old, user)))
                copies = self._behavior_graphs[behavior].edge_count(user, old)
            if new is not None:
                added.extend(((user, new), (new, user)) * copies)

        if removed:
            remove_keys = edges.new_tensor(sorted(source * self.num_nodes + target for source, target in removed))
            keys = edges[0] * self.num_nodes + edges[1]
            positions = torch.searchsorted(remove_keys, keys).clamp(max=len(removed) - 1)
            edges = edges[:, remove_keys[positions] != keys]
        if added:
            edges = torch.cat((edges, edges.new_tensor(sorted(added)).T), dim=1)
        return edges

    @staticmethod
    def _aggregate_inference(factual, candidates, masks, attention):
        directions = torch.zeros_like(factual)
        if not candidates:
            return directions
        valid = torch.stack(masks, dim=1)
        active = valid.any(dim=1)
        if active.any():
            values = torch.stack(candidates, dim=1)[active]
            directions[active] = attention(factual[active], values, valid[active])
        return directions

    def inference_directions(self, users, catalog_items, E_base, E_factual_all, dataset, device):
        self._index(dataset)
        offset = dataset.item_offset
        item_nodes = catalog_items + offset
        user_ids = users.tolist()
        catalog_ids = catalog_items.tolist()
        item_ids = item_nodes.tolist()
        observed = [self._observed(user, item) for user, item in zip(user_ids, item_ids)]
        edit_choices = [self._structural_edits(aux, item) for aux, item in zip(observed, item_ids)]
        graph = {behavior: edges.to(device) for behavior, edges in dataset.edge_index.items()}
        str_candidates, str_masks = [], []
        sem_candidates, sem_masks = [], []

        # Batch the full-graph passes by behavior or replacement rank (Section V-A.4).
        with torch.no_grad():
            for behavior in self.aux_behaviors:
                valid = [behavior in choices for choices in edit_choices]
                if not any(valid):
                    continue
                changes = [(user, *choices[behavior]) for user, choices in zip(user_ids, edit_choices)
                           if behavior in choices]
                edited = dict(graph)
                edited[behavior] = self._edit_full_edges(graph[behavior], changes, behavior)
                _, encoded = self.cascade_encoder(E_base, edited, device)
                str_candidates.append(encoded[users])
                str_masks.append(torch.tensor(valid, device=device))

            replacements = self._semantic_items(user_ids, catalog_ids, item_nodes, E_base, E_factual_all,
                                                dataset).tolist()
            for alternatives in zip(*replacements):
                valid = [bool(alternative and aux) for alternative, aux in zip(alternatives, observed)]
                if not any(valid):
                    continue
                edited = dict(graph)
                for behavior in self.aux_behaviors:
                    changes = [(user, item, alternative + offset)
                               for user, item, alternative, aux in zip(user_ids, item_ids, alternatives, observed)
                               if alternative and behavior in aux]
                    if changes:
                        edited[behavior] = self._edit_full_edges(graph[behavior], changes, behavior)
                _, encoded = self.cascade_encoder(E_base, edited, device)
                candidate_nodes = torch.tensor([item + offset for item in alternatives], device=device)
                sem_candidates.append(encoded[candidate_nodes])
                sem_masks.append(torch.tensor(valid, device=device))

        str_u = self._aggregate_inference(E_factual_all[users], str_candidates, str_masks, self.struct_attention)
        sem_i = self._aggregate_inference(E_factual_all[item_nodes], sem_candidates, sem_masks, self.sem_attention)
        return str_u, sem_i


class CausalRepresentationDisentanglement(nn.Module):
    """Subtract the structural and then semantic projection (Eq. 8)."""

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def _project_and_excise(self, target_emb, basis_emb):
        dot_product = torch.sum(target_emb * basis_emb, dim=-1, keepdim=True)
        norm_sq = torch.sum(basis_emb * basis_emb, dim=-1, keepdim=True)
        eps = max(self.eps, torch.finfo(norm_sq.dtype).tiny)
        return target_emb - (dot_product / (norm_sq + eps)) * basis_emb

    def refine(self, embedding, structural, semantic):
        residual = self._project_and_excise(embedding, structural)
        return self._project_and_excise(residual, semantic)

    def forward(self, factual_u, factual_i, str_u, sem_u, str_i=None, sem_i=None):
        if str_i is None:
            str_i = str_u
        if sem_i is None:
            sem_i = sem_u
        return self.refine(factual_u, str_u, sem_u), self.refine(factual_i, str_i, sem_i)


class DeconfoundedContrastiveConstraint(nn.Module):
    """Penalize cosine similarity to detached hard negatives (Eq. 12)."""

    def __init__(self, tau=0.1):
        super().__init__()
        if tau <= 0:
            raise ValueError('DCC temperature must be positive.')
        self.tau = tau

    def forward(self, E_anchor, E_hard_negative, reduction='mean'):
        anchor = F.normalize(E_anchor, p=2, dim=-1)
        negative = F.normalize(E_hard_negative.detach(), p=2, dim=-1)
        similarity = torch.sum(anchor * negative, dim=-1)
        penalty = F.softplus(similarity / self.tau)
        if reduction == 'sum':
            return penalty.sum()
        if reduction != 'mean':
            raise ValueError(f'Unsupported DCC reduction: {reduction}')
        return penalty.mean()


class ECRD(nn.Module):
    """Self-Explainable Dual Counterfactual Reasoning (Figure 3)."""

    def __init__(self, args, dataset: DataSet):
        super().__init__()
        self.args = args
        self.device = args.device
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.dataset = dataset
        self.item_offset = dataset.item_offset

        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.factual_cascade = FactualCascadeReasoning(
            args.layers_nums, self.embedding_size, getattr(args, 'node_dropout', 0.0), self.behaviors
        )
        self.dual_cf_reasoning = DualCounterfactualReasoning(args, self.factual_cascade)
        self.causal_disentanglement = CausalRepresentationDisentanglement()
        self.dcc_loss = DeconfoundedContrastiveConstraint(tau=getattr(args, 'tau', 0.1))
        self.bpr_loss = BPRLoss()
        self.reg_weight = args.reg_weight
        self.loss_coefficient = args.loss_coefficient

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.apply(self._init_weights)
        self._load_model()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _load_model(self):
        if self.if_load_model:
            path = os.path.join(self.model_path, self.check_point)
            self.load_state_dict(torch.load(path, map_location='cpu'))

    def _base_embeddings(self):
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    @torch.no_grad()
    def lazy_refresh_semantic_pool(self):
        """Refresh the semantic retrieval snapshot (Section V-A4)."""
        was_training = self.training
        self.eval()
        try:
            base = self._base_embeddings()
            _, factual = self.factual_cascade(base, self.dataset.edge_index, base.device)
            pool_nodes = self.dataset.popular_items_tensor.to(base.device) + self.item_offset
            self.dual_cf_reasoning.cached_base_candidates = base[pool_nodes].detach()
            self.dual_cf_reasoning.cached_out_candidates = factual[pool_nodes].detach()
        finally:
            self.train(was_training)

    def forward(self, batch_ids):
        base = self._base_embeddings()
        device = base.device
        behavior_embeddings, factual = self.factual_cascade(base, self.dataset.edge_index, device)
        loss = base.new_zeros(())
        target_index = len(self.behaviors) - 1

        for index, embeddings in enumerate(behavior_embeddings):
            triplets = batch_ids[:, index, :3].long()
            positive_rows = (triplets[:, :2] > 0).all(dim=-1)
            if not positive_rows.any():
                continue

            users, pos_items, neg_items = triplets[positive_rows].unbind(dim=-1)
            rankable = neg_items > 0
            factual_u = embeddings[users]
            factual_pos = embeddings[pos_items + self.item_offset]
            factual_neg = embeddings[neg_items[rankable] + self.item_offset]

            if index == target_index:
                positive_count = users.numel()
                query_users = torch.cat((users, users[rankable]))
                query_items = torch.cat((pos_items, neg_items[rankable]))
                str_u, str_i, sem_u, sem_i, hard_u, hard_i, str_valid, sem_valid, has_auxiliary = (
                    self.dual_cf_reasoning.training_directions(
                        query_users, query_items, base.detach(), factual.detach(), self.dataset, device
                    )
                )
                # Generate directions per query. Zero directions preserve factual scores.
                refined_u, refined_i = self.causal_disentanglement(
                    torch.cat((factual_u, factual_u[rankable])),
                    torch.cat((factual_pos, factual_neg)),
                    str_u, sem_u, str_i, sem_i
                )
                score_u, score_pos = refined_u[:positive_count], refined_i[:positive_count]
                score_neg_u, score_neg = refined_u[positive_count:], refined_i[positive_count:]

                str_valid = str_valid[:positive_count]
                sem_valid = sem_valid[:positive_count]
                dcc = self.dcc_loss(
                    factual_u[str_valid], hard_u[:positive_count][str_valid], reduction='sum'
                )
                dcc = dcc + self.dcc_loss(
                    factual_pos[sem_valid], hard_i[:positive_count][sem_valid], reduction='sum'
                )
                # Average over positive target pairs with auxiliary interactions (Eq. 12).
                dcc_count = has_auxiliary[:positive_count].sum().clamp_min(1)
                loss = loss + self.args.alpha * dcc / dcc_count
            else:
                score_u, score_pos, score_neg = factual_u, factual_pos, factual_neg
                score_neg_u = factual_u[rankable]

            if rankable.any():
                # Auxiliary factual and target refined dot products (Eqs. 10 and 13).
                pos_scores = torch.sum(score_u[rankable] * score_pos[rankable], dim=-1)
                neg_scores = torch.sum(score_neg_u * score_neg, dim=-1)
                loss = loss + self.loss_coefficient[index] * self.bpr_loss(pos_scores, neg_scores)

        # args.alpha is paper lambda; reg_weight is paper alpha (Eq. 13).
        squared_l2 = sum(param.square().sum() for param in self.parameters() if param.requires_grad)
        return loss + self.reg_weight * squared_l2

    @torch.no_grad()
    def full_predict(self, batch_users):
        base = self._base_embeddings()
        device = base.device
        _, factual = self.factual_cascade(base, self.dataset.edge_index, device)
        factual_users, factual_items = factual.split([self.n_users + 1, self.n_items + 1])
        batch_factual_users = factual_users[batch_users]
        scores = batch_factual_users @ factual_items.T
        if batch_users.numel() == 0 or not self.dataset.popular_items:
            return scores

        rows, items = self.dual_cf_reasoning.auxiliary_catalog_pairs(batch_users, self.dataset)
        if not rows.numel():
            return scores

        # The first popular item is the fixed reference (Section V-A4).
        reference = torch.full_like(batch_users, self.dataset.popular_items[0])
        structural, semantic = self.dual_cf_reasoning.inference_directions(
            batch_users, reference, base, factual, self.dataset, device
        )
        refine = self.causal_disentanglement.refine
        refined_users = refine(batch_factual_users, structural, semantic)
        refined_items = refine(
            factual_items, structural.mean(dim=0, keepdim=True), semantic.mean(dim=0, keepdim=True)
        )
        # Overwrite only observed auxiliary pairs, with bounded gather memory.
        for row_chunk, item_chunk in zip(rows.split(65536), items.split(65536)):
            scores[row_chunk, item_chunk] = (
                refined_users[row_chunk] * refined_items[item_chunk]
            ).sum(dim=-1)
        return scores
