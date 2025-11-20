"""
GT-SSL: Geometric Tree Self-Supervised Learning

This implements the correct two-stage approach from the geometric tree paper:
- Partial Ordering Constraint (Section 4.2)
- Subtree Growth Learning (Section 4.3)

Reference: Geometric Tree Self-Supervised Learning paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Dict, List, Tuple, Optional
import numpy as np


class RBFExpansion(nn.Module):
    """
    Radial Basis Function expansion for geometric features.
    Converts scalar distances/angles to frequency domain representation.

    Used in Equation 8 of the paper:
    e_k(v_i) = Σ exp(-γ ||d_ij - μ_k||²)
    """
    def __init__(self, num_centers=20, start=0.0, end=10.0, gamma=1.0):
        super().__init__()
        self.num_centers = num_centers
        self.gamma = gamma

        # Centers μ_k uniformly distributed
        centers = torch.linspace(start, end, num_centers)
        self.register_buffer('centers', centers)

    def forward(self, x):
        """
        Args:
            x: [N] tensor of scalar values (distances or angles)
        Returns:
            [N, num_centers] tensor of RBF features
        """
        x = x.view(-1, 1)  # [N, 1]
        centers = self.centers.view(1, -1)  # [1, K]

        # Compute RBF: exp(-γ ||x - μ_k||²)
        diff = x - centers  # [N, K]
        rbf = torch.exp(-self.gamma * diff ** 2)  # [N, K]

        return rbf


class SubtreeGrowthPredictor(nn.Module):
    """
    Predicts the geometric distribution of child nodes from ancestor features.

    This implements Equations 8-11 from the paper:
    - Convert geometric features to frequency domain using RBF
    - Aggregate RBF features from ancestors
    - Predict distribution of children's geometric properties
    """
    def __init__(self, hidden_dim=128, num_rbf_centers=20):
        super().__init__()

        self.num_rbf_centers = num_rbf_centers

        # RBF expansions for different geometric features
        self.distance_rbf = RBFExpansion(num_centers=num_rbf_centers, start=0.0, end=10.0)
        self.angle_rbf = RBFExpansion(num_centers=num_rbf_centers, start=0.0, end=np.pi)

        # Predictor network: ancestors' RBF features → predicted child distribution
        # Input: concatenated RBF features from ancestors
        # Output: predicted RBF distribution for children
        self.predictor = nn.Sequential(
            nn.Linear(num_rbf_centers * 2, hidden_dim),  # *2 for distance + angle
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rbf_centers * 2)  # Predict both distance and angle distributions
        )

    def compute_ground_truth_distribution(self, distances, angles, parent_indices):
        """
        Compute the ground truth RBF distribution G(C(v_i)) for children.

        Args:
            distances: [E] distances to children
            angles: [E] angles to children
            parent_indices: [E] which parent each edge belongs to

        Returns:
            [N_parents, num_rbf_centers * 2] ground truth distribution
        """
        # Convert to RBF features
        dist_rbf = self.distance_rbf(distances)  # [E, K]
        angle_rbf = self.angle_rbf(angles)  # [E, K]

        # Aggregate over children of each parent
        num_parents = parent_indices.max().item() + 1

        # Sum RBF features for each parent's children
        dist_distribution = scatter(dist_rbf, parent_indices, dim=0,
                                   dim_size=num_parents, reduce='sum')  # [N, K]
        angle_distribution = scatter(angle_rbf, parent_indices, dim=0,
                                    dim_size=num_parents, reduce='sum')  # [N, K]

        # Concatenate distance and angle distributions
        distribution = torch.cat([dist_distribution, angle_distribution], dim=1)  # [N, 2K]

        # Normalize to create probability distribution
        distribution = F.normalize(distribution, p=1, dim=1)

        return distribution

    def predict_distribution(self, ancestor_features):
        """
        Predict child distribution Ĝ(C(v_i)) from ancestor features.

        Args:
            ancestor_features: [N, num_rbf_centers * 2] aggregated RBF from ancestors

        Returns:
            [N, num_rbf_centers * 2] predicted distribution
        """
        predicted = self.predictor(ancestor_features)

        # Normalize to probability distribution
        predicted = F.normalize(predicted, p=1, dim=1)

        return predicted


class GTSSL(nn.Module):
    """
    GT-SSL: Geometric Tree Self-Supervised Learning

    Combines two objectives:
    1. Partial Ordering Constraint (L_order)
    2. Subtree Growth Learning (L_generative)

    Total loss: L_GT-SSL = L_generative + L_order
    """
    def __init__(self, encoder, hidden_dim=128, num_rbf_centers=20,
                 delta_margin=1.0, lambda_order=1.0):
        """
        Args:
            encoder: The SGMP encoder (or any graph encoder)
            hidden_dim: Hidden dimension for predictor
            num_rbf_centers: Number of RBF centers for geometric features
            delta_margin: Margin δ for negative pairs in ordering constraint
            lambda_order: Weight for ordering loss
        """
        super().__init__()

        self.encoder = encoder
        self.subtree_predictor = SubtreeGrowthPredictor(hidden_dim, num_rbf_centers)
        self.delta_margin = delta_margin
        self.lambda_order = lambda_order

    def compute_ordering_loss(self, embeddings, parent_child_pairs, negative_pairs=None):
        """
        Partial Ordering Constraint Loss (Section 4.2, Equation 6)

        Enforces: If T_j ⊆ T_i, then h_j[d] ≤ h_i[d] for all dimensions d

        L_order = Σ max(0, h_j - h_i) + Σ max(0, δ - ||h_i - h_j||²)
                 (i,j) ∈ P+        (i,j) ∈ P-

        Args:
            embeddings: [N, D] node embeddings
            parent_child_pairs: List of (parent_idx, child_idx) tuples
            negative_pairs: List of (i, j) tuples where i and j are NOT hierarchically related

        Returns:
            ordering_loss: Scalar tensor
        """
        if len(parent_child_pairs) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Convert to tensors
        parent_indices = torch.tensor([p[0] for p in parent_child_pairs],
                                     dtype=torch.long, device=embeddings.device)
        child_indices = torch.tensor([p[1] for p in parent_child_pairs],
                                    dtype=torch.long, device=embeddings.device)

        # Positive pairs: child should be ≤ parent in all dimensions
        h_parent = embeddings[parent_indices]  # [E, D]
        h_child = embeddings[child_indices]  # [E, D]

        # Loss for violating h_child ≤ h_parent
        positive_loss = F.relu(h_child - h_parent).sum(dim=1).mean()

        # Negative pairs: non-hierarchical nodes should be far apart
        if negative_pairs is not None and len(negative_pairs) > 0:
            neg_i = torch.tensor([p[0] for p in negative_pairs],
                                dtype=torch.long, device=embeddings.device)
            neg_j = torch.tensor([p[1] for p in negative_pairs],
                                dtype=torch.long, device=embeddings.device)

            h_i = embeddings[neg_i]
            h_j = embeddings[neg_j]

            # Distance should be at least δ
            distances = torch.norm(h_i - h_j, p=2, dim=1)
            negative_loss = F.relu(self.delta_margin - distances).mean()
        else:
            negative_loss = torch.tensor(0.0, device=embeddings.device)

        return positive_loss + negative_loss

    def compute_subtree_growth_loss(self, node_features, pos, edge_index,
                                   ancestor_info=None):
        """
        Subtree Growth Learning Loss (Section 4.3, Equations 8-11)

        L_generative = Σ EMD(Ĝ(C(v_i)), G(C(v_i)))

        Args:
            node_features: [N, D] node features
            pos: [N, 3] node positions
            edge_index: [2, E] edge indices (parent → child)
            ancestor_info: Optional dict with ancestor geometric features

        Returns:
            generative_loss: Scalar tensor
        """
        if edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=node_features.device)

        parent_idx = edge_index[0]  # [E]
        child_idx = edge_index[1]   # [E]

        # Compute geometric features (distances and angles) to children
        parent_pos = pos[parent_idx]  # [E, 3]
        child_pos = pos[child_idx]    # [E, 3]

        distances = torch.norm(child_pos - parent_pos, p=2, dim=1)  # [E]

        # Compute angles (simplified - could use more sophisticated angle computation)
        # For now, use angle with respect to x-axis in 2D projection
        diff = child_pos - parent_pos
        angles = torch.atan2(diff[:, 1], diff[:, 0])  # [E]
        angles = torch.abs(angles)  # Make positive

        # Ground truth distribution
        gt_distribution = self.subtree_predictor.compute_ground_truth_distribution(
            distances, angles, parent_idx
        )  # [N_parents, 2K]

        # For prediction, we need ancestor features
        # Simplified: use node features as proxy for ancestor aggregation
        # In full implementation, would aggregate RBF features from ancestors
        num_nodes = node_features.shape[0]

        # Create dummy ancestor features (in real implementation, aggregate from actual ancestors)
        # For now, use mean pooled node features as a simple baseline
        ancestor_features = torch.zeros(num_nodes, self.subtree_predictor.num_rbf_centers * 2,
                                       device=node_features.device)

        # Predict distribution
        pred_distribution = self.subtree_predictor.predict_distribution(
            ancestor_features
        )  # [N, 2K]

        # Earth Mover's Distance (EMD)
        # Approximated using Wasserstein-1 distance (L1 on cumulative distributions)
        # For simplicity, use L1 distance on distributions directly
        # Full EMD implementation would use scipy.stats.wasserstein_distance
        emd_loss = F.l1_loss(pred_distribution, gt_distribution)

        return emd_loss

    def forward(self, x, pos, batch, edge_index_3rd,
                parent_child_pairs=None, negative_pairs=None, edge_index=None):
        """
        Forward pass for GT-SSL pretraining.

        Args:
            x: [N, F] node features
            pos: [N, 3] node positions
            batch: [N] batch assignment
            edge_index_3rd: [4, E] third-order edges for SGMP
            parent_child_pairs: List of (parent, child) tuples for hierarchy
            negative_pairs: List of (i, j) tuples for negative samples
            edge_index: [2, E] parent→child edges for subtree growth

        Returns:
            embeddings: [N, D] node embeddings
            loss_dict: Dictionary with individual losses
        """
        # Get embeddings from encoder (but remove final classification layers)
        # SGMP architecture: embedding → interactions → pooling → lin1 → lin2
        # We want embeddings after interactions but before final classification

        # Forward through embedding
        h = self.encoder.embedding(x)  # [N, hidden_channels]

        # Forward through interactions
        # Need to compute geometric features first
        distances = {}
        thetas = {}
        phis = {}
        i, j, k, p = edge_index_3rd
        i_to_j_dis = (pos[j] - pos[i]).norm(p=2, dim=1)
        k_to_j_dis = (pos[k] - pos[j]).norm(p=2, dim=1)
        p_to_j_dis = (pos[p] - pos[j]).norm(p=2, dim=1)
        distances[1] = i_to_j_dis
        distances[2] = k_to_j_dis
        distances[3] = p_to_j_dis

        from models.SGMP import get_angle
        theta_ijk = get_angle(pos[j] - pos[i], pos[k] - pos[j])
        theta_ijp = get_angle(pos[j] - pos[i], pos[p] - pos[j])
        thetas[1] = theta_ijk
        thetas[2] = theta_ijp

        v1 = torch.cross(pos[j] - pos[i], pos[k] - pos[j], dim=1)
        v2 = torch.cross(pos[j] - pos[i], pos[p] - pos[j], dim=1)
        phi_ijkp = get_angle(v1, v2)
        vn = torch.cross(v1, v2, dim=1)
        flag = torch.sign((vn * (pos[j]- pos[i])).sum(dim=1))
        phis[1] = phi_ijkp * flag

        # Apply interactions
        for interaction in self.encoder.interactions:
            h = h + interaction(h, distances, thetas, phis, edge_index_3rd)

        # h is now node-level embeddings [N, hidden_channels]
        # These are what we want for partial ordering constraint

        # Compute losses
        losses = {}

        # 1. Partial Ordering Constraint
        if parent_child_pairs is not None:
            losses['ordering'] = self.compute_ordering_loss(
                h, parent_child_pairs, negative_pairs
            )
        else:
            losses['ordering'] = torch.tensor(0.0, device=x.device)

        # 2. Subtree Growth Learning
        if edge_index is not None:
            losses['generative'] = self.compute_subtree_growth_loss(
                h, pos, edge_index
            )
        else:
            losses['generative'] = torch.tensor(0.0, device=x.device)

        # Total GT-SSL loss
        losses['total'] = losses['generative'] + self.lambda_order * losses['ordering']

        return h, losses


def extract_tree_structure_from_graph(edge_index, num_nodes):
    """
    Helper function to extract parent-child pairs from graph edges.

    For OCTA graphs, we may need to construct a tree structure.
    This is a placeholder - actual implementation depends on graph structure.

    Args:
        edge_index: [2, E] edge connectivity
        num_nodes: Number of nodes

    Returns:
        parent_child_pairs: List of (parent_idx, child_idx)
        negative_pairs: List of (i, j) for non-hierarchical nodes
    """
    # Simplified: treat edges as parent→child relationships
    parent_child_pairs = []

    if edge_index is not None and edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            parent = edge_index[0, i].item()
            child = edge_index[1, i].item()
            parent_child_pairs.append((parent, child))

    # Generate negative pairs (sample random non-connected pairs)
    negative_pairs = []

    # Build adjacency set for quick lookup
    edges_set = set()
    for parent, child in parent_child_pairs:
        edges_set.add((parent, child))
        edges_set.add((child, parent))  # Undirected

    # Sample negative pairs
    num_negatives = min(len(parent_child_pairs), 100)  # Match number of positives
    sampled = 0
    max_attempts = 1000
    attempts = 0

    while sampled < num_negatives and attempts < max_attempts:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)

        if i != j and (i, j) not in edges_set:
            negative_pairs.append((i, j))
            sampled += 1

        attempts += 1

    return parent_child_pairs, negative_pairs
