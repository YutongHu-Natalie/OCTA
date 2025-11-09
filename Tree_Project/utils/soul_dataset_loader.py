#!/usr/bin/env python3
"""
GT-SSL Implementation for SOUL Dataset
Proper implementation of Geometric Tree Self-Supervised Learning
Based on "Representation Learning of Geometric Trees" paper
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
import glob
from scipy.stats import wasserstein_distance
import itertools

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VesselImageProcessor:
    """Handles vessel image processing operations"""
    
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
    
    def load_image(self, file_path, file_size=None):
        """Load image from file (binary or standard formats like JPG)"""
        try:
            # Check if it's a standard image format
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Could not load image: {file_path}")
            else:
                # Handle binary files
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                if file_size is None:
                    file_size = len(data)
                
                image = self._detect_and_reshape_image(data, file_size)
            
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size)
            
            return image
            
        except Exception as e:
            logger.warning(f"Error loading image {file_path}: {e}")
            return self.create_synthetic_vessel_image()
    
    def _detect_and_reshape_image(self, data, file_size):
        """Automatically detect image dimensions from binary data"""
        if file_size == 4096:  # 64x64 grayscale
            image = np.frombuffer(data, dtype=np.uint8).reshape(64, 64)
        elif file_size == 16384:  # 128x128 grayscale
            image = np.frombuffer(data, dtype=np.uint8).reshape(128, 128)
        elif file_size == 12288:  # 64x64 RGB
            image = np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 3)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            # Try to guess dimensions for square images
            total_pixels = file_size
            side = int(np.sqrt(total_pixels))
            if side * side == total_pixels:
                image = np.frombuffer(data, dtype=np.uint8).reshape(side, side)
            else:
                logger.warning(f"Unknown file size {file_size}, creating synthetic image")
                return self.create_synthetic_vessel_image()
        
        return image
    
    def process_vessel_image(self, image):
        """Convert vessel image to tree structure"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        skeleton = self.skeletonize(binary)
        return skeleton
    
    def skeletonize(self, binary_img):
        """Perform skeletonization on binary vessel image"""
        binary_img = binary_img // 255
        skeleton = np.zeros_like(binary_img, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        iterations = 0
        max_iterations = 100
        
        while iterations < max_iterations:
            opened = cv2.morphologyEx(binary_img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            temp = cv2.subtract(binary_img.astype(np.uint8), opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary_img = opened.copy()
            
            if cv2.countNonZero(binary_img) == 0:
                break
            iterations += 1
        
        return skeleton * 255
    
    def create_synthetic_vessel_image(self):
        """Create synthetic vessel pattern for fallback"""
        image = np.zeros(self.target_size, dtype=np.uint8)
        
        num_vessels = np.random.randint(2, 5)
        for _ in range(num_vessels):
            start_x = np.random.randint(0, self.target_size[1])
            start_y = np.random.randint(0, self.target_size[0])
            
            angle = np.random.uniform(0, 2*np.pi)
            length = np.random.randint(10, min(30, self.target_size[0]//2))
            
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            end_x = np.clip(end_x, 0, self.target_size[1]-1)
            end_y = np.clip(end_y, 0, self.target_size[0]-1)
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), 255, 1)
        
        return image


class GeometricTreeBuilder:
    """Converts vessel skeletons to geometric tree structures with hierarchy"""
    
    def __init__(self, max_nodes=50, distance_threshold=10):
        self.max_nodes = max_nodes
        self.distance_threshold = distance_threshold
    
    def skeleton_to_graph(self, skeleton):
        """Convert skeleton to NetworkX tree with proper hierarchy"""
        vessel_pixels = np.where(skeleton > 0)
        
        if len(vessel_pixels[0]) == 0:
            return self._create_minimal_tree()
        
        n_points = min(self.max_nodes, len(vessel_pixels[0]))
        
        if len(vessel_pixels[0]) > n_points:
            step = len(vessel_pixels[0]) // n_points
            indices = list(range(0, len(vessel_pixels[0]), step))[:n_points]
        else:
            indices = list(range(len(vessel_pixels[0])))
        
        points = [(vessel_pixels[0][i], vessel_pixels[1][i]) for i in indices]
        
        # Create tree structure instead of general graph
        tree = self._create_tree_from_points(points)
        
        return tree
    
    def _create_tree_from_points(self, points):
        """Create a tree structure from vessel points"""
        if len(points) < 2:
            return self._create_minimal_tree()
        
        # Find center point as root
        center_x = sum(p[1] for p in points) / len(points)
        center_y = sum(p[0] for p in points) / len(points)
        
        # Find closest point to center as root
        distances_to_center = [(i, np.sqrt((p[1]-center_x)**2 + (p[0]-center_y)**2)) 
                              for i, p in enumerate(points)]
        root_idx = min(distances_to_center, key=lambda x: x[1])[0]
        
        # Build tree using BFS-like approach from root
        T = nx.DiGraph()  # Directed tree
        visited = set()
        queue = [root_idx]
        visited.add(root_idx)
        
        # Add root node
        T.add_node(root_idx, pos=(points[root_idx][1], points[root_idx][0], 0))
        
        while queue and len(visited) < len(points):
            current = queue.pop(0)
            current_pos = points[current]
            
            # Find closest unvisited points
            candidates = []
            for i, point in enumerate(points):
                if i not in visited:
                    dist = np.sqrt((current_pos[1] - point[1])**2 + (current_pos[0] - point[0])**2)
                    if dist < self.distance_threshold:
                        candidates.append((i, dist))
            
            # Add closest candidates as children (limit branching factor)
            candidates.sort(key=lambda x: x[1])
            max_children = min(3, len(candidates))  # Limit branching
            
            for i, (child_idx, dist) in enumerate(candidates[:max_children]):
                if child_idx not in visited:
                    T.add_node(child_idx, pos=(points[child_idx][1], points[child_idx][0], 0))
                    T.add_edge(current, child_idx, weight=dist)
                    visited.add(child_idx)
                    queue.append(child_idx)
        
        # If tree is disconnected, connect remaining nodes
        while len(visited) < len(points):
            # Find closest unvisited point to any visited point
            min_dist = float('inf')
            best_connection = None
            
            for unvisited in set(range(len(points))) - visited:
                for visited_node in visited:
                    dist = np.sqrt((points[unvisited][1] - points[visited_node][1])**2 + 
                                 (points[unvisited][0] - points[visited_node][0])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_connection = (visited_node, unvisited)
            
            if best_connection:
                parent, child = best_connection
                T.add_node(child, pos=(points[child][1], points[child][0], 0))
                T.add_edge(parent, child, weight=min_dist)
                visited.add(child)
        
        return T
    
    def _create_minimal_tree(self):
        """Create minimal tree for fallback cases"""
        T = nx.DiGraph()
        positions = [(0, 0, 0), (10, 5, 0), (20, 10, 0), (15, 15, 0)]
        for i, pos in enumerate(positions):
            T.add_node(i, pos=pos)
        
        # Create tree structure: 0->1, 1->2, 1->3
        edges = [(0, 1), (1, 2), (1, 3)]
        for parent, child in edges:
            T.add_edge(parent, child, weight=10.0)
        
        return T
    
    def graph_to_geometric_data(self, T):
        """Convert NetworkX tree to PyTorch Geometric data with hierarchy info"""
        pos = nx.get_node_attributes(T, 'pos')
        if not pos:
            pos = {i: (i*10, 0, 0) for i in T.nodes()}
        
        # Ensure consecutive node indices
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(T.nodes()))}
        
        # Create coordinate matrix
        coords = torch.tensor([list(pos[old_id]) for old_id in sorted(T.nodes())], dtype=torch.float)
        
        # Create edge indices (directed edges for tree)
        if T.number_of_edges() > 0:
            edges = [(node_mapping[u], node_mapping[v]) for u, v in T.edges()]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Create normalized node features
        node_features = coords.clone()
        if coords.numel() > 0:
            coords_min = coords.min(dim=0, keepdim=True)[0]
            coords_max = coords.max(dim=0, keepdim=True)[0]
            coords_range = coords_max - coords_min + 1e-8
            node_features = (coords - coords_min) / coords_range
        
        # Add hierarchy information for GT-SSL - store as simple tensors
        hierarchy_info = self._compute_hierarchy_tensors(T, node_mapping)
        
        data = Data(
            x=node_features,
            pos=coords,
            edge_index=edge_index,
            num_nodes=len(T.nodes())
        )
        
        # Add GT-SSL specific attributes as tensors (not dicts)
        data.depths = hierarchy_info['depths']
        data.parent_child_pairs = hierarchy_info['parent_child_pairs']
        data.num_children = hierarchy_info['num_children']
        
        return data
    
    def _compute_hierarchy_tensors(self, T, node_mapping):
        """Compute hierarchy information as tensors for batching"""
        hierarchy = {}
        
        # Find root node (node with no predecessors)
        root_candidates = [n for n in T.nodes() if T.in_degree(n) == 0]
        root = root_candidates[0] if root_candidates else list(T.nodes())[0]
        
        # Compute depths
        depths = {}
        parent_child_pairs = []
        num_children = {}
        
        # BFS to compute depths and relationships
        queue = [(root, 0)]
        depths[root] = 0
        num_children[root] = len(list(T.successors(root)))
        
        while queue:
            node, depth = queue.pop(0)
            for child in T.successors(node):
                depths[child] = depth + 1
                parent_child_pairs.append((node, child))
                num_children[child] = len(list(T.successors(child)))
                queue.append((child, depth + 1))
        
        # Convert to tensors using node_mapping
        hierarchy['depths'] = torch.tensor([depths.get(old_id, 0) for old_id in sorted(T.nodes())], dtype=torch.float)
        hierarchy['parent_child_pairs'] = torch.tensor(
            [[node_mapping[p], node_mapping[c]] for p, c in parent_child_pairs], 
            dtype=torch.long
        ) if parent_child_pairs else torch.zeros((0, 2), dtype=torch.long)
        hierarchy['num_children'] = torch.tensor([num_children.get(old_id, 0) for old_id in sorted(T.nodes())], dtype=torch.float)
        
        return hierarchy


class SOULDatasetAdapter:
    """Adapter for SOUL dataset"""
    
    def __init__(self, excel_file, base_dir, max_samples=None):
        self.excel_file = excel_file
        self.base_dir = base_dir
        self.image_processor = VesselImageProcessor()
        self.tree_builder = GeometricTreeBuilder()
        self.data = []
        self.load_data(max_samples)
    
    def load_data(self, max_samples=None):
        """Load SOUL dataset"""
        try:
            basic_info = pd.read_excel(self.excel_file, sheet_name='basic_info')
            
            subject_labels = {}
            for _, row in basic_info.iterrows():
                subject_labels[row['SubID']] = int(row['Gender'] == 'M')
            
            logger.info(f"Loaded subject info for {len(subject_labels)} subjects")
            
            self.data = []
            subject_pattern = os.path.join(self.base_dir, "S*")
            subject_dirs = glob.glob(subject_pattern)
            
            for subject_dir in subject_dirs:
                subject_name = os.path.basename(subject_dir)
                try:
                    subid = int(subject_name[1:])
                except:
                    continue
                
                timepoint_pattern = os.path.join(subject_dir, "s*")
                timepoint_dirs = glob.glob(timepoint_pattern)
                
                for tp_dir in timepoint_dirs:
                    tp_name = os.path.basename(tp_dir)
                    
                    label_base = os.path.join(tp_dir, "label")
                    if os.path.exists(label_base):
                        for root, dirs, files in os.walk(label_base):
                            for file in files:
                                if not file.startswith('.'):
                                    file_path = os.path.join(root, file)
                                    
                                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) or not '.' in file:
                                        self.data.append({
                                            'subid': subid,
                                            'timepoint': tp_name,
                                            'filename': file,
                                            'file_path': file_path,
                                            'label': subject_labels.get(subid, 0),
                                            'file_size': os.path.getsize(file_path)
                                        })
            
            if max_samples and len(self.data) > max_samples:
                random.shuffle(self.data)  # Randomize selection
                self.data = self.data[:max_samples]
            
            logger.info(f"Loaded {len(self.data)} SOUL samples")
            
        except Exception as e:
            logger.error(f"Error loading SOUL data: {e}")
            raise


class SOULDataset(Dataset):
    """SOUL dataset for GT-SSL geometric tree encoder"""
    
    def __init__(self, adapter):
        self.adapter = adapter
    
    def __len__(self):
        return len(self.adapter.data)
    
    def __getitem__(self, idx):
        """Get processed sample as geometric tree data with GT-SSL info"""
        try:
            sample = self.adapter.data[idx]
            
            # Process image to vessel skeleton
            image = self.adapter.image_processor.load_image(
                sample['file_path'], sample['file_size']
            )
            skeleton = self.adapter.image_processor.process_vessel_image(image)
            
            # Convert to geometric tree with hierarchy
            tree = self.adapter.tree_builder.skeleton_to_graph(skeleton)
            geometric_tree = self.adapter.tree_builder.graph_to_geometric_data(tree)
            
            # Add metadata
            geometric_tree.y = torch.tensor([sample['label']], dtype=torch.long)
            geometric_tree.sample_id = idx
            geometric_tree.filename = sample['filename']
            
            return geometric_tree
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            # Fallback synthetic sample
            synthetic_image = self.adapter.image_processor.create_synthetic_vessel_image()
            skeleton = self.adapter.image_processor.process_vessel_image(synthetic_image)
            tree = self.adapter.tree_builder.skeleton_to_graph(skeleton)
            geometric_tree = self.adapter.tree_builder.graph_to_geometric_data(tree)
            
            geometric_tree.y = torch.tensor([0], dtype=torch.long)
            geometric_tree.sample_id = idx
            geometric_tree.filename = f"synthetic_{idx}"
            
            return geometric_tree


class GeometricTreeMessagePassing(MessagePassing):
    """GTMP layer based on the research paper"""
    
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__(aggr='add')
        
        self.message_net = nn.Sequential(
            nn.Linear(in_channels * 2 + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )
    
    def compute_geometric_features(self, pos_i, pos_j):
        """Compute geometric features between node pairs"""
        diff = pos_i - pos_j
        dist = torch.norm(diff, dim=1, keepdim=True)
        unit_diff = diff / (dist + 1e-8)
        
        geometric_features = torch.cat([
            dist,
            unit_diff,
            dist ** 2,
            torch.sum(diff ** 2, dim=1, keepdim=True)
        ], dim=1)
        
        return geometric_features
    
    def message(self, x_i, x_j, pos_i, pos_j):
        """Compute messages between nodes"""
        geometric_features = self.compute_geometric_features(pos_i, pos_j)
        message_input = torch.cat([x_i, x_j, geometric_features], dim=1)
        return self.message_net(message_input)
    
    def update(self, aggr_out, x):
        """Update node representations"""
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_net(update_input)
    
    def forward(self, x, pos, edge_index):
        """Forward pass"""
        return self.propagate(edge_index, x=x, pos=pos)


class SubtreeGenerator(nn.Module):
    """
    Subtree Generative Learning Module
    Predicts child branch distributions from parent node information
    """
    
    def __init__(self, hidden_dim=64, num_basis_functions=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_basis_functions = num_basis_functions
        
        # Network to predict child distributions
        self.child_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Predict mean child position offset
        )
    
    def forward(self, node_embeddings, data, batch_info):
        """
        Args:
            node_embeddings: Node embeddings from GTMP layers
            data: Batch data with hierarchy information
            batch_info: Batch information for multiple graphs
        """
        if not hasattr(data, 'parent_child_pairs') or not hasattr(data, 'num_children'):
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        parent_child_pairs = data.parent_child_pairs
        num_children = data.num_children
        
        if len(parent_child_pairs) == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        total_loss = 0.0
        num_predictions = 0
        
        # Process each parent-child relationship
        for parent_idx, child_idx in parent_child_pairs:
            if parent_idx >= len(node_embeddings) or child_idx >= len(node_embeddings):
                continue
            
            parent_emb = node_embeddings[parent_idx]
            child_pos = data.pos[child_idx]
            parent_pos = data.pos[parent_idx]
            
            # Predict child position offset from parent
            predicted_offset = self.child_predictor(parent_emb)
            actual_offset = child_pos - parent_pos
            
            # L2 loss for position prediction
            loss = F.mse_loss(predicted_offset, actual_offset)
            total_loss += loss
            num_predictions += 1
        
        if num_predictions > 0:
            return total_loss / num_predictions
        else:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)


class PartialOrderingConstraint(nn.Module):
    """
    Partial Ordering Constraint Module
    Enforces hierarchical relationships in embedding space
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, node_embeddings, data, batch_info):
        """
        Args:
            node_embeddings: Node embeddings from GTMP layers
            data: Batch data with hierarchy information
            batch_info: Batch information for multiple graphs
        """
        if not hasattr(data, 'parent_child_pairs'):
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        parent_child_pairs = data.parent_child_pairs
        
        if len(parent_child_pairs) == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Process parent-child pairs
        for parent_idx, child_idx in parent_child_pairs:
            if parent_idx >= len(node_embeddings) or child_idx >= len(node_embeddings):
                continue
            
            parent_emb = node_embeddings[parent_idx]
            child_emb = node_embeddings[child_idx]
            
            # Partial ordering constraint: encourage parent_emb >= child_emb component-wise
            ordering_diff = parent_emb - child_emb
            ordering_score = torch.sum(ordering_diff)
            
            # Max-margin loss for positive pairs (enforce ordering)
            loss = torch.relu(self.margin - ordering_score)
            total_loss += loss
            num_pairs += 1
        
        # Add some negative pairs (random non-hierarchical pairs)
        if num_pairs > 0 and len(node_embeddings) > 2:
            # Sample some random pairs that are not hierarchical
            num_neg_pairs = min(num_pairs, 5)  # Limit negative pairs
            all_hierarchical = set((p.item(), c.item()) for p, c in parent_child_pairs)
            
            for _ in range(num_neg_pairs):
                idx1 = random.randint(0, len(node_embeddings) - 1)
                idx2 = random.randint(0, len(node_embeddings) - 1)
                
                if idx1 != idx2 and (idx1, idx2) not in all_hierarchical and (idx2, idx1) not in all_hierarchical:
                    emb1 = node_embeddings[idx1]
                    emb2 = node_embeddings[idx2]
                    
                    # For negative pairs, penalize strong ordering in either direction
                    ordering_diff_12 = torch.sum(emb1 - emb2)
                    ordering_diff_21 = torch.sum(emb2 - emb1)
                    
                    # Penalize if either direction has strong ordering
                    neg_loss = torch.relu(ordering_diff_12 - self.margin) + torch.relu(ordering_diff_21 - self.margin)
                    total_loss += neg_loss
                    num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        if hierarchy is None or 'parent_child_pairs' not in hierarchy:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        parent_child_pairs = hierarchy['parent_child_pairs']
        if len(parent_child_pairs) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        total_loss = 0.0
        num_pairs = 0
        
        for parent_idx, child_idx in parent_child_pairs:
            if parent_idx >= len(embeddings) or child_idx >= len(embeddings):
                continue
            
            parent_emb = embeddings[parent_idx]
            child_emb = embeddings[child_idx]
            
            # Partial ordering constraint: child_emb <= parent_emb component-wise
            # Loss = max(0, margin - sum(parent_emb - child_emb))
            ordering_diff = parent_emb - child_emb
            ordering_score = torch.sum(ordering_diff)
            
            # Max-margin loss for positive pairs (enforce ordering)
            loss = torch.relu(self.margin - ordering_score)
            total_loss += loss
            num_pairs += 1
        
        # Add negative pairs (non-hierarchical pairs should not have ordering)
        # Sample random non-hierarchical pairs
        all_pairs = set((p.item(), c.item()) for p, c in parent_child_pairs)
        all_nodes = list(range(len(embeddings)))
        
        neg_pairs_added = 0
        for _ in range(min(len(parent_child_pairs), 10)):  # Limit negative pairs
            i, j = random.sample(all_nodes, 2)
            if (i, j) not in all_pairs and (j, i) not in all_pairs:
                emb_i = embeddings[i]
                emb_j = embeddings[j]
                
                # For negative pairs, penalize strong ordering in either direction
                ordering_diff_ij = torch.sum(emb_i - emb_j)
                ordering_diff_ji = torch.sum(emb_j - emb_i)
                
                # Penalize if either direction has strong ordering
                neg_loss = torch.relu(ordering_diff_ij - self.margin) + torch.relu(ordering_diff_ji - self.margin)
                total_loss += neg_loss
                neg_pairs_added += 1
        
        total_pairs = num_pairs + neg_pairs_added
        if total_pairs > 0:
            return total_loss / total_pairs
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


class GeometricTreeEncoder(nn.Module):
    """GT-SSL Geometric Tree Encoder with proper self-supervised objectives"""
    
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gtmp_layers = nn.ModuleList([
            GeometricTreeMessagePassing(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # GT-SSL modules
        self.subtree_generator = SubtreeGenerator(hidden_dim)
        self.partial_ordering = PartialOrderingConstraint(margin=1.0)
        
        # Global pooling for graph-level representation
        self.global_pool = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data, return_ssl_losses=False):
        """Forward pass through encoder with optional GT-SSL losses"""
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        
        # Input projection
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)
        
        # GTMP layers with residual connections
        for layer, norm in zip(self.gtmp_layers, self.layer_norms):
            x_new = layer(x, pos, edge_index)
            x_new = norm(x_new)
            x_new = self.dropout(x_new)
            x = F.relu(x_new) + x  # Residual connection
        
        # Store node embeddings for SSL losses
        node_embeddings = x.clone()
        
        # Output projection
        x = self.output_proj(x)
        
        # Global pooling for graph-level embeddings
        batch_size = batch.max().item() + 1
        graph_embeddings = []
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                graph_emb = x[mask].mean(dim=0)
                graph_embeddings.append(graph_emb)
            else:
                graph_embeddings.append(torch.zeros(self.output_dim, device=x.device))
        
        graph_embeddings = torch.stack(graph_embeddings)
        output = self.global_pool(graph_embeddings)
        
        if return_ssl_losses:
            # Compute GT-SSL losses
            ssl_losses = self._compute_ssl_losses(node_embeddings, data, batch)
            return output, ssl_losses
        
        return output
    
    def _compute_ssl_losses(self, node_embeddings, data, batch):
        """Compute GT-SSL self-supervised losses"""
        ssl_losses = {}
        
        try:
            # Compute subtree generative loss
            gen_loss = self.subtree_generator(node_embeddings, data, batch)
            ssl_losses['generative'] = gen_loss
            
            # Compute partial ordering loss
            order_loss = self.partial_ordering(node_embeddings, data, batch)
            ssl_losses['ordering'] = order_loss
            
        except Exception as e:
            logger.warning(f"Error computing SSL losses: {e}")
            ssl_losses['generative'] = torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
            ssl_losses['ordering'] = torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        return ssl_losses


def collate_fn(batch):
    """Custom collate function for geometric data with hierarchy info"""
    try:
        # Use standard PyG batching - hierarchy info is now stored as tensors
        batched_data = Batch.from_data_list(batch)
        return batched_data
    except Exception as e:
        logger.warning(f"Collate function error: {e}")
        # Fallback: try without problematic attributes
        clean_batch = []
        for item in batch:
            clean_item = Data(
                x=item.x,
                pos=item.pos,
                edge_index=item.edge_index,
                num_nodes=item.num_nodes
            )
            # Add hierarchy info if available
            if hasattr(item, 'depths'):
                clean_item.depths = item.depths
            if hasattr(item, 'parent_child_pairs'):
                clean_item.parent_child_pairs = item.parent_child_pairs  
            if hasattr(item, 'num_children'):
                clean_item.num_children = item.num_children
            if hasattr(item, 'y'):
                clean_item.y = item.y
            if hasattr(item, 'sample_id'):
                clean_item.sample_id = item.sample_id
            if hasattr(item, 'filename'):
                clean_item.filename = item.filename
            clean_batch.append(clean_item)
        
        return Batch.from_data_list(clean_batch)


def train_gtssl_epoch(encoder, dataloader, optimizer, device, lambda_gen=1.0, lambda_order=1.0):
    """Train encoder with GT-SSL objectives"""
    encoder.train()
    total_gen_loss = 0
    total_order_loss = 0
    total_combined_loss = 0
    
    for batch in tqdm(dataloader, desc="GT-SSL Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass with SSL losses
        embeddings, ssl_losses = encoder(batch, return_ssl_losses=True)
        
        # Combine GT-SSL losses
        gen_loss = ssl_losses['generative']
        order_loss = ssl_losses['ordering']
        
        combined_loss = lambda_gen * gen_loss + lambda_order * order_loss
        
        # Backward pass
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_gen_loss += gen_loss.item()
        total_order_loss += order_loss.item()
        total_combined_loss += combined_loss.item()
    
    return {
        'generative_loss': total_gen_loss / len(dataloader),
        'ordering_loss': total_order_loss / len(dataloader),
        'combined_loss': total_combined_loss / len(dataloader)
    }


def validate_gtssl(encoder, dataloader, device, lambda_gen=1.0, lambda_order=1.0):
    """Validate encoder with GT-SSL objectives"""
    encoder.eval()
    total_gen_loss = 0
    total_order_loss = 0
    total_combined_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="GT-SSL Validation"):
            batch = batch.to(device)
            
            embeddings, ssl_losses = encoder(batch, return_ssl_losses=True)
            
            gen_loss = ssl_losses['generative']
            order_loss = ssl_losses['ordering']
            combined_loss = lambda_gen * gen_loss + lambda_order * order_loss
            
            total_gen_loss += gen_loss.item()
            total_order_loss += order_loss.item()
            total_combined_loss += combined_loss.item()
    
    return {
        'generative_loss': total_gen_loss / len(dataloader),
        'ordering_loss': total_order_loss / len(dataloader),
        'combined_loss': total_combined_loss / len(dataloader)
    }


class TrainingConfig:
    """GT-SSL Training configuration"""
    def __init__(self):
        # Dataset configuration
        self.excel_file = "Tree_Project/Infomation.xlsx"
        self.base_dir = "SOUL_image"
        self.max_samples = 200  # Reasonable size for GT-SSL training
        
        # Training configuration
        self.batch_size = 8
        self.epochs = 50
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # Model configuration
        self.hidden_dim = 64
        self.output_dim = 128
        self.num_layers = 3
        self.dropout = 0.1
        
        # GT-SSL specific parameters
        self.lambda_generative = 1.0  # Weight for generative loss
        self.lambda_ordering = 0.5    # Weight for ordering loss
        
        # Output configuration
        self.save_dir = "./gtssl_results"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """Main GT-SSL training function"""
    logger.info("Starting GT-SSL Training on SOUL Dataset")
    logger.info("Using proper Geometric Tree Self-Supervised Learning")
    
    config = TrainingConfig()
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    try:
        # Verify files exist
        if not os.path.exists(config.excel_file):
            logger.error(f"Excel file not found: {config.excel_file}")
            return
        
        if not os.path.exists(config.base_dir):
            logger.error(f"Base directory not found: {config.base_dir}")
            return
        
        # Create dataset
        logger.info("Loading dataset with hierarchy information...")
        adapter = SOULDatasetAdapter(
            excel_file=config.excel_file,
            base_dir=config.base_dir,
            max_samples=config.max_samples
        )
        
        dataset = SOULDataset(adapter)
        
        if len(dataset) == 0:
            logger.error("No samples found!")
            return
        
        # Split dataset
        indices = list(range(len(dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                              shuffle=False, collate_fn=collate_fn)
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        # Create GT-SSL model
        encoder = GeometricTreeEncoder(
            input_dim=3,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(device)
        
        total_params = sum(p.numel() for p in encoder.parameters())
        logger.info(f"GT-SSL model created with {total_params:,} parameters")
        
        # Setup training
        optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        
        # Training loop with GT-SSL objectives
        best_val_loss = float('inf')
        training_history = []
        
        logger.info("Starting GT-SSL training with subtree generative and partial ordering objectives...")
        
        for epoch in range(config.epochs):
            logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
            
            # Training
            train_losses = train_gtssl_epoch(
                encoder, train_loader, optimizer, device,
                lambda_gen=config.lambda_generative,
                lambda_order=config.lambda_ordering
            )
            
            # Validation
            val_losses = validate_gtssl(
                encoder, val_loader, device,
                lambda_gen=config.lambda_generative,
                lambda_order=config.lambda_ordering
            )
            
            # Learning rate scheduling
            scheduler.step(val_losses['combined_loss'])
            current_lr = optimizer.param_groups[0]['lr']
            
            # Logging
            logger.info(f"Train - Gen: {train_losses['generative_loss']:.6f}, "
                       f"Order: {train_losses['ordering_loss']:.6f}, "
                       f"Combined: {train_losses['combined_loss']:.6f}")
            logger.info(f"Val   - Gen: {val_losses['generative_loss']:.6f}, "
                       f"Order: {val_losses['ordering_loss']:.6f}, "
                       f"Combined: {val_losses['combined_loss']:.6f}")
            logger.info(f"Learning Rate: {current_lr:.8f}")
            
            # Save best model
            if val_losses['combined_loss'] < best_val_loss:
                best_val_loss = val_losses['combined_loss']
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config.__dict__,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, os.path.join(config.save_dir, 'best_gtssl_encoder.pth'))
                logger.info(f"New best GT-SSL model saved! Combined Loss: {best_val_loss:.6f}")
            
            # Record training history
            epoch_info = {
                'epoch': epoch + 1,
                'train_generative': train_losses['generative_loss'],
                'train_ordering': train_losses['ordering_loss'],
                'train_combined': train_losses['combined_loss'],
                'val_generative': val_losses['generative_loss'],
                'val_ordering': val_losses['ordering_loss'],
                'val_combined': val_losses['combined_loss'],
                'learning_rate': current_lr
            }
            training_history.append(epoch_info)
        
        # Save training history
        with open(os.path.join(config.save_dir, 'gtssl_training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info("GT-SSL Training completed successfully!")
        logger.info(f"Best combined loss: {best_val_loss:.6f}")
        logger.info(f"Model saved to: {config.save_dir}/best_gtssl_encoder.pth")
        
        # Extract meaningful embeddings
        logger.info("Extracting GT-SSL embeddings...")
        encoder.eval()
        all_embeddings = []
        all_labels = []
        embedding_variances = []
        
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn):
                batch = batch.to(device)
                embeddings = encoder(batch, return_ssl_losses=False)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())
                
                # Compute embedding variance (should not be near zero)
                emb_var = torch.var(embeddings, dim=0).mean().item()
                embedding_variances.append(emb_var)
        
        embeddings_array = np.vstack(all_embeddings)
        labels_array = np.hstack(all_labels)
        avg_variance = np.mean(embedding_variances)
        
        np.savez(os.path.join(config.save_dir, 'gtssl_embeddings.npz'),
                embeddings=embeddings_array, labels=labels_array)
        
        logger.info(f"GT-SSL embeddings saved: shape {embeddings_array.shape}")
        logger.info(f"Average embedding variance: {avg_variance:.6f}")
        
        if avg_variance < 0.001:
            logger.warning("Low embedding variance detected - model may need more training or different hyperparameters")
        else:
            logger.info("Embeddings show good variance - ready for explanation analysis!")
        
        logger.info("GT-SSL training complete! The model now uses proper self-supervised objectives.")
        
    except Exception as e:
        logger.error(f"GT-SSL training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()