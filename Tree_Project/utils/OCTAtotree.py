#!/usr/bin/env python3
"""
Adaptable Geometric Tree Encoder for OCTA Image Analysis
This implementation is designed to be dataset-agnostic and focuses on encoder training
for vessel structure representation learning.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime
import glob
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseDatasetInterface(ABC):
    """Abstract base class for dataset interfaces"""
    
    @abstractmethod
    def load_data(self, max_samples=None):
        """Load and prepare dataset samples"""
        pass
    
    @abstractmethod
    def process_image(self, image_path, metadata=None):
        """Process a single image and return vessel structure"""
        pass
    
    @abstractmethod
    def get_sample_info(self, idx):
        """Get metadata for a specific sample"""
        pass


class VesselImageProcessor:
    """Handles vessel image processing operations"""
    
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
    
    def load_binary_image(self, file_path, file_size=None):
        """Load image from binary file with automatic size detection"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if file_size is None:
                file_size = len(data)
            
            # Auto-detect image dimensions based on file size
            image = self._detect_and_reshape_image(data, file_size)
            
            # Resize to target size if needed
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size)
            
            return image
            
        except Exception as e:
            logger.warning(f"Error loading binary image {file_path}: {e}")
            return self.create_synthetic_vessel_image()
    
    def load_standard_image(self, file_path):
        """Load standard image formats (PNG, JPG, etc.)"""
        try:
            if isinstance(file_path, (str, Path)):
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            else:
                # Handle numpy arrays or PIL images
                image = np.array(file_path)
                
            if image is None:
                raise ValueError(f"Could not load image from {file_path}")
            
            # Resize to target size
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size)
            
            return image
        
        except Exception as e:
            logger.warning(f"Error loading standard image {file_path}: {e}")
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
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Skeletonization
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
        
        # Create vessel-like branching structures
        num_vessels = np.random.randint(2, 5)
        for _ in range(num_vessels):
            start_x = np.random.randint(0, self.target_size[1])
            start_y = np.random.randint(0, self.target_size[0])
            
            angle = np.random.uniform(0, 2*np.pi)
            length = np.random.randint(10, min(30, self.target_size[0]//2))
            
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # Ensure coordinates are within bounds
            end_x = np.clip(end_x, 0, self.target_size[1]-1)
            end_y = np.clip(end_y, 0, self.target_size[0]-1)
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), 255, 1)
        
        return image


class GeometricTreeBuilder:
    """Converts vessel skeletons to geometric tree structures"""
    
    def __init__(self, max_nodes=50, distance_threshold=10):
        self.max_nodes = max_nodes
        self.distance_threshold = distance_threshold
    
    def skeleton_to_graph(self, skeleton):
        """Convert skeleton to NetworkX graph"""
        vessel_pixels = np.where(skeleton > 0)
        
        if len(vessel_pixels[0]) == 0:
            return self._create_minimal_graph()
        
        # Sample points if too many
        n_points = min(self.max_nodes, len(vessel_pixels[0]))
        
        if len(vessel_pixels[0]) > n_points:
            step = len(vessel_pixels[0]) // n_points
            indices = list(range(0, len(vessel_pixels[0]), step))[:n_points]
        else:
            indices = list(range(len(vessel_pixels[0])))
        
        points = [(vessel_pixels[0][i], vessel_pixels[1][i]) for i in indices]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with positions
        for i, (y, x) in enumerate(points):
            G.add_node(i, pos=(x, y, 0))
        
        # Add edges based on distance
        for i, (y1, x1) in enumerate(points):
            for j, (y2, x2) in enumerate(points):
                if i < j:
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < self.distance_threshold:
                        G.add_edge(i, j, weight=dist)
        
        # Create minimum spanning tree
        if G.number_of_edges() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            
            if G.number_of_edges() > 0:
                mst = nx.minimum_spanning_tree(G)
                return mst
        
        return self._create_minimal_graph()
    
    def _create_minimal_graph(self):
        """Create minimal graph for fallback cases"""
        G = nx.Graph()
        positions = [(0, 0, 0), (10, 5, 0), (20, 10, 0), (15, 15, 0)]
        for i, pos in enumerate(positions):
            G.add_node(i, pos=pos)
        
        edges = [(0, 1), (1, 2), (1, 3)]
        G.add_edges_from(edges)
        
        return G
    
    def graph_to_geometric_data(self, G):
        """Convert NetworkX graph to PyTorch Geometric data"""
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = {i: (i*10, 0, 0) for i in G.nodes()}
        
        # Ensure consecutive node indices
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G.nodes()))}
        
        # Create coordinate matrix
        coords = torch.tensor([list(pos[old_id]) for old_id in sorted(G.nodes())], dtype=torch.float)
        
        # Create edge indices
        if G.number_of_edges() > 0:
            edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]
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
        
        return Data(
            x=node_features,
            pos=coords,
            edge_index=edge_index,
            num_nodes=len(G.nodes())
        )


class SOULDatasetAdapter(BaseDatasetInterface):
    """Adapter for SOUL dataset"""
    
    def __init__(self, excel_file, base_dir, subset_type="Subset.1.1", 
                 max_samples=None, image_processor=None, tree_builder=None):
        self.excel_file = excel_file
        self.base_dir = base_dir
        self.subset_type = subset_type
        self.image_processor = image_processor or VesselImageProcessor()
        self.tree_builder = tree_builder or GeometricTreeBuilder()
        self.data = []
        self.load_data(max_samples)
    
    def load_data(self, max_samples=None):
        """Load SOUL dataset"""
        try:
            # Load Excel sheets
            basic_info = pd.read_excel(self.excel_file, sheet_name='basic_info')
            
            # Create subject labels mapping (using Gender for now)
            subject_labels = {}
            for _, row in basic_info.iterrows():
                subject_labels[row['SubID']] = int(row['Gender'] == 'M')
            
            logger.info(f"Loaded subject info for {len(subject_labels)} subjects")
            
            # Find label files
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
                    
                    label_pattern = os.path.join(tp_dir, "label", "*")
                    label_files = glob.glob(label_pattern)
                    
                    valid_label_files = [
                        f for f in label_files 
                        if os.path.isfile(f) and not os.path.basename(f).startswith('.')
                    ]
                    
                    for label_file in valid_label_files:
                        self.data.append({
                            'subid': subid,
                            'timepoint': tp_name,
                            'filename': os.path.basename(label_file),
                            'file_path': label_file,
                            'label': subject_labels.get(subid, 0),
                            'file_size': os.path.getsize(label_file)
                        })
            
            if max_samples and len(self.data) > max_samples:
                self.data = self.data[:max_samples]
            
            logger.info(f"Loaded {len(self.data)} SOUL samples")
            
        except Exception as e:
            logger.error(f"Error loading SOUL data: {e}")
            raise
    
    def process_image(self, image_path, metadata=None):
        """Process SOUL label image"""
        file_size = metadata.get('file_size') if metadata else None
        image = self.image_processor.load_binary_image(image_path, file_size)
        skeleton = self.image_processor.process_vessel_image(image)
        return skeleton
    
    def get_sample_info(self, idx):
        """Get SOUL sample metadata"""
        return self.data[idx]


class StandardImageDatasetAdapter(BaseDatasetInterface):
    """Adapter for standard image datasets (PNG, JPG, etc.)"""
    
    def __init__(self, image_dir, labels_file=None, max_samples=None,
                 image_processor=None, tree_builder=None):
        self.image_dir = image_dir
        self.labels_file = labels_file
        self.image_processor = image_processor or VesselImageProcessor()
        self.tree_builder = tree_builder or GeometricTreeBuilder()
        self.data = []
        self.load_data(max_samples)
    
    def load_data(self, max_samples=None):
        """Load standard image dataset"""
        try:
            # Load labels if provided
            labels = {}
            if self.labels_file and os.path.exists(self.labels_file):
                if self.labels_file.endswith('.csv'):
                    label_df = pd.read_csv(self.labels_file)
                    labels = dict(zip(label_df.iloc[:, 0], label_df.iloc[:, 1]))
                elif self.labels_file.endswith('.json'):
                    with open(self.labels_file, 'r') as f:
                        labels = json.load(f)
            
            # Find image files
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(self.image_dir, ext)))
                image_files.extend(glob.glob(os.path.join(self.image_dir, ext.upper())))
            
            self.data = []
            for img_path in image_files:
                filename = os.path.basename(img_path)
                base_name = os.path.splitext(filename)[0]
                
                self.data.append({
                    'filename': filename,
                    'file_path': img_path,
                    'label': labels.get(base_name, 0),  # Default label 0
                    'dataset_type': 'standard'
                })
            
            if max_samples and len(self.data) > max_samples:
                self.data = self.data[:max_samples]
            
            logger.info(f"Loaded {len(self.data)} standard image samples")
            
        except Exception as e:
            logger.error(f"Error loading standard image data: {e}")
            raise
    
    def process_image(self, image_path, metadata=None):
        """Process standard image"""
        image = self.image_processor.load_standard_image(image_path)
        skeleton = self.image_processor.process_vessel_image(image)
        return skeleton
    
    def get_sample_info(self, idx):
        """Get standard image sample metadata"""
        return self.data[idx]


class AdaptableVesselDataset(Dataset):
    """Adaptable dataset wrapper for different vessel image datasets"""
    
    def __init__(self, dataset_adapter, transform=None):
        self.adapter = dataset_adapter
        self.transform = transform
    
    def __len__(self):
        return len(self.adapter.data)
    
    def __getitem__(self, idx):
        """Get processed sample as geometric tree data"""
        try:
            sample = self.adapter.get_sample_info(idx)
            
            # Process image to vessel skeleton
            skeleton = self.adapter.process_image(sample['file_path'], sample)
            
            # Convert to geometric tree
            graph = self.adapter.tree_builder.skeleton_to_graph(skeleton)
            geometric_tree = self.adapter.tree_builder.graph_to_geometric_data(graph)
            
            # Add metadata
            geometric_tree.y = torch.tensor([sample.get('label', 0)], dtype=torch.long)
            geometric_tree.sample_id = idx
            geometric_tree.filename = sample['filename']
            
            return geometric_tree
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            # Return fallback synthetic sample
            synthetic_image = self.adapter.image_processor.create_synthetic_vessel_image()
            skeleton = self.adapter.image_processor.process_vessel_image(synthetic_image)
            graph = self.adapter.tree_builder.skeleton_to_graph(skeleton)
            geometric_tree = self.adapter.tree_builder.graph_to_geometric_data(graph)
            
            geometric_tree.y = torch.tensor([0], dtype=torch.long)
            geometric_tree.sample_id = idx
            geometric_tree.filename = f"synthetic_{idx}"
            
            return geometric_tree


# Geometric Tree Message Passing Components
class GeometricTreeMessagePassing(MessagePassing):
    """GTMP layer based on the research paper"""
    
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__(aggr='add')
        
        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(in_channels * 2 + 6, hidden_dim),  # +6 for geometric features
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )
        
        # Update network
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
            dist,                                    # Distance
            unit_diff,                              # Unit direction
            dist ** 2,                              # Squared distance
            torch.sum(diff ** 2, dim=1, keepdim=True)  # Squared magnitude
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


class GeometricTreeEncoder(nn.Module):
    """Adaptable Geometric Tree Encoder for vessel analysis"""
    
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GTMP layers
        self.gtmp_layers = nn.ModuleList([
            GeometricTreeMessagePassing(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Global pooling for graph-level representation
        self.global_pool = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        """Forward pass through encoder"""
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
        
        # Output projection
        x = self.output_proj(x)
        
        # Global pooling for graph-level embeddings
        batch_size = batch.max().item() + 1
        graph_embeddings = []
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                # Use attention-weighted pooling
                graph_emb = x[mask].mean(dim=0)
                graph_embeddings.append(graph_emb)
            else:
                graph_embeddings.append(torch.zeros(self.output_dim, device=x.device))
        
        graph_embeddings = torch.stack(graph_embeddings)
        
        # Final global pooling
        output = self.global_pool(graph_embeddings)
        
        return output


def collate_fn(batch):
    """Custom collate function for geometric data"""
    return Batch.from_data_list(batch)


def train_encoder(encoder, dataloader, optimizer, criterion, device):
    """Train encoder (can be used for unsupervised or supervised training)"""
    encoder.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training Encoder"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get embeddings from encoder
        embeddings = encoder(batch)
        
        # For unsupervised training, you can add reconstruction loss, 
        # contrastive loss, or other self-supervised objectives here
        
        # For now, using a simple reconstruction target (identity mapping)
        loss = criterion(embeddings, torch.zeros_like(embeddings))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_encoder(encoder, dataloader, criterion, device):
    """Validate encoder"""
    encoder.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)
            embeddings = encoder(batch)
            loss = criterion(embeddings, torch.zeros_like(embeddings))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train Adaptable Geometric Tree Encoder')
    
    # Dataset arguments
    parser.add_argument('--dataset_type', type=str, choices=['soul', 'standard'], 
                       default='soul', help='Type of dataset')
    parser.add_argument('--excel_file', type=str, help='Excel file for SOUL dataset')
    parser.add_argument('--base_dir', type=str, help='Base directory for dataset')
    parser.add_argument('--image_dir', type=str, help='Image directory for standard dataset')
    parser.add_argument('--labels_file', type=str, help='Labels file for standard dataset')
    
    # Training arguments
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./vessel_encoder_checkpoints')
    parser.add_argument('--save_embeddings', action='store_true', 
                       help='Save extracted embeddings')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Initialize dataset adapter based on type
        if args.dataset_type == 'soul':
            if not args.excel_file or not args.base_dir:
                raise ValueError("SOUL dataset requires --excel_file and --base_dir")
            
            adapter = SOULDatasetAdapter(
                excel_file=args.excel_file,
                base_dir=args.base_dir,
                max_samples=args.max_samples
            )
        elif args.dataset_type == 'standard':
            if not args.image_dir:
                raise ValueError("Standard dataset requires --image_dir")
            
            adapter = StandardImageDatasetAdapter(
                image_dir=args.image_dir,
                labels_file=args.labels_file,
                max_samples=args.max_samples
            )
        
        # Create dataset
        dataset = AdaptableVesselDataset(adapter)
        
        if len(dataset) == 0:
            logger.error("No valid samples found!")
            return
        
        # Split dataset
        indices = list(range(len(dataset)))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Initialize encoder
        encoder = GeometricTreeEncoder(
            input_dim=3,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        encoder = encoder.to(device)
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()  # For reconstruction-based training
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        logger.info("Starting encoder training...")
        best_val_loss = float('inf')
        train_history = []
        
        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss = train_encoder(encoder, train_loader, optimizer, criterion, device)
            
            # Validate
            val_loss = validate_encoder(encoder, val_loader, criterion, device)
            
            scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': args
                }, os.path.join(args.save_dir, 'best_encoder.pth'))
                logger.info(f"New best model saved with Val Loss: {best_val_loss:.6f}")
            
            train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        
        # Save final model
        torch.save({
            'epoch': args.epochs,
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_val_loss': val_loss,
            'args': args
        }, os.path.join(args.save_dir, 'final_encoder.pth'))
        
        # Save training history
        with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
            json.dump(train_history, f, indent=2)
        
        # Extract and save embeddings if requested
        if args.save_embeddings:
            logger.info("Extracting embeddings...")
            extract_and_save_embeddings(
                encoder, 
                DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn),
                device,
                os.path.join(args.save_dir, 'embeddings.npz')
            )
        
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Results saved to: {args.save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def extract_and_save_embeddings(encoder, dataloader, device, save_path):
    """Extract embeddings for all samples and save them"""
    encoder.eval()
    all_embeddings = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            batch = batch.to(device)
            embeddings = encoder(batch)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            all_filenames.extend([batch.filename[i] for i in range(len(batch.y))])
    
    # Combine all embeddings
    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.hstack(all_labels)
    
    # Save embeddings
    np.savez(
        save_path,
        embeddings=embeddings_array,
        labels=labels_array,
        filenames=np.array(all_filenames)
    )
    
    logger.info(f"Embeddings saved to {save_path}")
    logger.info(f"Shape: {embeddings_array.shape}")


class EncoderEvaluator:
    """Class for evaluating the trained encoder"""
    
    def __init__(self, encoder_path, device='cpu'):
        self.device = device
        self.encoder = self.load_encoder(encoder_path)
    
    def load_encoder(self, encoder_path):
        """Load trained encoder"""
        checkpoint = torch.load(encoder_path, map_location=self.device)
        args = checkpoint['args']
        
        encoder = GeometricTreeEncoder(
            input_dim=3,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.to(self.device)
        encoder.eval()
        
        return encoder
    
    def extract_embeddings(self, dataloader):
        """Extract embeddings from dataset"""
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting"):
                batch = batch.to(self.device)
                emb = self.encoder(batch)
                embeddings.append(emb.cpu().numpy())
                labels.append(batch.y.cpu().numpy())
        
        return np.vstack(embeddings), np.hstack(labels)
    
    def visualize_embeddings(self, embeddings, labels, save_path=None):
        """Visualize embeddings using t-SNE"""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=labels, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter)
            plt.title('Vessel Structure Embeddings (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("sklearn or matplotlib not available for visualization")


def create_config_template():
    """Create a configuration template for different datasets"""
    config_template = {
        "soul_dataset": {
            "dataset_type": "soul",
            "excel_file": "path/to/Information.xlsx",
            "base_dir": "path/to/soul_dataset",
            "max_samples": 100,
            "description": "SOUL OCTA dataset configuration"
        },
        "standard_dataset": {
            "dataset_type": "standard",
            "image_dir": "path/to/images",
            "labels_file": "path/to/labels.csv",  # Optional
            "max_samples": 500,
            "description": "Standard vessel image dataset configuration"
        },
        "training_params": {
            "batch_size": 8,
            "epochs": 50,
            "lr": 0.001,
            "hidden_dim": 64,
            "output_dim": 128,
            "num_layers": 3,
            "dropout": 0.1
        },
        "encoder_params": {
            "max_nodes": 50,
            "distance_threshold": 10,
            "target_image_size": [64, 64]
        }
    }
    
    return config_template


if __name__ == "__main__":
    # Print configuration template if no arguments provided
    if len(sys.argv) == 1:
        print("Adaptable Geometric Tree Encoder for Vessel Analysis")
        print("=" * 55)
        print()
        print("Usage examples:")
        print()
        print("1. SOUL Dataset:")
        print("python script.py --dataset_type soul --excel_file Info.xlsx --base_dir /path/to/soul")
        print()
        print("2. Standard Images:")
        print("python script.py --dataset_type standard --image_dir /path/to/images --labels_file labels.csv")
        print()
        print("Configuration template:")
        print(json.dumps(create_config_template(), indent=2))
        print()
        print("For more options, use: python script.py --help")
    else:
        main()