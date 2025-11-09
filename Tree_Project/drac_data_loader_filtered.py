import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from PIL import Image
import torchvision.transforms as transforms
from skimage import feature
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler

class DRACDataset(Dataset):
    """
    DRAC Dataset loader for Diabetic Retinopathy Grading
    Converts retinal images to geometric graphs for SGMP model
    
    NOW WITH IMAGE QUALITY FILTERING:
    - Only uses images with quality grade 0 (Good quality)
    - Filters based on Image Quality Assessment groundtruth
    """
    def __init__(self, 
                 image_dir, 
                 label_csv,
                 quality_csv,
                 transform=None,
                 use_superpixels=True,
                 num_features=5,
                 quality_threshold=0):
        """
        Args:
            image_dir: Directory containing retinal images
            label_csv: Path to CSV file with DR grading labels
            quality_csv: Path to CSV file with image quality labels
            transform: Optional image transformations
            use_superpixels: If True, use superpixel-based graph construction
            num_features: Number of node features to extract
            quality_threshold: Maximum quality grade to include (0=Good, 1=Poor, 2=Reject)
                              Default: 0 (only Good quality images)
        """
        super(DRACDataset, self).__init__()
        
        self.image_dir = image_dir
        self.use_superpixels = use_superpixels
        self.num_features = num_features
        self.quality_threshold = quality_threshold
        
        # Image preprocessing
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        # Load DR grading labels
        self.labels_df = pd.read_csv(label_csv)
        
        # Load image quality labels
        self.quality_df = pd.read_csv(quality_csv)
        
        # Merge on image name
        # Adjust column names based on actual CSV format
        if 'image name' in self.labels_df.columns:
            label_col = 'image name'
        elif 'Image name' in self.labels_df.columns:
            label_col = 'Image name'
        else:
            label_col = self.labels_df.columns[0]
        
        if 'image name' in self.quality_df.columns:
            quality_col = 'image name'
        elif 'Image name' in self.quality_df.columns:
            quality_col = 'Image name'
        else:
            quality_col = self.quality_df.columns[0]
        
        # Rename columns for consistency
        self.labels_df = self.labels_df.rename(columns={label_col: 'image_name'})
        self.quality_df = self.quality_df.rename(columns={quality_col: 'image_name'})
        
        # Get quality column name (usually 'image quality level' or similar)
        quality_grade_col = [col for col in self.quality_df.columns if 'quality' in col.lower() and col != 'image_name']
        if quality_grade_col:
            quality_grade_col = quality_grade_col[0]
        else:
            quality_grade_col = self.quality_df.columns[1]
        
        self.quality_df = self.quality_df.rename(columns={quality_grade_col: 'quality_grade'})
        
        # Merge dataframes
        merged_df = pd.merge(
            self.labels_df, 
            self.quality_df[['image_name', 'quality_grade']], 
            on='image_name', 
            how='inner'
        )
        
        # Filter by quality threshold
        print(f"\nImage Quality Filtering:")
        print(f"  Total images before filtering: {len(merged_df)}")
        print(f"  Quality distribution before filtering:")
        print(merged_df['quality_grade'].value_counts().sort_index())
        
        # Filter to only include images with quality <= threshold
        filtered_df = merged_df[merged_df['quality_grade'] <= quality_threshold]
        
        print(f"\n  Images after quality filtering (quality <= {quality_threshold}): {len(filtered_df)}")
        print(f"  Quality distribution after filtering:")
        print(filtered_df['quality_grade'].value_counts().sort_index())
        
        # Get DR grade column name
        dr_grade_col = [col for col in filtered_df.columns if 'DR' in col or 'grade' in col.lower()]
        dr_grade_col = [col for col in dr_grade_col if col not in ['quality_grade']]
        if dr_grade_col:
            dr_grade_col = dr_grade_col[0]
        else:
            # Try to find any numeric column that's not quality
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            dr_grade_col = [col for col in numeric_cols if col != 'quality_grade'][0]
        
        filtered_df = filtered_df.rename(columns={dr_grade_col: 'dr_grade'})
        
        print(f"\n  DR grade distribution in filtered dataset:")
        print(filtered_df['dr_grade'].value_counts().sort_index())
        
        # Get valid image files
        self.image_files = []
        self.labels = []
        self.quality_grades = []
        
        for idx, row in filtered_df.iterrows():
            img_name = row['image_name']
            dr_label = row['dr_grade']
            quality = row['quality_grade']
            
            # Check if image exists
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                self.image_files.append(img_path)
                self.labels.append(dr_label)
                self.quality_grades.append(quality)
        
        print(f"\n  Final dataset size (existing images): {len(self.image_files)}")
        print(f"  ✓ Using only {'Good quality (0)' if quality_threshold == 0 else f'quality <= {quality_threshold}'} images")
    
    def len(self):
        return len(self.image_files)
    
    def get(self, idx):
        """
        Convert retinal image to geometric graph structure
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            img_tensor = self.transform(image)
        
        # Convert to numpy for processing
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Extract keypoints and build graph
        if self.use_superpixels:
            graph_data = self._build_superpixel_graph(img_np)
        else:
            graph_data = self._build_keypoint_graph(img_np)
        
        # Add label and quality
        label = self.labels[idx]
        quality = self.quality_grades[idx]
        
        graph_data.y = torch.tensor([label], dtype=torch.long)
        graph_data.quality = torch.tensor([quality], dtype=torch.long)
        
        return graph_data
    
    def _build_keypoint_graph(self, image):
        """
        Build graph from image keypoints using corner detection
        """
        # Convert to grayscale
        gray = np.mean(image, axis=2)
        
        # Detect corners using Harris corner detection
        corners = feature.corner_harris(gray)
        corner_coords = feature.corner_peaks(corners, min_distance=5, threshold_rel=0.02)
        
        # If too few points, use grid sampling
        if len(corner_coords) < 50:
            corner_coords = self._grid_sample_points(image.shape[:2], n_points=100)
        
        # Extract positions (normalize to [0, 1])
        pos = corner_coords.astype(np.float32)
        pos[:, 0] /= image.shape[0]
        pos[:, 1] /= image.shape[1]
        
        # Add z-coordinate (can be intensity or zero)
        z_coords = np.zeros((len(pos), 1))
        pos = np.hstack([pos, z_coords])
        
        # Extract features at each point
        node_features = self._extract_node_features(image, corner_coords)
        
        # Build edges using Delaunay triangulation
        if len(pos) >= 4:
            tri = Delaunay(pos[:, :2])
            edge_index = self._delaunay_to_edges(tri)
        else:
            # Fully connected for very small graphs
            n = len(pos)
            edge_index = np.array([[i, j] for i in range(n) for j in range(n) if i != j]).T
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(pos, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        return Data(x=x, pos=pos, edge_index=edge_index)
    
    def _build_superpixel_graph(self, image):
        """
        Build graph from superpixels (SLIC algorithm)
        """
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        
        # Apply SLIC superpixel segmentation
        segments = slic(image, n_segments=100, compactness=10, sigma=1, 
                       start_label=0, channel_axis=2)
        
        # Extract region properties
        props = regionprops(segments + 1, intensity_image=image)
        
        # Build node features and positions
        node_features = []
        positions = []
        
        for prop in props:
            # Position (centroid)
            y, x = prop.centroid
            positions.append([x / image.shape[1], y / image.shape[0], 0])
            
            # Features: mean color, area, intensity, etc.
            mean_color = prop.intensity_image.mean(axis=(0, 1))
            area = prop.area / (image.shape[0] * image.shape[1])
            
            # Additional features
            features = [
                mean_color[0],  # R channel
                mean_color[1],  # G channel
                mean_color[2],  # B channel
                area,            # Normalized area
                prop.eccentricity if hasattr(prop, 'eccentricity') else 0,  # Shape
            ]
            node_features.append(features)
        
        pos = np.array(positions, dtype=np.float32)
        node_features = np.array(node_features, dtype=np.float32)
        
        # Normalize features
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
        
        # Build adjacency using spatial proximity
        edge_index = self._build_knn_edges(pos[:, :2], k=8)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(pos, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        return Data(x=x, pos=pos, edge_index=edge_index)
    
    def _extract_node_features(self, image, coords):
        """
        Extract features at specific coordinates
        """
        features = []
        h, w = image.shape[:2]
        
        for y, x in coords:
            # Sample patch around point
            patch_size = 5
            y_min = max(0, y - patch_size)
            y_max = min(h, y + patch_size)
            x_min = max(0, x - patch_size)
            x_max = min(w, x + patch_size)
            
            patch = image[y_min:y_max, x_min:x_max]
            
            # Extract features
            mean_color = patch.mean(axis=(0, 1))
            std_color = patch.std(axis=(0, 1))
            
            # Combine features
            feature_vec = np.concatenate([mean_color, std_color[:2]])  # 5 features
            features.append(feature_vec)
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features
    
    def _delaunay_to_edges(self, tri):
        """
        Convert Delaunay triangulation to edge list
        """
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                edges.add(edge)
        
        # Convert to bidirectional edge list
        edge_list = []
        for e in edges:
            edge_list.append([e[0], e[1]])
            edge_list.append([e[1], e[0]])
        
        return np.array(edge_list).T
    
    def _build_knn_edges(self, positions, k=8):
        """
        Build k-nearest neighbor edges
        """
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        
        # Build edge list (exclude self-connections)
        edges = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip first (self)
                edges.append([i, j])
        
        return np.array(edges).T
    
    def _grid_sample_points(self, shape, n_points=100):
        """
        Sample points on a regular grid
        """
        h, w = shape
        n_side = int(np.sqrt(n_points))
        
        y_coords = np.linspace(0, h-1, n_side, dtype=int)
        x_coords = np.linspace(0, w-1, n_side, dtype=int)
        
        yy, xx = np.meshgrid(y_coords, x_coords)
        points = np.column_stack([yy.ravel(), xx.ravel()])
        
        return points


def load_drac_data(drac_root, split='811', batch_size=32, quality_threshold=0):
    """
    Load DRAC dataset and create data loaders with quality filtering
    
    Args:
        drac_root: Root directory of DRAC dataset
        split: Train/valid/test split ratio (e.g., '811' for 80/10/10)
        batch_size: Batch size for data loaders
        quality_threshold: Maximum quality grade to include (0=Good, 1=Poor, 2=Reject)
                          Default: 0 (only Good quality images)
    
    Returns:
        train_loader, valid_loader, test_loader
    """
    from torch_geometric.loader import DataLoader
    
    # Paths for DR Grading
    train_img_dir = os.path.join(drac_root, 
                                 'C. Diabetic Retinopathy Grading',
                                 '1. Original Images',
                                 'a. Training Set')
    
    dr_label_csv = os.path.join(drac_root,
                                'C. Diabetic Retinopathy Grading',
                                '2. Groundtruths',
                                'a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv')
    
    # Path for Image Quality Assessment
    quality_csv = os.path.join(drac_root,
                               'B. Image Quality Assessment',
                               '2. Groundtruths',
                               'a. DRAC2022_ Image Quality Assessment_Training Labels.csv')
    
    # Create dataset with quality filtering
    print(f"\n{'='*70}")
    print(f"Loading DRAC Dataset with Quality Filtering")
    print(f"{'='*70}")
    print(f"Image directory: {train_img_dir}")
    print(f"DR labels: {dr_label_csv}")
    print(f"Quality labels: {quality_csv}")
    print(f"Quality threshold: {quality_threshold} (0=Good, 1=Poor, 2=Reject)")
    
    dataset = DRACDataset(
        train_img_dir, 
        dr_label_csv, 
        quality_csv,
        quality_threshold=quality_threshold
    )
    
    # Split dataset
    n_total = len(dataset)
    train_ratio = int(split[0]) / 10.0
    valid_ratio = int(split[1]) / 10.0
    
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    
    # Random permutation
    indices = torch.randperm(n_total)
    
    train_dataset = dataset[indices[:n_train]]
    valid_dataset = dataset[indices[n_train:n_train+n_valid]]
    test_dataset = dataset[indices[n_train+n_valid:]]
    
    print(f"\n{'='*70}")
    print(f"Dataset Split:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Valid: {len(valid_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"  Total: {n_total} images (after quality filtering)")
    print(f"{'='*70}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # Test the data loader with quality filtering
    drac_root = '../DRAC'
    
    print("Testing with different quality thresholds:\n")
    
    # Test 1: Only good quality (quality = 0)
    print("="*70)
    print("TEST 1: Only Good Quality Images (quality = 0)")
    print("="*70)
    train_loader, valid_loader, test_loader = load_drac_data(
        drac_root, 
        quality_threshold=0
    )
    
    # Test loading a batch
    for batch in train_loader:
        print(f"\nSample Batch:")
        print(f"  Number of graphs: {batch.num_graphs}")
        print(f"  Node features shape: {batch.x.shape}")
        print(f"  Positions shape: {batch.pos.shape}")
        print(f"  Edge index shape: {batch.edge_index.shape}")
        print(f"  DR grades: {batch.y.squeeze().numpy()}")
        print(f"  Quality grades: {batch.quality.squeeze().numpy()}")
        break
    
    # Optional: Test with quality <= 1 (Good + Poor)
    print("\n\n" + "="*70)
    print("TEST 2: Good + Poor Quality Images (quality <= 1)")
    print("="*70)
    train_loader2, valid_loader2, test_loader2 = load_drac_data(
        drac_root, 
        quality_threshold=1
    )