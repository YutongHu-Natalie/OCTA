"""
Graph Autoencoder using SGMP as encoder
For learning latent representations of OCTA graphs
"""
import torch
import torch.nn as nn
from torch_scatter import scatter
from .SGMP import SGMP


class GraphEncoder(nn.Module):
    """
    Encoder that uses SGMP to create latent representations
    Returns embeddings before the final classification layer
    """
    def __init__(self, input_channels_node=5, hidden_channels=128,
                 latent_dim=64, num_interactions=3,
                 num_gaussians=(50,6,12), cutoff=10.0, readout='add'):
        super(GraphEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.readout = readout

        # Use SGMP layers for encoding (without final classification layers)
        self.sgmp_base = SGMP(
            input_channels_node=input_channels_node,
            hidden_channels=hidden_channels,
            output_channels=1,  # Will be replaced
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout
        )

        # Replace final layers with latent space projection
        # Remove the classification head from SGMP
        self.sgmp_base.lin2 = nn.Identity()
        self.sgmp_base.lin1 = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x, pos, batch, edge_index_3rd):
        """
        Encode graph to latent representation
        Returns: latent vector of shape (batch_size, latent_dim)
        """
        z = self.sgmp_base(x, pos, batch, edge_index_3rd)
        return z


class GraphDecoder(nn.Module):
    """
    Decoder that reconstructs graph node features and structure from latent vector
    """
    def __init__(self, latent_dim=64, hidden_channels=128,
                 output_node_features=5, max_nodes=150):
        super(GraphDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.output_node_features = output_node_features

        # Decode latent to node-level features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, max_nodes * output_node_features),
        )

        # Optional: predict number of nodes
        self.node_count_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.ReLU()
        )

    def forward(self, z, num_nodes_per_graph=None):
        """
        Decode latent vector to graph features

        Args:
            z: latent vector (batch_size, latent_dim)
            num_nodes_per_graph: optional list of actual node counts per graph

        Returns:
            node_features: reconstructed node features
        """
        batch_size = z.shape[0]

        # Predict node count (for analysis)
        pred_node_count = self.node_count_predictor(z)

        # Decode to node features
        decoded = self.decoder(z)  # (batch_size, max_nodes * output_features)

        # Reshape to (batch_size, max_nodes, output_features)
        node_features = decoded.view(batch_size, self.max_nodes, self.output_node_features)

        return node_features, pred_node_count


class GraphAutoencoder(nn.Module):
    """
    Complete Graph Autoencoder combining encoder and decoder
    """
    def __init__(self, input_channels_node=5, hidden_channels=128,
                 latent_dim=64, num_interactions=3,
                 num_gaussians=(50,6,12), cutoff=10.0,
                 max_nodes=150, readout='add'):
        super(GraphAutoencoder, self).__init__()

        self.encoder = GraphEncoder(
            input_channels_node=input_channels_node,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout
        )

        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            output_node_features=input_channels_node,
            max_nodes=max_nodes
        )

        self.latent_dim = latent_dim
        self.max_nodes = max_nodes

    def encode(self, x, pos, batch, edge_index_3rd):
        """Encode graph to latent space"""
        return self.encoder(x, pos, batch, edge_index_3rd)

    def decode(self, z, num_nodes_per_graph=None):
        """Decode latent vector to graph"""
        return self.decoder(z, num_nodes_per_graph)

    def forward(self, x, pos, batch, edge_index_3rd, num_nodes_per_graph=None):
        """
        Full forward pass: encode then decode

        Returns:
            reconstructed node features, latent representation, predicted node count
        """
        z = self.encode(x, pos, batch, edge_index_3rd)
        node_features_recon, pred_node_count = self.decode(z, num_nodes_per_graph)

        return node_features_recon, z, pred_node_count

    def reconstruction_loss(self, x, pos, batch, edge_index_3rd,
                           node_features_recon, pred_node_count):
        """
        Compute reconstruction loss

        Args:
            x: original node features (num_nodes, num_features)
            batch: batch assignment vector
            node_features_recon: reconstructed features (batch_size, max_nodes, num_features)
            pred_node_count: predicted node counts (batch_size, 1)
        """
        batch_size = node_features_recon.shape[0]
        device = x.device

        # Aggregate original features by graph
        # Get number of nodes per graph
        num_nodes_per_graph = scatter(torch.ones_like(batch), batch, dim=0, reduce='sum')

        # Create masks and compute loss
        recon_loss = 0
        node_count_loss = 0

        for i in range(batch_size):
            # Get nodes for this graph
            mask = (batch == i)
            graph_nodes = x[mask]  # (n_nodes_i, num_features)
            n_nodes = graph_nodes.shape[0]

            # Only compare valid nodes (up to n_nodes)
            if n_nodes <= self.max_nodes:
                recon_loss += nn.functional.mse_loss(
                    node_features_recon[i, :n_nodes],
                    graph_nodes
                )
            else:
                # If graph has more nodes than max, truncate
                recon_loss += nn.functional.mse_loss(
                    node_features_recon[i],
                    graph_nodes[:self.max_nodes]
                )

            # Node count prediction loss
            node_count_loss += nn.functional.mse_loss(
                pred_node_count[i],
                torch.tensor([n_nodes], dtype=torch.float, device=device)
            )

        # Average over batch
        recon_loss = recon_loss / batch_size
        node_count_loss = node_count_loss / batch_size

        return recon_loss, node_count_loss


class GraphClassifier(nn.Module):
    """
    Classifier using SGMP encoder for healthy vs unhealthy classification
    Can use pretrained encoder from autoencoder
    """
    def __init__(self, input_channels_node=5, hidden_channels=128,
                 latent_dim=64, num_classes=2, num_interactions=3,
                 num_gaussians=(50,6,12), cutoff=10.0, readout='add',
                 use_pretrained_encoder=False):
        super(GraphClassifier, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder (same architecture as autoencoder)
        self.encoder = GraphEncoder(
            input_channels_node=input_channels_node,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, pos, batch, edge_index_3rd, return_latent=False):
        """
        Forward pass

        Args:
            return_latent: if True, also return latent representation
        """
        # Encode to latent space
        z = self.encoder(x, pos, batch, edge_index_3rd)

        # Classify
        logits = self.classifier(z)

        if return_latent:
            return logits, z
        return logits

    def load_pretrained_encoder(self, autoencoder_state_dict):
        """
        Load encoder weights from pretrained autoencoder

        Args:
            autoencoder_state_dict: state dict from GraphAutoencoder
        """
        # Extract encoder weights
        encoder_weights = {
            k.replace('encoder.', ''): v
            for k, v in autoencoder_state_dict.items()
            if k.startswith('encoder.')
        }

        self.encoder.load_state_dict(encoder_weights)
        print("Loaded pretrained encoder weights")
