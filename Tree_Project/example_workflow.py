"""
Example workflow demonstrating the complete pipeline:
1. Load data
2. Train classifier
3. Analyze latent space
4. Perform perturbations

This is a simplified example for demonstration purposes.
"""

import os
import torch
import numpy as np
from drac_data_loader import load_drac_data
from models.graph_autoencoder import GraphClassifier
from latent_analysis import LatentSpaceAnalyzer
from utils.utils import add_self_loops, find_higher_order_neighbors


def example_inference():
    """
    Example: Load trained model and make predictions
    """
    print("="*80)
    print("Example 1: Model Inference")
    print("="*80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = GraphClassifier(
        input_channels_node=5,
        hidden_channels=128,
        latent_dim=64,
        num_classes=2,
        num_interactions=3
    ).to(device)

    checkpoint_path = './results/drac_classifier/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first using train_drac_classifier.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Validation ROC-AUC: {checkpoint['val_roc']:.4f}")

    # Load test data
    drac_root = './DRAC'  # Adjust to your path
    if not os.path.exists(drac_root):
        print(f"DRAC dataset not found at {drac_root}")
        return

    _, _, test_loader = load_drac_data(
        drac_root=drac_root,
        split='811',
        batch_size=1,  # Single sample for example
        binary_classification=True
    )

    # Make predictions on first batch
    data = next(iter(test_loader))
    x, pos, edge_index, batch, y = (
        data.x.float().to(device),
        data.pos.to(device),
        data.edge_index.to(device),
        data.batch.to(device),
        data.y.long().to(device)
    )

    # Process graph
    num_nodes = data.num_nodes
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
        edge_index, num_nodes, order=3
    )

    # Forward pass
    with torch.no_grad():
        logits, latent = model(x, pos, batch, edge_index_3rd, return_latent=True)
        probs = torch.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1)

    print(f"\nPrediction Results:")
    print(f"  True label: {y.item()} ({'Healthy' if y.item() == 0 else 'Unhealthy'})")
    print(f"  Predicted: {prediction.item()} ({'Healthy' if prediction.item() == 0 else 'Unhealthy'})")
    print(f"  Confidence: {probs[0, prediction.item()].item():.4f}")
    print(f"  Probabilities: [Healthy: {probs[0, 0].item():.4f}, Unhealthy: {probs[0, 1].item():.4f}]")
    print(f"  Latent vector shape: {latent.shape}")
    print(f"  Latent vector (first 5 dims): {latent[0, :5].cpu().numpy()}")


def example_latent_analysis():
    """
    Example: Analyze latent space representations
    """
    print("\n" + "="*80)
    print("Example 2: Latent Space Analysis")
    print("="*80)

    latents_file = './results/drac_classifier/latent_representations.npz'
    if not os.path.exists(latents_file):
        print(f"Latent representations not found at {latents_file}")
        print("Please train a model first using train_drac_classifier.py")
        return

    # Load analyzer
    analyzer = LatentSpaceAnalyzer(latents_file)

    # Find discriminative dimensions
    print("\nIdentifying discriminative dimensions...")
    results = analyzer.identify_discriminative_dimensions(top_k=5)

    # Get most discriminative dimension
    top_dim = results[0]['dimension']
    print(f"\nMost discriminative dimension: {top_dim}")
    print(f"  Cohen's d: {results[0]['cohens_d']:.3f}")
    print(f"  Mean (healthy): {results[0]['mean_healthy']:.3f}")
    print(f"  Mean (unhealthy): {results[0]['mean_unhealthy']:.3f}")

    # Compute healthy-unhealthy direction
    direction, healthy_mean, unhealthy_mean = analyzer.compute_healthy_unhealthy_direction()
    print(f"\nHealthy→Unhealthy direction computed")
    print(f"  Direction vector (first 5 dims): {direction[:5]}")


def example_perturbation():
    """
    Example: Perturb latent vectors along discriminative dimensions
    """
    print("\n" + "="*80)
    print("Example 3: Latent Perturbation")
    print("="*80)

    latents_file = './results/drac_classifier/latent_representations.npz'
    if not os.path.exists(latents_file):
        print(f"Latent representations not found at {latents_file}")
        return

    analyzer = LatentSpaceAnalyzer(latents_file)

    # Find discriminative dimensions
    results = analyzer.identify_discriminative_dimensions(top_k=1)
    top_dim = results[0]['dimension']

    # Take a healthy test sample
    healthy_indices = np.where(analyzer.test_labels == 0)[0]
    if len(healthy_indices) == 0:
        print("No healthy samples in test set")
        return

    sample_idx = healthy_indices[0]
    base_latent = analyzer.test_latents[sample_idx]

    print(f"\nPerturbing healthy sample along dimension {top_dim}")
    print(f"Original latent value at dim {top_dim}: {base_latent[top_dim]:.3f}")

    # Create perturbations
    perturbations = analyzer.create_perturbation_vectors(
        dimension=top_dim,
        num_steps=5,
        scale=2.0
    )

    print(f"\nPerturbation amounts: {perturbations}")

    # Apply perturbations
    perturbed_latents = []
    for i, p in enumerate(perturbations):
        perturbed = analyzer.perturb_latent(base_latent, top_dim, p)
        perturbed_latents.append(perturbed)
        print(f"  Step {i}: {base_latent[top_dim]:.3f} → {perturbed[top_dim]:.3f} (Δ={p:.3f})")

    print("\nTo visualize these perturbations, you would:")
    print("1. Pass each perturbed latent through the classifier")
    print("2. Observe how predictions change")
    print("3. (Future) Decode to graph structure to visualize changes")


def example_interpolation():
    """
    Example: Interpolate between healthy and unhealthy samples
    """
    print("\n" + "="*80)
    print("Example 4: Healthy ↔ Unhealthy Interpolation")
    print("="*80)

    latents_file = './results/drac_classifier/latent_representations.npz'
    if not os.path.exists(latents_file):
        print(f"Latent representations not found at {latents_file}")
        return

    analyzer = LatentSpaceAnalyzer(latents_file)

    # Find one healthy and one unhealthy sample
    healthy_indices = np.where(analyzer.test_labels == 0)[0]
    unhealthy_indices = np.where(analyzer.test_labels == 1)[0]

    if len(healthy_indices) == 0 or len(unhealthy_indices) == 0:
        print("Need both healthy and unhealthy samples in test set")
        return

    healthy_latent = analyzer.test_latents[healthy_indices[0]]
    unhealthy_latent = analyzer.test_latents[unhealthy_indices[0]]

    print(f"Interpolating between:")
    print(f"  Healthy sample (first 5 dims): {healthy_latent[:5]}")
    print(f"  Unhealthy sample (first 5 dims): {unhealthy_latent[:5]}")

    # Create interpolation
    interpolated = analyzer.interpolate_between_samples(
        healthy_latent,
        unhealthy_latent,
        num_steps=5
    )

    print(f"\nInterpolation steps:")
    for i, z in enumerate(interpolated):
        alpha = i / (len(interpolated) - 1)
        print(f"  Step {i} (α={alpha:.2f}): {z[:5]}")

    print("\nTo use these interpolated latents:")
    print("1. Pass through classifier to see prediction transitions")
    print("2. Identify at which point the prediction changes")
    print("3. This boundary represents the decision boundary in latent space")


def example_nearest_neighbors():
    """
    Example: Find nearest neighbors in latent space
    """
    print("\n" + "="*80)
    print("Example 5: Nearest Neighbors")
    print("="*80)

    latents_file = './results/drac_classifier/latent_representations.npz'
    if not os.path.exists(latents_file):
        print(f"Latent representations not found at {latents_file}")
        return

    analyzer = LatentSpaceAnalyzer(latents_file)

    # Take a test sample
    query_idx = 0
    query_latent = analyzer.test_latents[query_idx]
    query_label = analyzer.test_labels[query_idx]

    print(f"Query sample: {'Healthy' if query_label == 0 else 'Unhealthy'}")

    # Find nearest neighbors (same class)
    indices, distances = analyzer.find_nearest_neighbors(
        query_latent,
        k=5,
        from_label=query_label
    )

    print(f"\nNearest neighbors (same class):")
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        print(f"  {i+1}. Index {idx}, Distance: {dist:.4f}")

    # Find nearest neighbors (opposite class)
    opposite_label = 1 - query_label
    indices_opp, distances_opp = analyzer.find_nearest_neighbors(
        query_latent,
        k=5,
        from_label=opposite_label
    )

    print(f"\nNearest neighbors (opposite class):")
    for i, (idx, dist) in enumerate(zip(indices_opp, distances_opp)):
        print(f"  {i+1}. Index {idx}, Distance: {dist:.4f}")

    print(f"\nObservations:")
    print(f"  Same-class distance range: {distances[0]:.4f} - {distances[-1]:.4f}")
    print(f"  Opposite-class distance range: {distances_opp[0]:.4f} - {distances_opp[-1]:.4f}")
    if distances_opp[0] > distances[-1]:
        print(f"  ✓ Good separation: opposite class is farther than same class")
    else:
        print(f"  ✗ Poor separation: opposite class overlaps with same class")


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("DRAC Classifier - Complete Workflow Examples")
    print("="*80)
    print("\nThis script demonstrates the complete pipeline.")
    print("Make sure you have:")
    print("1. Trained a model using train_drac_classifier.py")
    print("2. Generated latent representations")
    print("\n")

    try:
        example_inference()
        example_latent_analysis()
        example_perturbation()
        example_interpolation()
        example_nearest_neighbors()

        print("\n" + "="*80)
        print("Examples Complete!")
        print("="*80)
        print("\nNext steps:")
        print("1. Visualize latent space with: python latent_analysis.py")
        print("2. Implement graph decoder to visualize perturbations")
        print("3. Perform counterfactual analysis")
        print("4. Identify disease-critical graph features")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have trained a model first!")


if __name__ == '__main__':
    main()
