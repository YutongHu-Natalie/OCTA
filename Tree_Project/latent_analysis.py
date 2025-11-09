"""
Latent Space Analysis and Perturbation Tools
For understanding and visualizing learned representations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from scipy.stats import ttest_ind


class LatentSpaceAnalyzer:
    """
    Tools for analyzing and perturbing latent representations
    """

    def __init__(self, latents_file):
        """
        Load latent representations from saved file

        Args:
            latents_file: Path to .npz file with latent representations
        """
        data = np.load(latents_file)

        self.train_latents = data['train_latents']
        self.train_labels = data['train_labels']
        self.val_latents = data['val_latents']
        self.val_labels = data['val_labels']
        self.test_latents = data['test_latents']
        self.test_labels = data['test_labels']

        # Combine all for analysis
        self.all_latents = np.vstack([self.train_latents, self.val_latents, self.test_latents])
        self.all_labels = np.concatenate([self.train_labels, self.val_labels, self.test_labels])

        print(f"Loaded latent representations:")
        print(f"  Train: {self.train_latents.shape}")
        print(f"  Val: {self.val_latents.shape}")
        print(f"  Test: {self.test_latents.shape}")
        print(f"  Latent dimension: {self.train_latents.shape[1]}")

    def visualize_latent_space(self, method='pca', save_path=None):
        """
        Visualize latent space in 2D using PCA or t-SNE

        Args:
            method: 'pca' or 'tsne'
            save_path: Path to save figure
        """
        print(f"\nVisualizing latent space using {method.upper()}...")

        if method == 'pca':
            reducer = PCA(n_components=2)
            embeddings = reducer.fit_transform(self.all_latents)
            explained_var = reducer.explained_variance_ratio_
            title = f'Latent Space (PCA: {explained_var[0]:.1%} + {explained_var[1]:.1%} variance)'
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(self.all_latents)
            title = 'Latent Space (t-SNE)'
        else:
            raise ValueError("method must be 'pca' or 'tsne'")

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                            c=self.all_labels, cmap='RdYlGn_r',
                            alpha=0.6, s=50)
        plt.colorbar(scatter, label='Label (0=Healthy, 1=Unhealthy)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.show()

        return embeddings

    def identify_discriminative_dimensions(self, alpha=0.05, top_k=10):
        """
        Identify which latent dimensions are most discriminative
        between healthy and unhealthy using t-tests

        Args:
            alpha: Significance level for t-test
            top_k: Number of top dimensions to return

        Returns:
            Dictionary with discriminative dimension analysis
        """
        print("\nIdentifying discriminative latent dimensions...")

        healthy_latents = self.all_latents[self.all_labels == 0]
        unhealthy_latents = self.all_latents[self.all_labels == 1]

        n_dims = self.all_latents.shape[1]
        results = []

        for dim in range(n_dims):
            healthy_vals = healthy_latents[:, dim]
            unhealthy_vals = unhealthy_latents[:, dim]

            # T-test
            t_stat, p_value = ttest_ind(healthy_vals, unhealthy_vals)

            # Effect size (Cohen's d)
            mean_diff = np.mean(unhealthy_vals) - np.mean(healthy_vals)
            pooled_std = np.sqrt((np.std(healthy_vals)**2 + np.std(unhealthy_vals)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            results.append({
                'dimension': dim,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'mean_healthy': np.mean(healthy_vals),
                'mean_unhealthy': np.mean(unhealthy_vals),
                'std_healthy': np.std(healthy_vals),
                'std_unhealthy': np.std(unhealthy_vals)
            })

        # Sort by absolute effect size
        results = sorted(results, key=lambda x: abs(x['cohens_d']), reverse=True)

        # Print top dimensions
        print(f"\nTop {top_k} most discriminative dimensions:")
        print(f"{'Dim':<5} {'p-value':<10} {'Cohen\'s d':<12} {'Mean (H)':<12} {'Mean (UH)':<12}")
        print("-" * 60)

        for i, res in enumerate(results[:top_k]):
            sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < alpha else ""
            print(f"{res['dimension']:<5} {res['p_value']:<10.4f} {res['cohens_d']:<12.3f} "
                  f"{res['mean_healthy']:<12.3f} {res['mean_unhealthy']:<12.3f} {sig}")

        return results

    def plot_discriminative_dimensions(self, results, top_k=5, save_path=None):
        """
        Plot distributions of top discriminative dimensions

        Args:
            results: Output from identify_discriminative_dimensions()
            top_k: Number of top dimensions to plot
            save_path: Path to save figure
        """
        top_dims = [r['dimension'] for r in results[:top_k]]

        healthy_latents = self.all_latents[self.all_labels == 0]
        unhealthy_latents = self.all_latents[self.all_labels == 1]

        fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 4))
        if top_k == 1:
            axes = [axes]

        for i, dim in enumerate(top_dims):
            ax = axes[i]

            ax.hist(healthy_latents[:, dim], bins=30, alpha=0.5,
                   label='Healthy', color='green', density=True)
            ax.hist(unhealthy_latents[:, dim], bins=30, alpha=0.5,
                   label='Unhealthy', color='red', density=True)

            ax.set_xlabel(f'Dimension {dim} value')
            ax.set_ylabel('Density')
            ax.set_title(f"Dim {dim}\n(Cohen's d={results[i]['cohens_d']:.2f})")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.show()

    def create_perturbation_vectors(self, dimension, num_steps=10, scale=3.0):
        """
        Create perturbation vectors along a specific latent dimension

        Args:
            dimension: Which latent dimension to perturb
            num_steps: Number of perturbation steps
            scale: How many standard deviations to perturb

        Returns:
            Perturbation vectors
        """
        std = np.std(self.all_latents[:, dimension])
        perturbations = np.linspace(-scale * std, scale * std, num_steps)

        return perturbations

    def perturb_latent(self, base_latent, dimension, amount):
        """
        Perturb a latent vector along a specific dimension

        Args:
            base_latent: Original latent vector (shape: [latent_dim])
            dimension: Which dimension to perturb
            amount: How much to perturb (in original scale)

        Returns:
            Perturbed latent vector
        """
        perturbed = base_latent.copy()
        perturbed[dimension] += amount
        return perturbed

    def interpolate_between_samples(self, latent1, latent2, num_steps=10):
        """
        Linear interpolation between two latent vectors

        Args:
            latent1: First latent vector
            latent2: Second latent vector
            num_steps: Number of interpolation steps

        Returns:
            Array of interpolated latent vectors
        """
        alphas = np.linspace(0, 1, num_steps)
        interpolations = []

        for alpha in alphas:
            interpolated = (1 - alpha) * latent1 + alpha * latent2
            interpolations.append(interpolated)

        return np.array(interpolations)

    def find_nearest_neighbors(self, query_latent, k=5, from_label=None):
        """
        Find k nearest neighbors to a query latent vector

        Args:
            query_latent: Query latent vector
            k: Number of neighbors
            from_label: If specified, only search within this label class

        Returns:
            Indices and distances of nearest neighbors
        """
        if from_label is not None:
            mask = self.all_labels == from_label
            search_latents = self.all_latents[mask]
            search_indices = np.where(mask)[0]
        else:
            search_latents = self.all_latents
            search_indices = np.arange(len(self.all_latents))

        # Compute distances
        distances = np.linalg.norm(search_latents - query_latent, axis=1)

        # Get k nearest
        nearest_idx = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_idx]
        nearest_global_idx = search_indices[nearest_idx]

        return nearest_global_idx, nearest_distances

    def compute_healthy_unhealthy_direction(self):
        """
        Compute the average direction from healthy to unhealthy in latent space

        Returns:
            Direction vector (unit vector)
        """
        healthy_mean = np.mean(self.all_latents[self.all_labels == 0], axis=0)
        unhealthy_mean = np.mean(self.all_latents[self.all_labels == 1], axis=0)

        direction = unhealthy_mean - healthy_mean
        direction_normalized = direction / np.linalg.norm(direction)

        print("\nHealthy → Unhealthy direction computed")
        print(f"Direction magnitude: {np.linalg.norm(direction):.3f}")

        return direction_normalized, healthy_mean, unhealthy_mean

    def visualize_latent_statistics(self, save_path=None):
        """
        Visualize statistics of latent dimensions

        Args:
            save_path: Path to save figure
        """
        n_dims = self.all_latents.shape[1]

        # Compute statistics per dimension
        healthy_latents = self.all_latents[self.all_labels == 0]
        unhealthy_latents = self.all_latents[self.all_labels == 1]

        healthy_means = np.mean(healthy_latents, axis=0)
        unhealthy_means = np.mean(unhealthy_latents, axis=0)
        healthy_stds = np.std(healthy_latents, axis=0)
        unhealthy_stds = np.std(unhealthy_latents, axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Mean values
        axes[0, 0].plot(healthy_means, 'g.-', label='Healthy', alpha=0.7)
        axes[0, 0].plot(unhealthy_means, 'r.-', label='Unhealthy', alpha=0.7)
        axes[0, 0].set_xlabel('Latent Dimension')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].set_title('Mean Values per Dimension')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Standard deviations
        axes[0, 1].plot(healthy_stds, 'g.-', label='Healthy', alpha=0.7)
        axes[0, 1].plot(unhealthy_stds, 'r.-', label='Unhealthy', alpha=0.7)
        axes[0, 1].set_xlabel('Latent Dimension')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_title('Standard Deviations per Dimension')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Mean difference
        mean_diff = unhealthy_means - healthy_means
        axes[1, 0].bar(range(n_dims), mean_diff, color=['red' if x > 0 else 'green' for x in mean_diff])
        axes[1, 0].set_xlabel('Latent Dimension')
        axes[1, 0].set_ylabel('Mean Difference (Unhealthy - Healthy)')
        axes[1, 0].set_title('Mean Difference per Dimension')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # Correlation between dimensions
        corr_matrix = np.corrcoef(self.all_latents.T)
        im = axes[1, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 1].set_xlabel('Latent Dimension')
        axes[1, 1].set_ylabel('Latent Dimension')
        axes[1, 1].set_title('Correlation Between Dimensions')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.show()


def main():
    """Example usage of latent space analysis tools"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze latent space')
    parser.add_argument('--latents_file', type=str, required=True,
                       help='Path to latent_representations.npz file')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                       help='Directory to save analysis results')
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and analyze
    analyzer = LatentSpaceAnalyzer(args.latents_file)

    # 1. Visualize latent space
    print("\n" + "="*80)
    print("1. Visualizing Latent Space")
    print("="*80)
    analyzer.visualize_latent_space(method='pca',
                                    save_path=os.path.join(args.output_dir, 'latent_space_pca.png'))
    analyzer.visualize_latent_space(method='tsne',
                                    save_path=os.path.join(args.output_dir, 'latent_space_tsne.png'))

    # 2. Identify discriminative dimensions
    print("\n" + "="*80)
    print("2. Identifying Discriminative Dimensions")
    print("="*80)
    results = analyzer.identify_discriminative_dimensions(top_k=10)

    # 3. Plot discriminative dimensions
    print("\n" + "="*80)
    print("3. Plotting Discriminative Dimensions")
    print("="*80)
    analyzer.plot_discriminative_dimensions(results, top_k=5,
                                           save_path=os.path.join(args.output_dir, 'discriminative_dims.png'))

    # 4. Visualize latent statistics
    print("\n" + "="*80)
    print("4. Visualizing Latent Statistics")
    print("="*80)
    analyzer.visualize_latent_statistics(save_path=os.path.join(args.output_dir, 'latent_statistics.png'))

    # 5. Compute healthy-unhealthy direction
    print("\n" + "="*80)
    print("5. Computing Healthy → Unhealthy Direction")
    print("="*80)
    direction, healthy_mean, unhealthy_mean = analyzer.compute_healthy_unhealthy_direction()

    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
