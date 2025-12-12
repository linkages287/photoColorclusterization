"""
Image Color Clusterer

A tool for extracting color palettes from images using k-means clustering.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ColorClusterer:
    """A class for clustering colors in images using k-means algorithm."""

    def __init__(self, n_colors: int = 8, random_state: int = 42):
        """
        Initialize the color clusterer.

        Args:
            n_colors: Number of colors to cluster into
            random_state: Random seed for reproducible results
        """
        self.n_colors = n_colors
        self.random_state = random_state
        self.kmeans = None
        self.palette = None
        self.pixels = None

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image.

        Args:
            image_path: Path to the image file

        Returns:
            Numpy array of pixel colors (N, 3) where N is number of pixels
        """
        # Load image
        img = Image.open(image_path)

        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Get image dimensions
        width, height = img.size

        # Convert to numpy array and reshape to (N, 3)
        pixels = np.array(img).reshape(-1, 3)

        # Store for later use
        self.pixels = pixels
        self.image_size = (width, height)

        return pixels

    def fit(self, pixels: Optional[np.ndarray] = None) -> 'ColorClusterer':
        """
        Fit the k-means clustering model to the pixel data.

        Args:
            pixels: Pixel data as (N, 3) array. If None, uses previously loaded data.

        Returns:
            Self for method chaining
        """
        if pixels is None:
            if self.pixels is None:
                raise ValueError("No pixel data available. Load an image first or provide pixels.")
            pixels = self.pixels

        # Perform k-means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_colors,
            random_state=self.random_state,
            n_init=10
        )

        # Fit the model
        self.kmeans.fit(pixels)

        # Get the cluster centers (these are our color palette)
        self.palette = self.kmeans.cluster_centers_.astype(int)

        return self

    def predict(self, pixels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict cluster labels for pixels.

        Args:
            pixels: Pixel data to predict. If None, uses training data.

        Returns:
            Array of cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if pixels is None:
            pixels = self.pixels

        return self.kmeans.predict(pixels)

    def get_palette(self) -> np.ndarray:
        """
        Get the extracted color palette.

        Returns:
            Array of RGB colors (n_colors, 3)
        """
        if self.palette is None:
            raise ValueError("Palette not available. Call fit() first.")
        return self.palette

    def get_dominant_colors(self, sort_by_frequency: bool = True) -> List[Tuple[Tuple[int, int, int], int]]:
        """
        Get dominant colors with their frequencies.

        Args:
            sort_by_frequency: Whether to sort colors by frequency (descending)

        Returns:
            List of tuples: (RGB tuple, frequency)
        """
        if self.kmeans is None or self.pixels is None:
            raise ValueError("Model not fitted or no pixel data available.")

        # Get cluster labels for all pixels
        labels = self.predict()

        # Count frequency of each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Create list of (color, frequency) tuples
        color_freq = []
        for label, count in zip(unique_labels, counts):
            color = tuple(self.palette[label])
            color_freq.append((color, count))

        # Sort by frequency if requested
        if sort_by_frequency:
            color_freq.sort(key=lambda x: x[1], reverse=True)

        return color_freq

    def reduce_image_colors(self, image_path: str = None, output_path: str = None) -> Optional[Image.Image]:
        """
        Reduce the colors in an image using the learned palette.

        Args:
            image_path: Path to image to reduce. If None, uses loaded image.
            output_path: Path to save reduced image. If None, returns PIL Image.

        Returns:
            PIL Image with reduced colors if output_path is None, else None
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Load image if path provided
        if image_path:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            # Reconstruct from pixels
            img = Image.fromarray(self.pixels.reshape(self.image_size[1], self.image_size[0], 3), 'RGB')

        # Convert to numpy array
        pixels = np.array(img)

        # Reshape for clustering
        original_shape = pixels.shape
        pixels_flat = pixels.reshape(-1, 3)

        # Predict clusters
        labels = self.predict(pixels_flat)

        # Replace each pixel with its cluster center
        reduced_pixels = self.palette[labels].reshape(original_shape).astype(np.uint8)

        # Convert back to PIL Image
        reduced_img = Image.fromarray(reduced_pixels, 'RGB')

        if output_path:
            reduced_img.save(output_path)
            return None
        else:
            return reduced_img

    def visualize_palette(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 2)):
        """
        Visualize the color palette.

        Args:
            save_path: Path to save the visualization. If None, displays it.
            figsize: Figure size for matplotlib
        """
        if self.palette is None:
            raise ValueError("Palette not available. Call fit() first.")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow([self.palette], aspect='auto')
        ax.set_xticks(range(len(self.palette)))
        ax.set_xticklabels([f'#{r:02x}{g:02x}{b:02x}' for r, g, b in self.palette])
        ax.set_yticks([])
        ax.set_title(f'Color Palette ({len(self.palette)} colors)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_color_distribution(self, save_path: Optional[str] = None, max_samples: int = 10000):
        """
        Visualize the original color distribution in 3D RGB space.

        Args:
            save_path: Path to save the visualization. If None, displays it.
            max_samples: Maximum number of pixels to sample for visualization
        """
        if self.pixels is None:
            raise ValueError("No pixel data available. Load an image first.")

        # Sample pixels for visualization
        pixels = self.pixels
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            pixels = pixels[indices]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original color distribution
        ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2],
                  c=pixels/255.0, s=1, alpha=0.6, edgecolors='none')

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title(f'Original Color Distribution ({len(pixels)} samples)')
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()

    def visualize_clustering(self, save_path: Optional[str] = None, max_samples: int = 10000):
        """
        Visualize the color clustering results in 3D RGB space.

        Args:
            save_path: Path to save the visualization. If None, displays it.
            max_samples: Maximum number of pixels to sample for visualization
        """
        if self.kmeans is None or self.pixels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Sample pixels for visualization
        pixels = self.pixels
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            pixels = pixels[indices]

        # Get cluster labels for sampled pixels
        labels = self.kmeans.predict(pixels)

        fig = plt.figure(figsize=(15, 8))

        # Original color distribution
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2],
                              c=labels, cmap='tab10', s=1, alpha=0.6, edgecolors='none')
        ax1.set_xlabel('Red')
        ax1.set_ylabel('Green')
        ax1.set_zlabel('Blue')
        ax1.set_title('Color Clustering Results')
        ax1.set_xlim(0, 255)
        ax1.set_ylim(0, 255)
        ax1.set_zlim(0, 255)

        # Color palette with cluster centers
        ax2 = fig.add_subplot(122)
        # Show palette as before
        ax2.imshow([self.palette], aspect='auto')
        ax2.set_xticks(range(len(self.palette)))
        ax2.set_xticklabels([f'#{r:02x}{g:02x}{b:02x}' for r, g, b in self.palette])
        ax2.set_yticks([])
        ax2.set_title(f'Extracted Color Palette ({len(self.palette)} colors)')

        # Add legend to 3D plot
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=scatter1.cmap(scatter1.norm(i)),
                                    markersize=10, label=f'Cluster {i+1}')
                          for i in range(len(self.palette))]
        ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()

    def visualize_color_projections(self, save_path: Optional[str] = None, max_samples: int = 5000):
        """
        Visualize color projections in 2D planes (RG, GB, BR).

        Args:
            save_path: Path to save the visualization. If None, displays it.
            max_samples: Maximum number of pixels to sample for visualization
        """
        if self.kmeans is None or self.pixels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Sample pixels for visualization
        pixels = self.pixels
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            pixels = pixels[indices]

        # Get cluster labels for sampled pixels
        labels = self.kmeans.predict(pixels)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # RG plane (Red-Green)
        scatter1 = axes[0].scatter(pixels[:, 0], pixels[:, 1], c=labels,
                                  cmap='tab10', s=2, alpha=0.7, edgecolors='none')
        axes[0].set_xlabel('Red')
        axes[0].set_ylabel('Green')
        axes[0].set_title('Red-Green Projection')
        axes[0].set_xlim(0, 255)
        axes[0].set_ylim(0, 255)

        # GB plane (Green-Blue)
        axes[1].scatter(pixels[:, 1], pixels[:, 2], c=labels,
                       cmap='tab10', s=2, alpha=0.7, edgecolors='none')
        axes[1].set_xlabel('Green')
        axes[1].set_ylabel('Blue')
        axes[1].set_title('Green-Blue Projection')
        axes[1].set_xlim(0, 255)
        axes[1].set_ylim(0, 255)

        # BR plane (Blue-Red)
        axes[2].scatter(pixels[:, 2], pixels[:, 0], c=labels,
                       cmap='tab10', s=2, alpha=0.7, edgecolors='none')
        axes[2].set_xlabel('Blue')
        axes[2].set_ylabel('Red')
        axes[2].set_title('Blue-Red Projection')
        axes[2].set_xlim(0, 255)
        axes[2].set_ylim(0, 255)

        # Add cluster centers to each plot
        for i, center in enumerate(self.palette):
            # RG
            axes[0].scatter(center[0], center[1], c=[scatter1.cmap(scatter1.norm(i))],
                           s=100, marker='x', linewidth=3, edgecolors='black')
            # GB
            axes[1].scatter(center[1], center[2], c=[scatter1.cmap(scatter1.norm(i))],
                           s=100, marker='x', linewidth=3, edgecolors='black')
            # BR
            axes[2].scatter(center[2], center[0], c=[scatter1.cmap(scatter1.norm(i))],
                           s=100, marker='x', linewidth=3, edgecolors='black')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()

    def visualize_clusters(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8), max_points: int = 5000):
        """
        Visualize color clusters in 3D RGB space.

        Shows original pixels as points and cluster centers as larger markers.

        Args:
            save_path: Path to save the visualization. If None, displays it.
            figsize: Figure size for matplotlib
            max_points: Maximum number of pixels to plot for performance
        """
        if self.kmeans is None or self.pixels is None:
            raise ValueError("Clusters not available. Call fit() first.")

        # Sample pixels for visualization (too many points slow down plotting)
        if len(self.pixels) > max_points:
            indices = np.random.choice(len(self.pixels), max_points, replace=False)
            pixels_sample = self.pixels[indices]
        else:
            pixels_sample = self.pixels

        # Get cluster labels for sampled pixels
        labels_sample = self.kmeans.predict(pixels_sample)

        # Create 3D scatter plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot original pixels colored by their cluster
        scatter = ax.scatter(
            pixels_sample[:, 0], pixels_sample[:, 1], pixels_sample[:, 2],
            c=labels_sample, cmap='tab10', alpha=0.6, s=20, edgecolors='none'
        )

        # Plot cluster centers as larger stars
        ax.scatter(
            self.palette[:, 0], self.palette[:, 1], self.palette[:, 2],
            c=range(len(self.palette)), cmap='tab10',
            marker='*', s=300, edgecolors='black', linewidth=2
        )

        # Add cluster center labels
        for i, center in enumerate(self.palette):
            ax.text(
                center[0], center[1], center[2],
                f'C{i+1}\n#{center[0]:02x}{center[1]:02x}{center[2]:02x}',
                fontsize=8, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title(f'Color Clusters in RGB Space ({len(self.palette)} clusters)')

        # Set axis limits
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()

    def visualize_clusters_2d(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8), max_points: int = 5000):
        """
        Visualize color clusters in 2D using PCA for dimensionality reduction.

        Shows original pixels and cluster centers in 2D space.

        Args:
            save_path: Path to save the visualization. If None, displays it.
            figsize: Figure size for matplotlib
            max_points: Maximum number of pixels to plot for performance
        """
        if self.kmeans is None or self.pixels is None:
            raise ValueError("Clusters not available. Call fit() first.")

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("PCA visualization requires scikit-learn. Install with: pip install scikit-learn")

        # Sample pixels for visualization
        if len(self.pixels) > max_points:
            indices = np.random.choice(len(self.pixels), max_points, replace=False)
            pixels_sample = self.pixels[indices]
        else:
            pixels_sample = self.pixels

        # Get cluster labels for sampled pixels
        labels_sample = self.kmeans.predict(pixels_sample)

        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        pixels_2d = pca.fit_transform(pixels_sample)
        centers_2d = pca.transform(self.palette)

        # Create 2D scatter plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot original pixels colored by their cluster
        scatter = ax.scatter(
            pixels_2d[:, 0], pixels_2d[:, 1],
            c=labels_sample, cmap='tab10', alpha=0.6, s=20, edgecolors='none'
        )

        # Plot cluster centers as larger stars
        ax.scatter(
            centers_2d[:, 0], centers_2d[:, 1],
            c=range(len(self.palette)), cmap='tab10',
            marker='*', s=300, edgecolors='black', linewidth=2
        )

        # Add cluster center labels
        for i, (center_2d, center_rgb) in enumerate(zip(centers_2d, self.palette)):
            ax.annotate(
                f'C{i+1}\n#{center_rgb[0]:02x}{center_rgb[1]:02x}{center_rgb[2]:02x}',
                (center_2d[0], center_2d[1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Color Clusters (PCA 2D projection, {len(self.palette)} clusters)')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Cluster')
        cbar.set_ticks(range(len(self.palette)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()


def cluster_image_colors(image_path: str, n_colors: int = 8, random_state: int = 42) -> ColorClusterer:
    """
    Convenience function to cluster colors in an image.

    Args:
        image_path: Path to the image
        n_colors: Number of colors to cluster into
        random_state: Random seed

    Returns:
        Fitted ColorClusterer instance
    """
    clusterer = ColorClusterer(n_colors=n_colors, random_state=random_state)
    pixels = clusterer.load_image(image_path)
    clusterer.fit(pixels)
    return clusterer


def visualize_interactive_3d(self, save_path: Optional[str] = None, max_samples: int = 5000, show_palette: bool = True):
    """
    Create an interactive 3D visualization of color distribution and clustering.

    Args:
        save_path: Path to save the interactive HTML file. If None, displays it.
        max_samples: Maximum number of pixels to sample for visualization
        show_palette: Whether to show the color palette alongside the 3D plot
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")

    if self.kmeans is None or self.pixels is None:
        raise ValueError("Model not fitted. Call fit() first.")

    # Sample pixels for visualization
    pixels = self.pixels
    if len(pixels) > max_samples:
        indices = np.random.choice(len(pixels), max_samples, replace=False)
        pixels = pixels[indices]

    # Get cluster labels for sampled pixels
    labels = self.kmeans.predict(pixels)

    # Create the main 3D scatter plot
    fig = go.Figure()

    # Add clustered points
    for cluster_id in range(len(self.palette)):
        cluster_pixels = pixels[labels == cluster_id]
        if len(cluster_pixels) > 0:
            cluster_color = self.palette[cluster_id]
            hex_color = f'rgb({cluster_color[0]}, {cluster_color[1]}, {cluster_color[2]})'

            fig.add_trace(go.Scatter3d(
                x=cluster_pixels[:, 0],
                y=cluster_pixels[:, 1],
                z=cluster_pixels[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=hex_color,
                    opacity=0.7,
                    line=dict(width=0)
                ),
                name=f'Cluster {cluster_id + 1}',
                hovertemplate=f'Cluster {cluster_id + 1}<br>R: %{{x}}<br>G: %{{y}}<br>B: %{{z}}<extra></extra>'
            ))

    # Add cluster centers as larger markers
    center_colors = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in self.palette]
    fig.add_trace(go.Scatter3d(
        x=self.palette[:, 0],
        y=self.palette[:, 1],
        z=self.palette[:, 2],
        mode='markers',
        marker=dict(
            size=12,
            color=center_colors,
            symbol='diamond',
            line=dict(width=2, color='black'),
            opacity=1.0
        ),
        name='Cluster Centers',
        hovertemplate=[
            f'Center<br>R: {c[0]}<br>G: {c[1]}<br>B: {c[2]}<br>Hex: #{c[0]:02x}{c[1]:02x}{c[2]:02x}<extra></extra>'
            for c in self.palette
        ]
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            xaxis=dict(range=[0, 255]),
            yaxis=dict(range=[0, 255]),
            zaxis=dict(range=[0, 255]),
            aspectmode='cube'
        ),
        title=dict(
            text=f'Interactive 3D Color Clustering ({len(self.palette)} colors, {len(pixels)} samples)',
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Add color palette as annotations if requested
    if show_palette:
        # Add palette colors as text annotations
        palette_text = '<br>'.join([f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in self.palette])

        fig.add_annotation(
            text=f"<b>Color Palette:</b><br>{palette_text}",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            font=dict(size=10)
        )

    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        fig.write_html(save_path)
        print(f"Interactive visualization saved to: {save_path}")
        print("Open this file in any web browser to interact with the 3D visualization.")
    else:
        fig.show()


def visualize_interactive_distribution(self, save_path: Optional[str] = None, max_samples: int = 5000):
    """
    Create an interactive 3D visualization of the original color distribution.

    Args:
        save_path: Path to save the interactive HTML file. If None, displays it.
        max_samples: Maximum number of pixels to sample for visualization
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")

    if self.pixels is None:
        raise ValueError("No pixel data available. Load an image first.")

    # Sample pixels for visualization
    pixels = self.pixels
    if len(pixels) > max_samples:
        indices = np.random.choice(len(pixels), max_samples, replace=False)
        pixels = pixels[indices]

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=pixels[:, 0],
        y=pixels[:, 1],
        z=pixels[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=[f'rgb({int(r)}, {int(g)}, {int(b)})' for r, g, b in pixels],
            opacity=0.6,
            line=dict(width=0)
        ),
        hovertemplate='R: %{x}<br>G: %{y}<br>B: %{z}<extra></extra>'
    )])

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            xaxis=dict(range=[0, 255]),
            yaxis=dict(range=[0, 255]),
            zaxis=dict(range=[0, 255]),
            aspectmode='cube'
        ),
        title=dict(
            text=f'Interactive 3D Color Distribution ({len(pixels)} samples)',
            x=0.5,
            font=dict(size=16)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        fig.write_html(save_path)
        print(f"Interactive visualization saved to: {save_path}")
        print("Open this file in any web browser to interact with the 3D visualization.")
    else:
        fig.show()


# Add the methods to the ColorClusterer class
ColorClusterer.visualize_interactive_3d = visualize_interactive_3d
ColorClusterer.visualize_interactive_distribution = visualize_interactive_distribution
