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
