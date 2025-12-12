# Image Color Clusterer

A powerful tool for extracting and clustering colors from images using k-means algorithm. Reduce any image to its most dominant colors or create beautiful color palettes.

## Features

- **Color Extraction**: Automatically extract colors from any image format
- **K-means Clustering**: Uses scikit-learn's robust k-means implementation
- **Color Reduction**: Reduce images to use only the dominant colors
- **Visualization**: Generate beautiful color palette visualizations
- **Command-line Interface**: Easy-to-use CLI with multiple output options
- **Reproducible Results**: Configurable random seed for consistent results

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Basic usage - extract 8 colors and show info
python cluster_colors.py your_image.jpg

# Extract 16 colors and save palette visualization
python cluster_colors.py your_image.jpg --colors 16 --output-palette palette.png

# Reduce image to 4 colors (auto-saves as your_image_clusterized.jpg)
python cluster_colors.py your_image.jpg --colors 4

# Reduce image to 4 colors with custom filename
python cluster_colors.py your_image.jpg --colors 4 --reduce-colors custom_reduced.jpg

# Get detailed color information as text
python cluster_colors.py your_image.jpg --text

# Create comprehensive color visualizations
python cluster_colors.py photo.jpg --colors 8 --visualize-distribution color_space.png --visualize-clustering clusters.png --visualize-projections projections.png

# Create 2D cluster visualization (PCA projection for easier viewing)
python cluster_colors.py photo.jpg --colors 6 --visualize-clusters-2d clusters_2d.png

# Combine options
python cluster_colors.py photo.jpg --colors 12 --output-palette colors.png --reduce-colors posterized.jpg --text --visualize-clusters-2d analysis.png
```

### Python API Usage

```python
from color_clusterer import ColorClusterer, cluster_image_colors

# Simple usage
clusterer = cluster_image_colors('image.jpg', n_colors=8)
palette = clusterer.get_palette()

# Advanced usage
clusterer = ColorClusterer(n_colors=16, random_state=42)
pixels = clusterer.load_image('image.jpg')
clusterer.fit(pixels)

# Get dominant colors with frequencies
dominant_colors = clusterer.get_dominant_colors()

# Visualize palette
clusterer.visualize_palette(save_path='palette.png')

# Reduce image colors
clusterer.reduce_image_colors(output_path='reduced_image.jpg')
```

## Command Line Options

- `image_path`: Path to the image file (required)
- `--colors, -c`: Number of colors to cluster into (default: 8)
- `--output-palette, -p`: Save color palette visualization to file
- `--reduce-colors, -r`: Save clusterized image to specified file (default: auto-generate with _clusterized suffix)
- `--visualize-distribution, -vd`: Save 3D color distribution visualization
- `--visualize-clustering, -vc`: Save 3D clustering results visualization
- `--visualize-projections, -vp`: Save 2D color projection visualizations
- `--text, -t`: Output colors as text with hex codes and frequencies
- `--random-state, -s`: Random seed for reproducible results (default: 42)
- `--no-save`: Don't save the clusterized image (only show analysis)

## Examples

### Extract Color Palette from a Photo

```bash
python cluster_colors.py sunset.jpg --colors 6 --output-palette sunset_palette.png --text
```

This will:
- Analyze `sunset.jpg`
- Extract 6 dominant colors
- Save a visual palette to `sunset_palette.png`
- Print color information to terminal

### Create a Posterized Effect

```bash
python cluster_colors.py photo.jpg --colors 4 --reduce-colors posterized.jpg
```

This reduces the image to use only 4 colors, creating a poster-like effect.

### Generate Color Scheme for Design

```bash
python cluster_colors.py artwork.jpg --colors 5 --text > color_scheme.txt
```

Extract 5 colors and save them to a text file for use in design projects.

## Supported Image Formats

The tool supports all image formats that PIL (Pillow) can read, including:
- JPEG/JPG
- PNG
- BMP
- TIFF
- GIF
- WebP
- And many more...

## Algorithm Details

The color clustering uses k-means algorithm on the RGB color space:

1. **Color Extraction**: Each pixel becomes a 3D point (R, G, B)
2. **Clustering**: K-means groups similar colors together
3. **Centroid Calculation**: Each cluster's center becomes the representative color
4. **Color Reduction**: Pixels are reassigned to their nearest cluster center

## Visualization Features

### Color Palette (`--output-palette`)
Horizontal bar showing the dominant colors as a palette with hex codes.

### 3D Color Distribution (`--visualize-distribution`)
- **3D scatter plot** of original image pixels in RGB color space
- **Color-coded points** show the natural distribution of colors
- **Helps understand** the color space coverage of your image

### 3D Clustering Results (`--visualize-clustering`)
- **Left plot**: 3D scatter plot showing clustered pixels with color coding
- **Right plot**: Extracted color palette
- **Shows how** the k-means algorithm grouped colors together
- **Legend** indicates which cluster each color belongs to

### 2D Color Projections (`--visualize-projections`)
- **Three 2D plots** showing color projections (RG, GB, BR planes)
- **X marks** show cluster centers on each projection
- **Helps visualize** color relationships in different dimensions
- **Color-coded clusters** make grouping patterns clear

These visualizations help you understand:
- How colors are distributed in your image
- Which colors dominate the composition
- How the clustering algorithm groups similar colors
- The spatial relationships between color clusters
- The effectiveness of your chosen number of colors

## Tips

- **Color Count**: Start with 8-16 colors for most images. More colors = more detail, fewer = more dramatic effect
- **Large Images**: The algorithm works on all pixels, so very large images may take longer to process
- **Reproducibility**: Use the same `--random-state` value for consistent results across runs
- **Quality**: PNG format preserves quality better than JPEG for reduced color images

## Dependencies

- Pillow: Image processing
- scikit-learn: K-means clustering
- numpy: Numerical operations
- matplotlib: Visualization
- click: Command-line interface

## License

This project is open source. Feel free to use and modify as needed.
