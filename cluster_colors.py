#!/usr/bin/env python3
"""
Command-line interface for the Image Color Clusterer.

Usage:
    python cluster_colors.py image.jpg --colors 8 --output-palette palette.png
    python cluster_colors.py image.jpg --colors 6 --visualize-clusters-3d clusters_3d.png
    # Image is automatically saved as image_clusterized.jpg

Examples:
    # Extract 8 colors and show them
    python cluster_colors.py my_image.jpg

    # Extract 16 colors and save palette visualization
    python cluster_colors.py my_image.jpg --colors 16 --output-palette colors.png

    # Reduce image to 4 colors and save
    python cluster_colors.py my_image.jpg --colors 4 --reduce-colors reduced_image.jpg

    # Get colors as text output
    python cluster_colors.py my_image.jpg --text
"""

import click
import os
import sys
from pathlib import Path
from color_clusterer import cluster_image_colors


@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--colors', '-c', default=8, type=int, help='Number of colors to cluster into (default: 8)')
@click.option('--output-palette', '-p', type=click.Path(), help='Save color palette visualization to file')
@click.option('--reduce-colors', '-r', type=click.Path(), help='Save clusterized image to specified file (default: auto-generate with _clusterized suffix)')
@click.option('--text', '-t', is_flag=True, help='Output colors as text (hex codes and frequencies)')
@click.option('--random-state', '-s', default=42, type=int, help='Random seed for reproducible results')
@click.option('--no-save', is_flag=True, help="Don't save the clusterized image (only show analysis)")
@click.option('--visualize-clusters-3d', '-3', type=click.Path(), help='Save 3D RGB cluster visualization to file')
@click.option('--visualize-clusters-2d', '-2', type=click.Path(), help='Save 2D PCA cluster visualization to file')
def cluster_colors(image_path, colors, output_palette, reduce_colors, text, random_state, no_save, visualize_clusters_3d, visualize_clusters_2d):
    """
    Cluster colors in an image using k-means algorithm.

    IMAGE_PATH: Path to the image file to analyze
    """
    try:
        # Validate inputs
        if colors < 1:
            click.echo("Error: Number of colors must be at least 1", err=True)
            sys.exit(1)

        # Check if image file exists and is readable
        image_path = Path(image_path)
        if not image_path.exists():
            click.echo(f"Error: Image file '{image_path}' does not exist", err=True)
            sys.exit(1)

        click.echo(f"Analyzing image: {image_path}")
        click.echo(f"Clustering into {colors} colors...")

        # Perform clustering
        clusterer = cluster_image_colors(str(image_path), n_colors=colors, random_state=random_state)

        # Output text information if requested
        if text:
            click.echo("\nDominant Colors (sorted by frequency):")
            dominant_colors = clusterer.get_dominant_colors()
            for i, (color, freq) in enumerate(dominant_colors, 1):
                r, g, b = color
                hex_color = f'#{r:02x}{g:02x}{b:02x}'
                percentage = (freq / len(clusterer.pixels)) * 100
                click.echo(f"{i:2d}. {hex_color} RGB({r:3d}, {g:3d}, {b:3d}) - {percentage:.1f}% ({freq} pixels)")

        # Save palette visualization if requested
        if output_palette:
            click.echo(f"Saving color palette to: {output_palette}")
            clusterer.visualize_palette(save_path=output_palette)
            click.echo("Palette saved!")

        # Save 3D cluster visualization if requested
        if visualize_clusters_3d:
            click.echo(f"Saving 3D cluster visualization to: {visualize_clusters_3d}")
            clusterer.visualize_clusters(save_path=visualize_clusters_3d)
            click.echo("3D cluster visualization saved!")

        # Save 2D cluster visualization if requested
        if visualize_clusters_2d:
            click.echo(f"Saving 2D cluster visualization to: {visualize_clusters_2d}")
            clusterer.visualize_clusters_2d(save_path=visualize_clusters_2d)
            click.echo("2D cluster visualization saved!")

        # Reduce image colors - save clusterized version by default
        if not no_save:
            if reduce_colors:
                # Use custom output filename
                output_path = reduce_colors
            else:
                # Auto-generate filename with "_clusterized" suffix
                stem = image_path.stem
                suffix = image_path.suffix
                output_path = image_path.parent / f"{stem}_clusterized{suffix}"

            click.echo(f"Saving clusterized image to: {output_path}")
            clusterer.reduce_image_colors(output_path=str(output_path))
            click.echo("Clusterized image saved!")

        # If no output options specified, show basic info and palette
        if not text and not output_palette and not reduce_colors:
            click.echo(f"\nExtracted {colors} colors from image.")
            dominant_colors = clusterer.get_dominant_colors()
            click.echo("Top colors:")
            for i, (color, freq) in enumerate(dominant_colors[:5], 1):  # Show top 5
                r, g, b = color
                hex_color = f'#{r:02x}{g:02x}{b:02x}'
                percentage = (freq / len(clusterer.pixels)) * 100
                click.echo(f"  {hex_color} - {percentage:.1f}%")
            if len(dominant_colors) > 5:
                click.echo(f"  ... and {len(dominant_colors) - 5} more")

            # Try to show palette (will work in interactive environments)
            try:
                clusterer.visualize_palette()
            except:
                click.echo("\nTip: Use --output-palette to save the color palette as an image")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cluster_colors()
