#!/bin/bash
# Activation script for the image clustering virtual environment

echo "Activating image clustering virtual environment..."
source image_clustering_env/bin/activate
echo "Environment activated! You can now run:"
echo "  python cluster_colors.py [image] [options]"
echo "  python -c \"import color_clusterer; print('Ready!')\""
echo ""
echo "To deactivate later, run: deactivate"
