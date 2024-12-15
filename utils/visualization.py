import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(point_cloud, rgb=None, scalar_field=None, cmap='viridis', point_size=1):
    """
    Visualize a 3D point cloud with optional RGB colors or scalar field coloring.

    Parameters:
        point_cloud (numpy.ndarray): Nx3 array with XYZ coordinates.
        rgb (numpy.ndarray, optional): Nx3 array with RGB values (0-1 range).
        scalar_field (numpy.ndarray, optional): Nx1 array with scalar values for coloring.
        cmap (str, optional): Matplotlib colormap name to use for scalar field coloring.
    """
    if point_cloud.shape[1] != 3:
        raise ValueError("Point cloud must be a Nx3 array.")

    if rgb is not None and (rgb.shape[0] != point_cloud.shape[0] or rgb.shape[1] != 3):
        raise ValueError("RGB array must have the same number of points as the point cloud and be Nx3.")

    if scalar_field is not None and scalar_field.shape[0] != point_cloud.shape[0]:
        raise ValueError("Scalar field must have the same number of points as the point cloud.")

    # Extract XYZ coordinates
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    # Calculate aspect ratio
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    max_range = max(x_range, y_range, z_range)

    # Set up the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Configure aspect ratio to avoid skewing
    ax.set_box_aspect((x_range, y_range, z_range))

    # Coloring logic
    if rgb is not None:
        colors = rgb
    elif scalar_field is not None:
        colors = scalar_field.flatten()
    else:
        colors = 'b'  # Default to blue if no color data provided

    # Plot the point cloud
    scatter = ax.scatter(x, y, z, c=colors, cmap=cmap if scalar_field is not None else None, s=point_size)

    # Add a colorbar if using a scalar field
    if scalar_field is not None:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Scalar Field')

    # Set labels and show
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
