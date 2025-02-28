import numpy as np
from scipy.interpolate import griddata

def grid_points_in_quadrilateral(bottom_left, bottom_right, top_right, top_left, n_x, n_y):
    """
    Generate an equispaced grid of points inside a quadrilateral using bilinear interpolation.

    Args:
        bottom_left (tuple): (x, y) coordinates of the bottom-left corner.
        bottom_right (tuple): (x, y) coordinates of the bottom-right corner.
        top_right (tuple): (x, y) coordinates of the top-right corner.
        top_left (tuple): (x, y) coordinates of the top-left corner.
        n_x (int): Number of points along the horizontal direction.
        n_y (int): Number of points along the vertical direction.

    Returns:
        np.ndarray: Array of shape (n_x * n_y, 2) containing the grid points.
    """
    # Convert corners to NumPy arrays
    p0 = np.array(bottom_left)
    p1 = np.array(bottom_right)
    p2 = np.array(top_right)
    p3 = np.array(top_left)

    grid_points = []
    # u parameter: horizontal coordinate, v parameter: vertical coordinate
    # u, v vary from 0 to 1
    for i in range(n_y):
        v = i / (n_y - 1) if n_y > 1 else 0.5
        for j in range(n_x):
            u = j / (n_x - 1) if n_x > 1 else 0.5
            # Bilinear interpolation formula:
            # point = (1-u)*(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3
            point = (1 - u) * (1 - v) * p0 + u * (1 - v) * p1 + u * v * p2 + (1 - u) * v * p3
            grid_points.append(point)
    return np.array(grid_points)

# Example usage:
if __name__ == '__main__':
    # Define the quadrilateral corners (order: bottom_left, bottom_right, top_right, top_left)
    bottom_left = (0, 0)
    bottom_right = (4, 1)
    top_right = (3.5, 5)
    top_left = (-0.5, 4)

    # Generate a grid of 10x10 points inside the quadrilateral
    points = grid_points_in_quadrilateral(bottom_left, bottom_right, top_right, top_left, n_x=10, n_y=10)
    print(points)

    # Do the same as above using the RegulatGridInterpolator class
    points = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])
    values = np.array([
        [0, 1],
        [-0.5, 4],
        [4, 1],
        [3.5, 5]
    ])
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    interpolated_values = griddata(points, values, grid_points, method='linear')
    print("-"*40)
    print(interpolated_values)