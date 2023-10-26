import matplotlib.pyplot as plt
import numpy as np

from vicsek_model.vicsek import Vicsek


def plot_binary_matrix_with_integer_ticks(matrix: np.ndarray[int]):
    """
    Plot a binary matrix with integer ticks using matplotlib.

    Args:
        matrix (list[list[int]]): The binary matrix containing only 0's and 1's.

    Returns:
        None
    """
    n = len(matrix)
    x = []
    y = []

    for i in range(n):
        for j in range(n):
            if matrix[i, j] == 1:
                x.append(i)
                y.append(j)

    plt.scatter(x, y, color="blue")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.xlim(-1, n)
    plt.ylim(-1, n)
    plt.gca().invert_yaxis()
    plt.show()


def plot_circles_around_points(
    points: np.ndarray[int], radius: float, container_dimension: int
):
    """
    Plot circles around specified points.

    Args:
        points (list): A list of 2D points, where each point is a tuple (x, y).
        radius (float): The radius of the circles.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, point in enumerate(points):
        x, y = point

        # Create a circle patch for each point
        circle = plt.Circle((x, y), radius, fill=False, color="blue", linewidth=2)

        # Add the circle patch to the plot
        ax.add_patch(circle)

        # Draw a point at the center of the circle
        plt.scatter(
            x,
            y,
            color=np.random.rand(
                3,
            ),
        )

        # Add label for the legend
        plt.text(x - 0.5, y + 0.5, f"bird {i}", fontsize=10, color="black")

    # Set aspect of the plot to be equal, so the circles look circular
    ax.set_aspect("equal", adjustable="box")

    # Calculate the limits to include all circles
    min_x = 0
    max_x = container_dimension
    min_y = 0
    max_y = container_dimension

    # Set limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Set ticks to display only integers
    plt.xticks(range(int(min_x) - 1, int(max_x) + 2))
    plt.yticks(range(int(min_y) - 1, int(max_y) + 2))

    # Add gridlines
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Invert y-axis
    plt.gca().invert_yaxis()

    # Show the plot
    plt.show()


def visualise_birds(model: Vicsek):
    bird_positions = model.get_bird_positions()
    plot_binary_matrix_with_integer_ticks(bird_positions)


def visualise_search_radii(model: Vicsek, radius: float, container_dimension: int):
    positions = [bird.position for bird in model.birds]
    plot_circles_around_points(positions, radius, container_dimension)
