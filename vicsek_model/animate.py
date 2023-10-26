import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize

from vicsek_model.vicsek import Vicsek

from typing import Tuple, List


class Animate:
    def __init__(
        self, model: Vicsek, figsize: Tuple[int], interval_between_frames: float
    ) -> None:
        self.model = model

        # Initialise the plot
        container_dimension = model.container_dimension
        fig, ax = plt.subplots(figsize=figsize)
        self.fig = fig
        self.ax = ax

        self.interval = interval_between_frames

        self.initialise_figure()

        self.cm, self.norm = self.initialise_colour_mapping()

    def initialise_colour_mapping(self):
        # Define the colors and corresponding positions in the colormap
        colors = [(0, "violet"), (0.5, "green"), (1, "red")]
        n_bins = 1000  # Number of bins for the colormap
        cmap_name = "angle_mapping"

        # Create the colormap using LinearSegmentedColormap
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # Create a Normalize instance to scale values from 0 to 2*pi
        norm = Normalize(vmin=0, vmax=2 * np.pi)
        return cm, norm

    def initialise_figure(self):
        # Add grid lines
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Add in integer x,y ticks
        plt.xticks(range(self.model.container_dimension))
        plt.yticks(range(self.model.container_dimension))

        # Set x,y limits
        plt.xlim(-6, self.model.container_dimension + 5)
        plt.ylim(-6, self.model.container_dimension + 5)

        # Invert y axis to represent matrix indexing
        plt.gca().invert_yaxis()

    def update(self, frame: int) -> None:
        plt.clf()  # Clear the previous plot
        self.initialise_figure()
        plt.text(
            0.5,
            1.1,
            f"Timestep: {frame}",
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="center",
        )
        plt.text(
            0.5,
            1.02,
            f"Vicsek order parameter: {self.model.calculate_vicsek_order_parameter():.5f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="center",
        )
        plt.text(
            0.5,
            1.05,
            f"Noise parameter: {self.model.noise}, Number of birds: {self.model.num_birds}",
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="center",
        )
        bird_positions: List[np.ndarray[int]] = [
            bird.position for bird in self.model.birds
        ]
        angles, xs, ys = [], [], []
        for index, (x, y) in enumerate(bird_positions):
            angles.append(self.model.birds[index].orientation % (2 * np.pi))
            xs.append(x)
            ys.append(y)
        plt.scatter(xs, ys, c=angles, cmap=self.cm, norm=self.norm)
        cbar = plt.colorbar(orientation="vertical")
        cbar.set_label(r"$\theta$ in radians")
        # Make a time step
        self.model = self.model.execute_time_step()

    def main(self, time_steps: int):
        animation = FuncAnimation(
            self.fig,
            self.update,
            frames=np.arange(0, time_steps, 1),
            interval=self.interval,
        )
        f = "./vicsek.mp4" 
        writervideo = FFMpegWriter(fps=60) 
        animation.save(f, writer=writervideo)
