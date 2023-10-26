import numpy as np

from numpy.random import normal

from typing import List

Co_Ordinate = np.ndarray[int]
Angle = float


class Bird:
    def __init__(
        self,
        position: Co_Ordinate,
        orientation: Angle,
        velocity: float,
        noise: float,
        container_dimension: int,
    ) -> None:
        self.position = position
        self.orientation = orientation
        self.velocity = velocity

        self.noise = noise
        self.container_dimension = container_dimension

        # Calculate the direction of travel
        self.direction = np.array([np.cos(orientation), np.sin(orientation)])

    def generate_angle_noise(self) -> float:
        # Returns a random sample from a normal distribution with a mean of 0 and a standard
        # deviation of the noise argument.
        return normal(loc=0, scale=self.noise)

    def implement_periodic_condition(self, co_ordinate: Co_Ordinate) -> Co_Ordinate:
        """Implementing the boundary conditions that essentially make the bird
        behave like it's moving around a sphere where if it goes too far left it
        will emerge on the right (moving to the left) and if it goes too far up
        it will emerge on the bottom (moving to the top) and visa versa.

        Parameters
        ----------
        co_ordinate : Co_Ordinate
            A 2 dimensional co-ordinate.

        Returns
        -------
        Co_Ordinate
            An updated co-ordinate that satisfies the periodic boundary conditions
            of the container
        """
        row, column = co_ordinate

        # Case when the bird has gone below the container
        if row >= self.container_dimension:
            # Reset row to (highest index) 0
            row = 0

        # Case when the bird has gone too far right of the container
        if column >= self.container_dimension:
            # Reset column to (farthest left index) 0
            column = 0

        # Case when the bird has gone above the container
        if row < 0:
            # Reset the row to be the bottom of the container
            row = self.container_dimension - 1

        # Case when the bird has to the left of the container
        if column < 0:
            # Reset the column to be the farthest right of the container
            column = self.container_dimension - 1

        return np.array([row, column])

    def update_position(self) -> Co_Ordinate:
        """Updates the position according to the vicsek position update model.

        Returns
        -------
        Co_Ordinate
            Updated co-ordinate
        """
        # Find the updated position
        displacement = self.direction * self.velocity
        updated_position = self.position + displacement

        # Next we need to check that the position lies within the grid's boundaries
        # If not then we move it to the opposite side of the grid
        updated_position = self.implement_periodic_condition(updated_position)

        return updated_position

    def update_angle(self, nearby_birds: List) -> float:
        """Update angles according the vicsek angle update rule.

        Parameters
        ----------
        nearby_birds : List
            A list of nearby birds calculated in the vicsek file.

        Returns
        -------
        float
            An updated angle
        """
        angle_noise = self.generate_angle_noise()
        direction_sum = sum([bird.direction for bird in nearby_birds]) + self.direction

        direction_sum_norm = np.linalg.norm(direction_sum)
        if direction_sum_norm == 0:
            return angle_noise
        else:
            normalised_direction_sum = direction_sum / np.linalg.norm(direction_sum)
            general_direction = np.arccos(normalised_direction_sum[0])

            return general_direction + angle_noise

    def time_step(self, nearby_birds: List):
        return Bird(
            position=self.update_position(),
            orientation=self.update_angle(nearby_birds),
            velocity=self.velocity,
            noise=self.noise,
            container_dimension=self.container_dimension,
        )
