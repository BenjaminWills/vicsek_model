import numpy as np

from numpy.random import uniform

from vicsek_model.bird import Bird

from typing import List
from tqdm import tqdm

Co_Ordinate = np.ndarray[float]
Angle = float


class Vicsek:
    def __init__(
        self,
        container_dimension: int,
        num_birds: int,
        velocity: float,
        noise: float,
        search_radius: float,
        birds: List[Bird] = None,
    ) -> None:
        self.container_dimension = container_dimension
        self.num_birds = num_birds
        self.velocity = velocity
        self.noise = noise
        self.search_radius = search_radius

        self.bird_indexes = range(self.num_birds)

        self.birds = self.initialise_birds() or birds
        self.distance_matrix = self.generate_near_bird_matrix()

    def initialise_positions(self) -> List[Co_Ordinate]:
        # At most we can have N ** 2 birds.
        if self.num_birds > self.container_dimension**2:
            raise ValueError("Error, there are too many birds!")

        # All other scenarios
        else:
            positions: List[Co_Ordinate] = []

            while len(positions) != self.num_birds:
                # Generate 2 integer co-ordinates for the bird's starting position
                row, column = list(uniform(0, self.container_dimension, 2))

                if (row, column) not in positions:
                    positions.append((row, column))

            # Map all the position tuples to arrays
            return list(map(np.array, positions))

    def initialise_orientations(self) -> List[Angle]:
        # Generates a list of random angles for the birds to be facing
        return uniform(0, 1, self.num_birds) * 2 * np.pi

    def initialise_birds(self) -> List[Bird]:
        initial_positions = self.initialise_positions()
        initial_orientations = self.initialise_orientations()
        birds: List[Bird] = [
            Bird(
                position,
                orientation,
                self.velocity,
                self.noise,
                self.container_dimension,
            )
            for position, orientation in zip(initial_positions, initial_orientations)
        ]
        return birds

    def generate_near_bird_matrix(self) -> np.ndarray[float]:
        """This function will generate a distance metric given a list of birds (and therefore their
        positions). Due to the periodic boundary conditions of the container that the birds live in
        it is necessary to calculate 2 distances rather than one, the first is the classic euclidian
        distance, i.e d_i,j = |r_i - r_j| or d_i,j = condtainer_dimension - |r_i - r_j|, thus we calculate
        Distance_matrix_i,j = min(|r_i - r_j|, condtainer_dimension - |r_i - r_j|).

        This matrix is symmetric thus we only need to calculate the values in the upper triangle, further
        all of the elements on the main diagonal are 0 for trivial reasons.

        Parameters
        ----------
        birds : List[Bird]
            A list of Bird objects
        container_dimension : int
            The dimension of the container that the birds are in

        Returns
        -------
        np.ndarray[float]
            A distance matrix defined in the intro to this function.
        """
        # Matrix is symmetric
        distance_matrix = np.zeros((self.num_birds, self.num_birds))
        # Fill in the above diagonal with numbers
        for row in self.bird_indexes:
            for col in self.bird_indexes:
                above_diagonal = row < col
                if above_diagonal:
                    bird_1_position = self.birds[row].position
                    bird_2_position = self.birds[col].position

                    distance_1 = np.linalg.norm(bird_1_position - bird_2_position)
                    distance_2 = self.container_dimension - distance_1

                    distance_matrix[row, col] = min(distance_1, distance_2)

        # Reflect over the lower diagonal and add
        distance_matrix = distance_matrix.T + distance_matrix
        return distance_matrix

    def radius_search(self, bird_index: int) -> List[Bird]:
        """Search for nearby birds around the bird whose index we are checking in a circle
        of radius r.

        Parameters
        ----------
        bird_index : int
            The index of the bird in self.birds

        Returns
        -------
        List[Bird]
            A list of birds that are nearby to the bird with the given index
        """
        near_birds = []
        # One row of the distance matrix shows the distance from the other brids in the list
        row = self.distance_matrix[bird_index, :]
        for col in self.bird_indexes:
            if bird_index != col:
                distance = row[col]
                if distance <= self.search_radius:
                    near_bird = self.birds[col]
                    near_birds.append(near_bird)
        return near_birds

    def complete_radius_search(self) -> List[List[Bird]]:
        """This implements the radius search for each bird.

        Returns
        -------
        List[List[Bird]]
            A list of lists of birds, where the index corresponds to the i'th bird
        """
        nearest_bird_list = []
        for bird_index in self.bird_indexes:
            nearest_bird_list.append(self.radius_search(bird_index))
        return nearest_bird_list

    def execute_time_step(self):
        """This function executes a time step in the Viscek model.
        for each bird in the total list of birds, it checks the search
        radius of cells around the bird, for near neighbours for which it will
        adjust its position accordingly, once this has been completed for all birds
        a new model is instantiated with a new set of birds that have new positions
        and angles.

        The reason for returning a new model in this method is that we need the __init__
        method to run to generate a new distance matrix, this can only happen when the class
        is instantiated.

        Returns
        -------
        Vicsek
            A new model with updated bird metadata.
        """
        birds = []
        nearby_birds = self.complete_radius_search()
        for index, bird in enumerate(self.birds):
            nearby_birds_for_bird = nearby_birds[index]
            next_bird = bird.time_step(nearby_birds_for_bird)
            birds.append(next_bird)

        return Vicsek(
            self.container_dimension,
            self.num_birds,
            self.velocity,
            self.noise,
            birds=birds,
            search_radius=self.search_radius,
        )

    def get_bird_positions(self) -> np.ndarray[int]:
        """Generates a matrix of 1's and 0's, the i,j entry
        of the matrix is 1 if a bird is inhabiting (i,j) on
        the plane, otherwise it is 0.

        Returns
        -------
        np.ndarray[int]
            Matrix of bird positions
        """
        bird_positions = np.zeros((self.container_dimension, self.container_dimension))
        for bird in self.birds:
            row, column = bird.position
            bird_positions[row, column] = 1
        return bird_positions

    def calculate_vicsek_order_parameter(self) -> float:
        """Calculates the VOP for a given configuration of birds

        Returns
        -------
        float
            A floating point number that represents the average magnitude of a birds direction.
        """
        direction_sum = sum([bird.direction for bird in self.birds])
        return np.linalg.norm(direction_sum) / self.num_birds

    def simulate(self, iterations: int) -> List[int]:
        model = self
        VOP = []
        for iteration in tqdm(range(iterations)):
            VOP.append(model.calculate_vicsek_order_parameter())
            model = model.execute_time_step()

        return VOP
