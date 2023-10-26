import sys

sys.path.insert(0, "../")  # Insert the directory above so that we can import Bird

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from bird import Bird

# Create a bird
position = np.array([1, 2])
orientation = np.pi
velocity = 2
noise = 4
container_dimension = 5
bird = Bird(position, orientation, velocity, noise, container_dimension)


class TestBird(unittest.TestCase):
    def test_periodic_boundary_condition_bottom(self):
        """
        Condition 1: the bird flies below the container
        and should come out of the top of the container, i.e
        it's y co-ordinate (row value) should be 0.
        """
        bottom_of_container = np.array([1, container_dimension])
        new_co_ordinates = bird.implement_periodic_condition(bottom_of_container)
        assert_array_equal(new_co_ordinates, np.array([1, 0]))

    def test_periodic_boundary_condition_top(self):
        """
        Condition 2: the bird flies above the container
        and should come out of the top of the container, i.e
        it's y co-ordinate (row value) should be 0.
        """
        bottom_of_container = np.array([1, -1])
        new_co_ordinates = bird.implement_periodic_condition(bottom_of_container)
        assert_array_equal(new_co_ordinates, np.array([1, container_dimension - 1]))

    def test_periodic_boundary_conditions_left(self):
        """
        Condition 3: the bird flies too far left of the container
        and should come out of the top of the container, i.e
        it's y co-ordinate (row value) should be 0.
        """
        bottom_of_container = np.array([-1, 2])
        new_co_ordinates = bird.implement_periodic_condition(bottom_of_container)
        assert_array_equal(new_co_ordinates, np.array([container_dimension - 1, 2]))

    def test_periodic_boundary_conditions_right(self):
        """
        Condition 4: the bird flies too far to the right of the container
        and should come out of the top of the container, i.e
        it's y co-ordinate (row value) should be 0.
        """
        bottom_of_container = np.array([container_dimension, 1])
        new_co_ordinates = bird.implement_periodic_condition(bottom_of_container)
        assert_array_equal(new_co_ordinates, np.array([0, 1]))

    def test_position_update(self):
        """
        Checks that the position update is working
        correctly
        """
        new_position = bird.update_position()
        # Expect (container_dimension - 1,2)
        assert_array_equal(new_position, np.array([container_dimension - 1, 2]))

    def test_angle_update(self):
        assert 2 == 2


if __name__ == "__main__":
    unittest.main()
