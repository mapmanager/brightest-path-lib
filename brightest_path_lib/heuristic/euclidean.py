from brightest_path_lib.heuristic import Heuristic
import math
import numpy as np
from typing import Tuple
from numba import njit, float64

# Ultra-simple but very fast 2D distance calculation
@njit(fastmath=True)
def _fast_euclidean_distance_2d(current_y, current_x, goal_y, goal_x, scale_x, scale_y):
    """Minimal, efficient 2D Euclidean distance calculation"""
    dx = (goal_x - current_x) * scale_x
    dy = (goal_y - current_y) * scale_y
    return math.sqrt(dx*dx + dy*dy)

# Ultra-simple but very fast 3D distance calculation
@njit(fastmath=True)
def _fast_euclidean_distance_3d(current_z, current_y, current_x, goal_z, goal_y, goal_x, 
                               scale_x, scale_y, scale_z):
    """Minimal, efficient 3D Euclidean distance calculation"""
    dx = (goal_x - current_x) * scale_x
    dy = (goal_y - current_y) * scale_y
    dz = (goal_z - current_z) * scale_z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

# Simple but efficient dispatcher
@njit(fastmath=True)
def _fast_estimate_cost(current_point, goal_point, scale_x, scale_y, scale_z):
    """Simplified but efficient cost estimation"""
    # Direct dimension check
    if current_point.shape[0] == 2:  # 2D case
        return _fast_euclidean_distance_2d(
            current_point[0], current_point[1],  # y, x for current
            goal_point[0], goal_point[1],        # y, x for goal
            scale_x, scale_y
        )
    else:  # 3D case
        return _fast_euclidean_distance_3d(
            current_point[0], current_point[1], current_point[2],  # z, y, x for current
            goal_point[0], goal_point[1], goal_point[2],          # z, y, x for goal
            scale_x, scale_y, scale_z
        )

class Euclidean(Heuristic):
    """Simplified and optimized heuristic cost using Euclidean distance

    Parameters
    ----------
    scale : Tuple
        the scale of the image's axes. For example (1.0 1.0) for a 2D image.
        - for 2D points, the order of scale is: (x, y)
        - for 3D points, the order of scale is: (x, y, z)
    
    Attributes
    ----------
    scale_x : float
        the scale of the image's X-axis
    scale_y : float
        the scale of the image's Y-axis
    scale_z : float
        the scale of the image's Z-axis
    """
    def __init__(self, scale: Tuple):
        if scale is None:
            raise TypeError("Scale cannot be None")
        if len(scale) == 0:
            raise ValueError("Scale cannot be empty")

        self.scale_x = scale[0]
        self.scale_y = scale[1]
        self.scale_z = 1.0
        if len(scale) == 3:
            self.scale_z = scale[2]

    def estimate_cost_to_goal(self, current_point: np.ndarray, goal_point: np.ndarray) -> float:
        """Calculate the estimated cost from current point to the goal
    
        Parameters
        ----------
        current_point : numpy ndarray
            the coordinates of the current point
        goal_point : numpy ndarray
            the coordinates of the goal point
        
        Returns
        -------
        float
            the estimated cost to goal in the form of Euclidean distance
        """
        if current_point is None or goal_point is None:
            raise TypeError("Points cannot be None")
        if (len(current_point) == 0 or len(goal_point) == 0) or (len(current_point) != len(goal_point)):
            raise ValueError("Points must have the same dimensions and cannot be empty")

        # Use the simplified Numba-optimized function
        return _fast_estimate_cost(
            current_point, 
            goal_point, 
            self.scale_x, 
            self.scale_y, 
            self.scale_z
        )