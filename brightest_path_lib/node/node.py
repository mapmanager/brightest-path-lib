import numpy as np
import numba as nb

# Numba-optimized helper functions for node operations
@nb.njit(fastmath=True)
def compute_f_score(g_score: float, h_score: float) -> float:
    """Compute f_score from g_score and h_score with Numba optimization"""
    return g_score + h_score

@nb.njit
def validate_point(point):
    """Validate point array with Numba optimization"""
    return len(point) > 0

class Node:
    """Class holding information about a node, optimized for numerical operations

    Parameters
    ----------
    point : numpy ndarray
        the 2D/3D coordinates of the node (can be a pixel or a voxel)
    g_score : float
        the distance from a starting node to the current node
    h_score : float
        the estimated distance from the current node to a goal_node
    predecessor : Node
        the current node's immediate predecessor, from which we
        travelled to the current node
    
    Attributes
    ----------
    point : numpy ndarray
        the 2D/3D coordinates of the node
    g_score : float
        the actual cost from a starting node to the current node
    h_score : float
        the estimated cost from the current node to a goal_node
    f_score : float
        the sum of the node's g_score and h_score
    predecessor : Node
        the current node's immediate predecessor, from which we
        travelled to the current node
    """
    __slots__ = ('_point', '_g_score', '_h_score', '_f_score', '_predecessor')
    
    def __init__(
        self,
        point: np.ndarray,
        g_score: float,
        h_score: float,
        predecessor: 'Node' = None
    ):
        # Convert point to int64 for better performance with Numba
        if isinstance(point, list):
            point = np.array(point)
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        if point.dtype != np.int64:
            point = point.astype(np.int64)
            
        self.point = point
        self._g_score = float(g_score)
        self._h_score = float(h_score)
        # Use Numba function for f_score calculation
        self._f_score = compute_f_score(g_score, h_score)
        self._predecessor = predecessor
    
    @property
    def point(self):
        return self._point
    
    @point.setter
    def point(self, value: np.ndarray):
        if value is None:
            raise TypeError
        if not validate_point(value):
            raise ValueError
        # Ensure int64 type for better performance
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.int64)
        if value.dtype != np.int64:
            value = value.astype(np.int64)
        self._point = value

    @property
    def g_score(self):
        return self._g_score
    
    @g_score.setter
    def g_score(self, value: float):
        if value is None:
            raise TypeError
        self._g_score = float(value)
        # Update f_score when g_score changes
        self._f_score = compute_f_score(self._g_score, self._h_score)
    
    @property
    def h_score(self):
        return self._h_score
    
    @h_score.setter
    def h_score(self, value: float):
        if value is None:
            raise TypeError
        self._h_score = float(value)
        # Update f_score when h_score changes
        self._f_score = compute_f_score(self._g_score, self._h_score)
    
    @property
    def f_score(self):
        return self._f_score
    
    @f_score.setter
    def f_score(self, value: float):
        if value is None:
            raise TypeError
        self._f_score = float(value)
    
    @property
    def predecessor(self):
        return self._predecessor
    
    @predecessor.setter
    def predecessor(self, value):
        self._predecessor = value