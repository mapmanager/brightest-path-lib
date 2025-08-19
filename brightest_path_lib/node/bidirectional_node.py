import numpy as np
import numba as nb

# Numba-optimized helper functions
@nb.njit(fastmath=True)
def compute_f_score(g_score: float, h_score: float) -> float:
    """Compute f_score from g_score and h_score with Numba optimization"""
    return g_score + h_score

@nb.njit
def validate_point(point):
    """Validate point array with Numba optimization"""
    return len(point) > 0

@nb.njit
def get_score_by_direction(from_start, start_score, goal_score):
    """Get the appropriate score based on direction with Numba optimization"""
    if from_start:
        return start_score
    return goal_score

class BidirectionalNode:
    """Class holding attributes and properties of a Bidirectional Node, optimized for performance

    Parameters
    ----------
    point : numpy ndarray
        the 2D/3D coordinates of the node (can be a pixel or a voxel)
    g_score_from_start : float
        the distance from a starting node to the current node
    g_score_from_goal : float
        the distance from a goal node to the current node
    h_score_from_start : float
        the estimated distance from the current node to a goal node
    h_score_from_goal : float
        the estimated distance from the current node to a start node
    predecessor_from_start : BidirectionalNode
        the current node's immediate predecessor, from which we
        travelled to the current node
        The predecessor's first ancestor is the start node
    predecessor_from_goal : BidirectionalNode
        the current node's immediate predecessor, from which we
        travelled to the current node
        The predecessor's first ancestor is the goal node
    
    Attributes
    ----------
    point : numpy ndarray
        the 2D/3D coordinates of the node (can be a pixel or a voxel)
    g_score_from_start : float
        the distance from a starting node to the current node
    g_score_from_goal : float
        the distance from a goal node to the current node
    h_score_from_start : float
        the estimated distance from the current node to a goal node
    h_score_from_goal : float
        the estimated distance from the current node to a start node
    f_score_from_start : float
        the sum of g_score_from_start and h_score_from_start
    f_score_from_goal : float
        the sum of g_score_from_goal and h_score_from_goal
    predecessor_from_start : BidirectionalNode
        the current node's immediate predecessor, from which we
        travelled to the current node
        The predecessor's first ancestor is the start node
    predecessor_from_goal : BidirectionalNode
        the current node's immediate predecessor, from which we
        travelled to the current node The predecessor's first ancestor
        is the goal node
    """
    # Use __slots__ to reduce memory overhead and access time
    __slots__ = (
        '_point', '_g_score_from_start', '_g_score_from_goal', 
        '_h_score_from_start', '_h_score_from_goal',
        '_f_score_from_start', '_f_score_from_goal',
        '_predecessor_from_start', '_predecessor_from_goal'
    )
    
    def __init__(
        self,
        point: np.ndarray,
        g_score_from_start: float = float('inf'),
        g_score_from_goal: float = float('inf'),
        h_score_from_start: float = float('inf'),
        h_score_from_goal: float = float('inf'),
        f_score_from_start: float = float('inf'),
        f_score_from_goal: float = float('inf'),
        predecessor_from_start: 'BidirectionalNode' = None,
        predecessor_from_goal: 'BidirectionalNode' = None
    ):
        # Convert point to int64 for better performance with Numba
        if isinstance(point, list):
            point = np.array(point)
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        if point.dtype != np.int64:
            point = point.astype(np.int64)
            
        self.point = point
        self._g_score_from_start = float(g_score_from_start)
        self._g_score_from_goal = float(g_score_from_goal)
        self._h_score_from_start = float(h_score_from_start)
        self._h_score_from_goal = float(h_score_from_goal)
        
        # Use Numba functions for f_score calculations if they're not provided
        if f_score_from_start == float('inf') and g_score_from_start != float('inf') and h_score_from_start != float('inf'):
            self._f_score_from_start = compute_f_score(g_score_from_start, h_score_from_start)
        else:
            self._f_score_from_start = float(f_score_from_start)
            
        if f_score_from_goal == float('inf') and g_score_from_goal != float('inf') and h_score_from_goal != float('inf'):
            self._f_score_from_goal = compute_f_score(g_score_from_goal, h_score_from_goal)
        else:
            self._f_score_from_goal = float(f_score_from_goal)
            
        self._predecessor_from_start = predecessor_from_start
        self._predecessor_from_goal = predecessor_from_goal
    
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
    def g_score_from_start(self):
        return self._g_score_from_start
    
    @g_score_from_start.setter
    def g_score_from_start(self, value: float):
        self._g_score_from_start = float(value)
        # Update f_score when g_score changes
        if self._h_score_from_start != float('inf'):
            self._f_score_from_start = compute_f_score(value, self._h_score_from_start)
    
    @property
    def g_score_from_goal(self):
        return self._g_score_from_goal
    
    @g_score_from_goal.setter
    def g_score_from_goal(self, value: float):
        self._g_score_from_goal = float(value)
        # Update f_score when g_score changes
        if self._h_score_from_goal != float('inf'):
            self._f_score_from_goal = compute_f_score(value, self._h_score_from_goal)
    
    @property
    def h_score_from_start(self):
        return self._h_score_from_start
    
    @h_score_from_start.setter
    def h_score_from_start(self, value: float):
        self._h_score_from_start = float(value)
        # Update f_score when h_score changes
        if self._g_score_from_start != float('inf'):
            self._f_score_from_start = compute_f_score(self._g_score_from_start, value)
    
    @property
    def h_score_from_goal(self):
        return self._h_score_from_goal
    
    @h_score_from_goal.setter
    def h_score_from_goal(self, value: float):
        self._h_score_from_goal = float(value)
        # Update f_score when h_score changes
        if self._g_score_from_goal != float('inf'):
            self._f_score_from_goal = compute_f_score(self._g_score_from_goal, value)
    
    @property
    def f_score_from_start(self):
        return self._f_score_from_start
    
    @f_score_from_start.setter
    def f_score_from_start(self, value: float):
        self._f_score_from_start = float(value)
    
    @property
    def f_score_from_goal(self):
        return self._f_score_from_goal
    
    @f_score_from_goal.setter
    def f_score_from_goal(self, value: float):
        self._f_score_from_goal = float(value)
    
    @property
    def predecessor_from_start(self):
        return self._predecessor_from_start
    
    @predecessor_from_start.setter
    def predecessor_from_start(self, value):
        self._predecessor_from_start = value
    
    @property
    def predecessor_from_goal(self):
        return self._predecessor_from_goal
    
    @predecessor_from_goal.setter
    def predecessor_from_goal(self, value):
        self._predecessor_from_goal = value
    
    # Fast accessor methods optimized for performance
    def get_g(self, from_start: bool) -> float:
        """Get the appropriate g_score based on direction"""
        return get_score_by_direction(from_start, self._g_score_from_start, self._g_score_from_goal)
    
    def get_f(self, from_start: bool) -> float:
        """Get the appropriate f_score based on direction"""
        return get_score_by_direction(from_start, self._f_score_from_start, self._f_score_from_goal)
    
    def set_g(self, g_score: float, from_start: bool):
        """Set the appropriate g_score based on direction"""
        if from_start:
            self.g_score_from_start = g_score
        else:
            self.g_score_from_goal = g_score
    
    def set_f(self, f_score: float, from_start: bool):
        """Set the appropriate f_score based on direction"""
        if from_start:
            self.f_score_from_start = f_score
        else:
            self.f_score_from_goal = f_score
    
    def set_predecessor(self, predecessor, from_start: bool):
        """Set the appropriate predecessor based on direction"""
        if from_start:
            self.predecessor_from_start = predecessor
        else:
            self.predecessor_from_goal = predecessor