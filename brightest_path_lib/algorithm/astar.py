# algorithm/astar.py

"""This implementation of the A* search algorithm finds the brightest path
in a graph. Each node in the graph represents a point in space, and the
weight of each edge represents the brightness of the path between the two nodes. 
The goal is to find the path that maximizes the total brightness.
     
To use the A* search algorithm for brightest path finding, a heuristic 
function is needed that estimates the maximum brightness that can be 
achieved from the current node to the goal node. One way to do this 
is to use a greedy heuristic that always selects the edge with the
highest brightness from the current node. 

The A* search algorithm starts at the start node and explores
neighboring nodes, selecting the node with the highest brightness
at each step. The algorithm uses the heuristic function to estimate
the maximum brightness that can be achieved from the current node to
the goal node. If the algorithm reaches the goal node, it terminates
and returns the brightest path. If not, it continues searching until
all nodes have been explored.

To search for the brightest path between two points in an image:

1. Initialize the AStarSearch class with the 2D/3D image,
   start point and the goal point: `astar = AStarSearch(image, start_point, goal_point)`
2. Call the search method: `path = astar.search()`
"""

from collections import defaultdict
import math
import numpy as np
from queue import PriorityQueue, Queue
from typing import List, Tuple
from brightest_path_lib.cost import Reciprocal
from brightest_path_lib.heuristic import Euclidean
from brightest_path_lib.image import ImageStats
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node


class AStarSearch:
    """A* Search Implementation

    Parameters
    ----------
    image : numpy ndarray
        the 2D/3D image on which we will run an A star search
    start_point : numpy ndarray
        the 2D/3D coordinates of the starting point (could be a pixel or a voxel)
        For 2D images, the coordinates are of the form (y, x)
        For 3D images, the coordinates are of the form (z, x, y)
    goal_point : numpy ndarray
        the 2D/3D coordinates of the goal point (could be a pixel or a voxel)
        For 2D images, the coordinates are of the form (y, x)
        For 3D images, the coordinates are of the form (z, x, y)
    scale : Tuple
        the scale of the image; defaults to (1.0, 1.0), i.e. image is not zoomed in/out
        For 2D images, the scale is of the form (x, y)
        For 3D images, the scale is of the form (x, y, z)
    cost_function : Enum CostFunction
        this enum value specifies the cost function to be used for computing 
        the cost of moving to a new point
        Default type is CostFunction.RECIPROCAL to use the reciprocal function
    heuristic_function : Enum HeuristicFunction
        this enum value specifies the heuristic function to be used to compute
        the estimated cost of moving from a point to the goal
        Default type is HeuristicFunction.EUCLIDEAN to use the 
        euclidean function for cost estimation
    open_nodes : Queue
        contains a list of points that are in the open set;
        can be used by the calling application to show a visualization
        of where the algorithm is searching currently
        Default value is None

    Attributes
    ----------
    image : numpy ndarray
        The 2D/3D image on which we will run an A star search
    start_point : numpy ndarray
        the coordinates of the start point
    goal_point : numpy ndarray
        the coordinates of the goal point
    scale : Tuple
        the scale of the image; defaults to (1.0, 1.0), i.e. image is not zoomed in/out
    cost_function : Cost
        the cost function to be used for computing the cost of moving 
        to a new point
        Default type is Reciprocal
    heuristic_function : Heuristic
        the heuristic function to be used to compute the estimated
        cost of moving from a point to the goal
        Default type is Euclidean
    is_canceled : bool
        should be set to True if the search needs to be stopped;
        false by default
    open_nodes : Queue
        contains a list of points that are in the open set;
        can be used by the calling application to show a visualization
        of where the algorithm is searching currently
    result : List[numpy ndarray]
        the result of the A star search containing the list of
        points that constitute the brightest path between start and goal
        points
    """

    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        scale: Tuple = (1.0, 1.0),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN,
        open_nodes: Queue = None
    ):

        self._validate_inputs(image, start_point, goal_point)

        self.image = image
        self.image_stats = ImageStats(image)
        self.start_point = np.round(start_point).astype(int)
        self.goal_point = np.round(goal_point).astype(int)
        self.scale = scale
        self.open_nodes = open_nodes

        if cost_function == CostFunction.RECIPROCAL:
            self.cost_function = Reciprocal(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = Euclidean(scale=self.scale)
        
        self.is_canceled = False
        self.found_path = False
        self.evaluated_nodes = 0
        self.result = []

    def _validate_inputs(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
    ):
        """Checks for a non-empty image, start point and goal point before
        the A* search
        """
        if image is None or start_point is None or goal_point is None:
            raise TypeError
        if len(image) == 0 or len(start_point) == 0 or len(goal_point) == 0:
            raise ValueError

    @property
    def found_path(self) -> bool:
        return self._found_path

    @found_path.setter
    def found_path(self, value: bool):
        if value is None:
            raise TypeError
        self._found_path = value

    @property
    def is_canceled(self) -> bool:
        return self._is_canceled

    @is_canceled.setter
    def is_canceled(self, value: bool):
        if value is None:
            raise TypeError
        self._is_canceled = value

    def search(self, verbose : bool = False) -> List[np.ndarray]:
        """Performs A star search to find the brightest path

        Parameters
        ----------
        verbose (bool)
            If True, will print `Found goal!` when goal is found.

        Returns
        -------
        List[np.ndarray]
            the list containing the 2D/3D point coordinates
            that constitute the brightest path between the
            start_point and the goal_point
        """
        count = 0
        open_set = PriorityQueue()
        start_node = Node(
            point=self.start_point, 
            g_score=0, 
            h_score=self._estimate_cost_to_goal(self.start_point), 
            predecessor=None
            )
        open_set.put((0, count, start_node)) # f_score, count: priority of occurence, current node
        open_set_hash = {tuple(self.start_point)} # hashset contains tuple of node coordinates to be visited
        close_set_hash = set() # hashset contains tuple of node coordinates already been visited
        f_scores = defaultdict(self._default_value) # key: tuple of node coordinates, value: f_score
        f_scores[tuple(self.start_point)] = start_node.f_score
        
        while not open_set.empty():
            if self.is_canceled:
                break
            current_node = open_set.get()[2]
            current_coordinates = tuple(current_node.point)
            if current_coordinates in close_set_hash:
                continue
            
            open_set_hash.remove(current_coordinates)

            if self._found_goal(current_node.point):
                if verbose:
                    print("Found goal!")
                self._construct_path_from(current_node)
                self.found_path = True
                break

            neighbors = self._find_neighbors_of(current_node)
            for neighbor in neighbors:
                neighbor_coordinates = tuple(neighbor.point)
                if neighbor_coordinates in close_set_hash:
                    # this neighbor has already been visited
                    continue
                if neighbor_coordinates not in open_set_hash:
                    count += 1
                    open_set.put((neighbor.f_score, count, neighbor))
                    open_set_hash.add(neighbor_coordinates)
                    if self.open_nodes is not None:
                        # add to our queue
                        # can be monitored from caller to update plots
                        self.open_nodes.put(neighbor_coordinates)
                else:
                    if neighbor.f_score < f_scores[neighbor_coordinates]:
                        f_scores[neighbor_coordinates] = neighbor.f_score
                        count += 1
                        open_set.put((neighbor.f_score, count, neighbor))
            
            close_set_hash.add(current_coordinates)

        self.evaluated_nodes = count
        return self.result
    
    def _default_value(self) -> float:
        """the default value f_score of all nodes in the image

        Returns
        -------
        float
            returns infinity as the default f_score
        """
        return float("inf")
    
    def _find_neighbors_of(self, node: Node) -> List[Node]:
        """Finds the neighbors of a node (2D/3D)

        Parameters
        ----------
        node : Node
            the node whose neighbors we are interested in
        
        Returns
        -------
        List[Node]
            a list of nodes that are the neighbors of the given node
        """
        if len(node.point) == 2:
            return self._find_2D_neighbors_of(node)
        else:
            return self._find_3D_neighbors_of(node)
    
    def _find_2D_neighbors_of(self, node: Node) -> List[Node]:
        """Finds the neighbors of a 2D node

        Parameters
        ----------
        node : Node
            the node whose neighbors we are interested in

        Returns
        -------
        List[Node]
            a list of nodes that are the neighbors of the given node
        
        Notes
        -----
        - At max a given 2D node can have 8 neighbors-
        vertical neighbors: top, bottom,
        horizontal neighbors: left, right
        diagonal neighbors: top-left, top-right, bottom-left, bottom-right
        - Of course, we need to check for invalid cases where we can't move
        in these directions
        - 2D coordinates are of the type (y, x)
        """
        neighbors = []
        steps = [-1, 0, 1]
        for xdiff in steps:
            new_x = node.point[1] + xdiff
            if new_x < self.image_stats.x_min or new_x > self.image_stats.x_max:
                continue

            for ydiff in steps:
                if xdiff == ydiff == 0:
                    continue

                new_y = node.point[0] + ydiff
                if new_y < self.image_stats.y_min or new_y > self.image_stats.y_max:
                    continue

                new_point = np.array([new_y, new_x])

                h_for_new_point = self._estimate_cost_to_goal(new_point)

                intensity_at_new_point = self.image[new_y, new_x]

                cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(float(intensity_at_new_point))
                if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                    cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                g_for_new_point = node.g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff)) * cost_of_moving_to_new_point
                neighbor = Node(
                    point=new_point,
                    g_score=g_for_new_point,
                    h_score=h_for_new_point,
                    predecessor=node
                )

                neighbors.append(neighbor)

        return neighbors

    def _find_3D_neighbors_of(self, node: Node) -> List[Node]:
        """Finds the neighbors of a 3D node

        Parameters
        ----------
        node : Node
            the node whose neighbors we are interested in

        Returns
        -------
        List[Node]
            a list of nodes that are the neighbors of the given node
        
        Notes
        -----
        - At max a given 3D node can have 26 neighbors-
        Imagine a 3X3X3 3D cube. It will contain 27 nodes.
        If we consider the center node as the current node, it will have 26 neighbors
        (excluding itself.)
        - Of course, we need to check for invalid cases where we can't have
        26 neighbors (when the current node is closer to,
        or on the edges of the image)
        - 3D coordinates are of the form (z, y, x)
        """
        neighbors = []
        steps = [-1, 0, 1]
        
        for xdiff in steps:
            new_x = node.point[2] + xdiff
            if new_x < self.image_stats.x_min or new_x > self.image_stats.x_max:
                continue

            for ydiff in steps:
                new_y = node.point[1] + ydiff
                if new_y < self.image_stats.y_min or new_y > self.image_stats.y_max:
                    continue

                for zdiff in steps:
                    if xdiff == ydiff == zdiff == 0:
                        continue

                    new_z = node.point[0] + zdiff
                    if new_z < self.image_stats.z_min or new_z > self.image_stats.z_max:
                        continue

                    new_point = np.array([new_z, new_y, new_x])

                    h_for_new_point = self._estimate_cost_to_goal(new_point)

                    intensity_at_new_point = self.image[new_z, new_y, new_x]
                    cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(float(intensity_at_new_point))
                    if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                        cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                    g_for_new_point = node.g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff) + (zdiff*zdiff)) * cost_of_moving_to_new_point
                    neighbor = Node(
                        point=new_point,
                        g_score=g_for_new_point,
                        h_score=h_for_new_point,
                        predecessor=node
                    ) 

                    neighbors.append(neighbor)
                
        return neighbors
    
    def _found_goal(self, point: np.ndarray) -> bool:
        """Checks if the goal point is reached

        Parameters
        ----------
        point : numpy ndarray
            the point whose coordinates are to be compared to the goal
            point for equality

        Returns
        -------
        bool
            returns True if the goal is reached; False otherwise
        """
        return np.array_equal(point, self.goal_point)

    def _estimate_cost_to_goal(self, point: np.ndarray) -> float:
        """Estimates the heuristic cost (h_score)
        from a point to the goal point

        Parameters
        ----------
        point : numpy ndarray
            the point from which we have to estimate the heuristic cost to
            goal

        Returns
        -------
        float
            returns the heuristic cost between the point and goal point
        """
        return self.cost_function.minimum_step_cost() * self.heuristic_function.estimate_cost_to_goal(
            current_point=point, goal_point=self.goal_point
        )

    def _construct_path_from(self, node: Node):
        """constructs the brightest path upon reaching the goal_point by 
        backtracing steps from goal point to the start point
        
        Parameters
        ----------
        node : Node
            a node that lies on the brightest path
        """
        while node is not None:
            self.result.insert(0, node.point)
            node = node.predecessor
        