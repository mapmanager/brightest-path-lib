from collections import defaultdict
import math
import numpy as np
from queue import PriorityQueue
from typing import List

import sys
sys.path.append("..")
from cost import reciprocal
from heuristic import euclidean
from image import stats
from input import inputs
from node import node
# from cost import Reciprocal
# from heuristic import Euclidean
# from image import ImageStats
# from input import CostFunction, HeuristicFunction
# from node import Node


class AStarSearch:
    """Class that implements the A-Star Search Algorithm

    Parameters
    ----------
    image : numpy ndarray
        the 2D/3D image on which we will run an A star search
    start_point : numpy ndarray
        the 2D/3D coordinates of the starting point (could be a pixel or a voxel)
    goal_point : numpy ndarray
        the 2D/3D coordinates of the goal point (could be a pixel or a voxel)
    scale : float
        the scale of the image; defaults to 1.0 if image is not zoomed in/out
    cost_function : Enum CostFunction
        this enum value specifies the cost function to be used for computing 
        the cost of moving to a new point
        Default type is CostFunction.RECIPROCAL to use the reciprocal function
    heuristic_function : Enum HeuristicFunction
        this enum value specifies the heuristic function to be used to compute
        the estimated cost of moving from a point to the goal
        Default type is HeuristicFunction.EUCLIDEAN to use the 
        euclidean function for cost estimation

    Attributes
    ----------
    image : numpy ndarray
        the image where A star search is suppossed to run on
    start_point : numpy ndarray
        the coordinates of the start point
    goal_point : numpy ndarray
        the coordinates of the goal point
    scale : float
        the scale of the image; defaults to 1.0 if image is not zoomed in/out
    cost_function : Cost
        the cost function to be used for computing the cost of moving 
        to a new point
        Default type is Reciprocal
    heuristic_function : Heuristic
        the heuristic function to be used to compute the estimated
        cost of moving from a point to the goal
        Default type is Euclidean
    result : List[numpy ndarray]
        the result of the A star search containing the list of actual
        points that constitute the brightest path from start_point to
        goal_point    
    """

    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        scale: np.ndarray = np.array([1.0, 1.0]),
        cost_function: inputs.CostFunction = inputs.CostFunction.RECIPROCAL,
        heuristic_function: inputs.HeuristicFunction = inputs.HeuristicFunction.EUCLIDEAN
    ):
        self.image = image
        self.image_stats = stats.ImageStats(image)
        self.start_point = start_point
        self.goal_point = goal_point
        self.scale = scale

        if cost_function == inputs.CostFunction.RECIPROCAL:
            self.cost_function = reciprocal.Reciprocal(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == inputs.HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = euclidean.Euclidean(scale=self.scale)
        
        self.result = []  

    def search(self) -> List[np.ndarray]:
        """Function that performs A star search

        Returns
        -------
        List[np.ndarray]
            the list containing the 2D/3D point coordinates
            that constitute the brightest path between the
            start_point and the goal_point
        """
        count = 0
        open_set = PriorityQueue()
        start_node = node.Node(
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
            current_node = open_set.get()[2]
            current_coordinates = tuple(current_node.point)
            if current_coordinates in close_set_hash:
                continue
            
            open_set_hash.remove(current_coordinates)

            if self._found_goal(current_node.point):
                print("Found goal!")
                self._construct_path_from(current_node)
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
                else:
                    if neighbor.f_score < f_scores[neighbor_coordinates]:
                        f_scores[neighbor_coordinates] = neighbor.f_score
                        count += 1
                        open_set.put((neighbor.f_score, count, neighbor))
            
            close_set_hash.add(current_coordinates)

        return self.result
    
    def _default_value(self) -> float:
        """the default value f_score of all nodes in the image

        Returns
        -------
        float
            returns infinity as the default f_score
        """
        return float("inf")
    
    def _find_neighbors_of(self, node: node.Node) -> List[node.Node]:
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
    
    def _find_2D_neighbors_of(self, node: node.Node) -> List[node.Node]:
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
        - 2D coordinates are of the type (x, y)
        """
        neighbors = []
        steps = [-1, 0, 1]
        for xdiff in steps:
            for ydiff in steps:
                if xdiff == ydiff == 0:
                    continue

                new_x = node.point[0] + xdiff
                if new_x < self.image_stats.x_min or new_x > self.image_stats.x_max:
                    continue
                    
                new_y = node.point[1] + ydiff
                if new_y < self.image_stats.y_min or new_y > self.image_stats.y_max:
                    continue

                new_point = np.array([new_x, new_y])

                h_for_new_point = self._estimate_cost_to_goal(new_point)

                intensity_at_new_point = self.image[new_x, new_y]
                cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(intensity_at_new_point)
                if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                    cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                g_for_new_point = node.g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff)) * cost_of_moving_to_new_point
                neighbor = node.Node(
                    point=new_point,
                    g_score=g_for_new_point,
                    h_score=h_for_new_point,
                    predecessor=node
                ) 

                neighbors.append(neighbor)

        return neighbors

    def _find_3D_neighbors_of(self, node: node.Node) -> List[node.Node]:
        """Finds the neighbors of a 3D node

        Parameters
        ----------
        node : node.Node
            the node whose neighbors we are interested in

        Returns
        -------
        List[node.Node]
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
        - 3D coordinates are of the form (z, x, y)
        """
        neighbors = []
        steps = [-1, 0, 1]
        
        for xdiff in steps:
            for ydiff in steps:
                for zdiff in steps:
                    if xdiff == ydiff == zdiff == 0:
                        continue

                    # new_z = node.point[2] + zdiff
                    new_z = node.point[0] + zdiff
                    if new_z < self.image_stats.z_min or new_z > self.image_stats.z_max:
                        continue

                    # new_x = node.point[0] + xdiff
                    new_x = node.point[1] + xdiff
                    if new_x < self.image_stats.x_min or new_x > self.image_stats.x_max:
                        continue
                        
                    # new_y = node.point[1] + ydiff
                    new_y = node.point[2] + ydiff
                    if new_y < self.image_stats.y_min or new_y > self.image_stats.y_max:
                        continue

                    # new_point = np.array([new_x, new_y, new_z])
                    new_point = np.array([new_z, new_x, new_y])

                    h_for_new_point = self._estimate_cost_to_goal(new_point)

                    # intensity_at_new_point = self.image[new_x, new_y, new_z]
                    intensity_at_new_point = self.image[new_z, new_x, new_y]
                    cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(intensity_at_new_point)
                    if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                        cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                    g_for_new_point = node.g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff) + (zdiff*zdiff)) * cost_of_moving_to_new_point
                    neighbor = node.Node(
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
        """Estimates the heuristic cost (h_score) between a point
        and the goal

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

    def _construct_path_from(self, node: node.Node):
        """stores the coodinates of nodes that constitute the brightest path 
        from the goal_point to the start_point once the goal is reached.

        Parameters
        ----------
        node : Node
            a node that lies on the brightest path
        """
        while node is not None:
            self.result.insert(0, node.point)
            node = node.predecessor
        