from collections import defaultdict
import math
import numpy as np
from queue import PriorityQueue, Queue
from typing import List, Tuple, Dict
from brightest_path_lib.cost import Reciprocal
from brightest_path_lib.heuristic import Euclidean
from brightest_path_lib.image import ImageStats
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node, BidirectionalNode


class NBAStarSearch:
    """Class that implements the New Bidirectional A-Star Search Algorithm

    Parameters
    ----------
    image : numpy ndarray
        the 2D/3D image on which we will run an A star search
    start_point : numpy ndarray
        the 2D/3D coordinates of the starting point (could be a pixel or a voxel)
        For 2D images, the coordinates are of the form (y, x)
        For 2D images, the coordinates are of the form (z, x, y)
    goal_point : numpy ndarray
        the 2D/3D coordinates of the goal point (could be a pixel or a voxel)
        For 2D images, the coordinates are of the form (y, x)
        For 2D images, the coordinates are of the form (z, x, y)
    scale : Tuple
        the scale of the image; defaults to (1.0, 1.0), i.e. image is not zoomed in/out
        For 2D images, the scale is of the form (x, y)
        For 2D images, the scale is of the form (x, y, z)
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
        the result of the A star search containing the list of actual
        points that constitute the brightest path from start_point to
        goal_point    
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
        self.open_set_from_start = PriorityQueue()
        self.open_set_from_goal = PriorityQueue()
        self.node_at_coordinates: Dict[Tuple, BidirectionalNode] = {}
        self.close_set_hash_from_start = set() # hashset contains tuple of node coordinates already been visited
        self.close_set_hash_from_goal = set()

        if cost_function == CostFunction.RECIPROCAL:
            self.cost_function = Reciprocal(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = Euclidean(scale=self.scale)
        
        self.best_path_length = float("inf")
        self.touch_node: BidirectionalNode = None
        self.is_canceled = False
        self.found_path = False
        self.result = []

    def _validate_inputs(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
    ):

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

    def search(self) -> List[np.ndarray]:
        """Function that performs A star search

        Returns
        -------
        List[np.ndarray]
            the list containing the 2D/3D point coordinates
            that constitute the brightest path between the
            start_point and the goal_point
        
        # Steps:

        """
        start_node = BidirectionalNode(point=self.start_point)
        goal_node = BidirectionalNode(point=self.goal_point)

        start_node.g_score_from_start = 0.0
        goal_node.g_score_from_goal = 0.0

        # since g_score from start to itself is 0, best f_score from start = h_score from start to goal
        best_f_score_from_start = self._estimate_cost_to_goal(self.start_point, self.goal_point)
        start_node.f_score_from_start = best_f_score_from_start

        # since g_score from goal to itself is 0, best f_score from goal = h_score from goal to start
        best_f_score_from_goal = self._estimate_cost_to_goal(self.goal_point, self.start_point)
        goal_node.f_score_from_goal = best_f_score_from_goal

        self.open_set_from_start.put((0, start_node)) # f_score, count: priority of occurence, current node
        self.open_set_from_goal.put((0, goal_node)) # f_score, count: priority of occurence, current node

        self.node_at_coordinates[tuple(self.start_point)] = start_node
        self.node_at_coordinates[tuple(self.goal_point)] = goal_node
        
        while not self.open_set_from_start.empty() and not self.open_set_from_goal.empty():
            if self.is_canceled:
                break

            from_start = self.open_set_from_start.qsize() < self.open_set_from_goal.qsize()
            if from_start:
                current_node = self.open_set_from_start.get()[1] # get the node object
                current_coordinates = tuple(current_node.point)
                self.close_set_hash_from_start.add(current_coordinates)
                
                best_f_score_from_start = current_node.f_score_from_start
                current_node_f_score = current_node.g_score_from_start + self._estimate_cost_to_goal(
                    current_point=current_node.point, 
                    goal_point=self.goal_point
                )

                if (current_node_f_score >= self.best_path_length) or ((current_node.g_score_from_start + best_f_score_from_goal - self._estimate_cost_to_goal(current_node.point, self.start_point)) >= self.best_path_length):
                    # reject the current node
                    continue
                else:
                    # stabilize the current node
                    self._expand_neighbors_of(current_node, from_start)
            else:
                current_node = self.open_set_from_goal.get()[1]
                current_coordinates = tuple(current_node.point)
                self.close_set_hash_from_goal.add(current_coordinates)
                
                best_f_score_from_goal = current_node.f_score_from_goal
                current_node_f_score = current_node.g_score_from_goal + self._estimate_cost_to_goal(
                    current_point=current_node.point, 
                    goal_point=self.start_point
                )

                if current_node_f_score >= self.best_path_length or ((current_node.g_score_from_goal + best_f_score_from_start - self._estimate_cost_to_goal(current_node.point, self.goal_point)) >= self.best_path_length):
                    # reject the current node
                    continue
                else:
                    # stabilize the current node
                    self._expand_neighbors_of(current_node, from_start)

        if not self.touch_node:
            print("NBA* Search finished without finding the path")
            return []
        
        self._construct_path()
        self.found_path = True
        return self.result
    
    def _expand_neighbors_of(self, node: BidirectionalNode, from_start: bool):
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
            return self._expand_2D_neighbors_of(node, from_start)
        else:
            return self._expand_3D_neighbors_of(node, from_start)
    
    def _expand_2D_neighbors_of(self, node: BidirectionalNode, from_start: bool):
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

                current_g_score = node.get_g(from_start)
                intensity_at_new_point = self.image[new_y, new_x]

                cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(intensity_at_new_point)
                if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                    cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                tentative_g_score = current_g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff)) * cost_of_moving_to_new_point
                tentative_h_score = self._estimate_cost_to_goal(new_point, self.goal_point if from_start else self.start_point)
                tentative_f_score = tentative_g_score + tentative_h_score
                self.is_touch_node(new_point, tentative_g_score, tentative_f_score, node, from_start)

    def _expand_3D_neighbors_of(self, node: BidirectionalNode, from_start: bool):
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
        - 3D coordinates are of the form (z, x, y)
        """
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

                    current_g_score = node.get_g(from_start)
                    intensity_at_new_point = self.image[new_z, new_y, new_x]

                    cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(intensity_at_new_point)
                    if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                        cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                    tentative_g_score = current_g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff) + (zdiff*zdiff)) * cost_of_moving_to_new_point
                    tentative_h_score = self._estimate_cost_to_goal(new_point, self.goal_point if from_start else self.start_point)
                    tentative_f_score = tentative_g_score + tentative_h_score
                    self.is_touch_node(new_point, tentative_g_score, tentative_f_score, node, from_start)

    def is_touch_node(
        self,
        new_point: np.ndarray,
        tentative_g_score: float,
        tentative_f_score: float,
        predecessor: BidirectionalNode,
        from_start: bool
    ):
        open_queue = self.open_set_from_start if from_start else self.open_set_from_goal
        new_point_coordinates = tuple(new_point)
        already_there = self.node_at_coordinates.get(new_point_coordinates, None)

        if not already_there:
            new_node = BidirectionalNode(new_point)
            new_node.set_g(tentative_g_score, from_start)
            new_node.set_f(tentative_f_score, from_start)
            new_node.set_predecessor(predecessor, from_start)
            open_queue.put((tentative_f_score, new_node))
            self.open_nodes.put(new_point_coordinates)
            self.node_at_coordinates[new_point_coordinates] = new_node
        elif self._in_closed_set(new_point_coordinates, from_start):
            return
        elif already_there.get_f(from_start) > tentative_f_score:
            already_there.set_g(tentative_g_score, from_start)
            already_there.set_f(tentative_f_score, from_start)
            already_there.set_predecessor(predecessor, from_start)
            open_queue.put((tentative_f_score, already_there))
            self.open_nodes.put(new_point_coordinates)
            path_length = already_there.g_score_from_start + already_there.g_score_from_goal
            if path_length < self.best_path_length:
                self.best_path_length = path_length
                self.touch_node = already_there

    def _in_closed_set(self, coordinates: Tuple, from_start: bool) -> bool:
        if from_start:
            return coordinates in self.close_set_hash_from_start
        else:
            return coordinates in self.close_set_hash_from_goal

    def _estimate_cost_to_goal(self, current_point: np.ndarray, goal_point: np.ndarray) -> float:
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
            current_point=current_point, goal_point=goal_point
        )
    
    def _construct_path(self):
        current_node = self.touch_node

        while not np.array_equal(current_node.point, self.start_point):
            self.result.insert(0, current_node.point)
            current_node = current_node.predecessor_from_start
             
        current_node = self.touch_node

        while not np.array_equal(current_node.point, self.goal_point):
            self.result.append(current_node.point)
            current_node = current_node.predecessor_from_goal