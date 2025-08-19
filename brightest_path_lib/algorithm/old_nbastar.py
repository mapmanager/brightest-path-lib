# algorithm/nbastar.py

""" 

*** DEPRECATED ***

The New Bidirectional A* Search Algorithm is an improvement over the
original Bidirectional A* Search algorithm, which is a variation of the
A* Search algorithm that searches from both the start and goal nodes 
simultaneously in order to find the shortest path more efficiently.

The New Bidirectional A* Search Algorithm was proposed to address a 
limitation of the original Bidirectional A* Search algorithm, which is 
that it often expands too many nodes and wastes computational resources. 
The new algorithm works by using two heuristic functions, one for the 
forward search and one for the backward search, and dynamically
adjusting them during the search.

The algorithm starts with two search trees, one rooted at the start node
and one rooted at the goal node. The forward search tree expands nodes 
in the direction of the goal node, and the backward search tree expands 
nodes in the direction of the start node. The search terminates when the
two trees meet in the middle, i.e., when they have a common node. 

During the search, the heuristic functions are dynamically adjusted
based on the cost of the path found so far. If the cost of the path
found so far is greater than the estimated cost of the path from the 
start node to the goal node, the heuristic function for the forward 
search is increased, and if the cost is less than the estimated cost, 
the heuristic function is decreased. The same adjustments are made 
to the heuristic function for the backward search.

This dynamic adjustment of the heuristic functions helps to reduce the
number of nodes expanded during the search and improve the efficiency
of the algorithm.

To search for the brightest path between two points in an image:

1. Initialize the NBAStarSearch class with the 2D/3D image,
   start point and the goal point: `nbastar = NBAStarSearch(image, start_point, goal_point)`
2. Call the search method: `path = nbastar.search()`
"""


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
    """NBA* Implementation

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
    open_nodes : Queue
        contains a list of points that are in the open set;
        can be used by the calling application to show a visualization
        of the algorithm's current search space
        Default value is None

    Attributes
    ----------
    image : numpy ndarray
        the image on which the brightest path search will be run
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
    open_nodes : Queue
        contains a list of points that are in the open set;
        can be used by the calling application to show a visualization
        of the algorithm's current search space
    node_priority_from_start : int
        a number given to a node whenever its added to the open set
        from start to goal; this is so that if we have to choose between
        two nodes with the same f_score, we choose the one which was added
        earlier to the open set, i.e, the one with lower priority 
    node_priority_from_goal : int
        a number given to a node whenever its added to the open set
        from start to goal; this is so that if we have to choose between
        two nodes with the same f_score, we choose the one which was added
        earlier to the open set, i.e, the one with lower priority
    open_set_from_start : PriorityQueue
        a priority queue containing tuples of the form:
        (f_score, node_priority_from_start, node);
        the node is what we will evaulate to find the brightest path from
        start to goal point 
    open_set_from_goal : PriorityQueue
        a priority queue containing tuples of the form:
        (f_score, node_priority_from_goal, node);
        the node is what we will evaulate to find the brightest path from
        goal to start point
    node_at_coordinates : Dict
        a mapping of a 2D/3D point to its corresponding node
    best_path_length : int
        this attribute is used to reject nodes that are too far; 
        initially initialized to infinity when the distance from start to goal
        or goal to start is unknown, it keeps shrinking to reflect the shortening
        of distance between our terminal points
    touch_node : BidirectionalNode
        the common node that is encountered when going from start to goal
        and goal to start
    is_canceled : bool
        should be set to True if the search needs to be stopped;
        false by default
    evaluated_nodes : int
        the number of nodes that have been evaluated to yet in search of the brightest path
    result : List[numpy ndarray]
        the result of the NBA* search containing the list of actual
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
        self.node_priority_from_start, self.node_priority_from_goal = 0, 0
        self.node_at_coordinates: Dict[Tuple, BidirectionalNode] = {}
        # self.close_set_hash_from_start = set() # hashset contains tuple of node coordinates already been visited
        # self.close_set_hash_from_goal = set()

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
        self.evaluated_nodes = 2 # since we will add the start and goal node to the open queue
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

    def search(self, verbose=False) -> List[np.ndarray]:
        """Performs bidirectional brightest path search

        Returns
        -------
        List[np.ndarray]
            the list containing the 2D/3D point coordinates
            that constitute the brightest path between the
            start_point and the goal_point
        
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

        self.open_set_from_start.put((0, self.node_priority_from_start, start_node)) # f_score, count: priority of occurence, current node
        self.open_set_from_goal.put((0, self.node_priority_from_goal, goal_node)) # f_score, count: priority of occurence, current node

        self.node_at_coordinates[tuple(self.start_point)] = start_node
        self.node_at_coordinates[tuple(self.goal_point)] = goal_node
        
        _countPixelsSearched = 0

        while not self.open_set_from_start.empty() and not self.open_set_from_goal.empty():
            if self.is_canceled:
                break

            _countPixelsSearched += 1

            from_start = self.open_set_from_start.qsize() < self.open_set_from_goal.qsize()
            if from_start:
                current_node = self.open_set_from_start.get()[2] # get the node object
                #current_coordinates = tuple(current_node.point)
                #self.close_set_hash_from_start.add(current_coordinates)
                
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
                current_node = self.open_set_from_goal.get()[2]
                #current_coordinates = tuple(current_node.point)
                #self.close_set_hash_from_goal.add(current_coordinates)
                
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

        # print(_countPixelsSearched)

        if not self.touch_node:
            if verbose:
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
        current_g_score = node.get_g(from_start) # optimization: will be the same for all neighbors

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

                # current_g_score = node.get_g(from_start)
                intensity_at_new_point = self.image[new_y, new_x]

                cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(float(intensity_at_new_point))
                if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                    cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                tentative_g_score = current_g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff)) * cost_of_moving_to_new_point
                tentative_h_score = self._estimate_cost_to_goal(new_point, self.goal_point if from_start else self.start_point)
                tentative_f_score = tentative_g_score + tentative_h_score
                self._is_touch_node(new_point, tentative_g_score, tentative_f_score, node, from_start)

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
        current_g_score = node.get_g(from_start)

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

                    # current_g_score = node.get_g(from_start)
                    intensity_at_new_point = self.image[new_z, new_y, new_x]

                    cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(float(intensity_at_new_point))
                    if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                        cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                    tentative_g_score = current_g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff) + (zdiff*zdiff)) * cost_of_moving_to_new_point

                    tentative_h_score = self._estimate_cost_to_goal(new_point, self.goal_point if from_start else self.start_point)

                    tentative_f_score = tentative_g_score + tentative_h_score
                    self._is_touch_node(new_point, tentative_g_score, tentative_f_score, node, from_start)

    def _is_touch_node(
        self,
        new_point: np.ndarray,
        tentative_g_score: float,
        tentative_f_score: float,
        predecessor: BidirectionalNode,
        from_start: bool
    ):
        """Modifies various parameters based on whether a given point
        has already been explored from one direction
        
        Parameters
        ----------
        new_point : numpy ndarray
            the coordinates of point that is being examined for touch node
        tentative_g_score : float
            the tentative g_score of the new point
        tentative_f_score: float
            the tentative f_score of the new point
        predecessor : BidirectionalNode
            the node that is predecessor of the current point
        from_start : bool
            True/False value representing our direction of traversal,
            True meaning we are traversing in from start to goal,
            False meaning traversal from goal to start
        """
        open_queue = self.open_set_from_start if from_start else self.open_set_from_goal
        
        new_point_coordinates = tuple(new_point)
        already_there = self.node_at_coordinates.get(new_point_coordinates, None)

        if not already_there:
            new_node = BidirectionalNode(new_point)
            new_node.set_g(tentative_g_score, from_start)
            new_node.set_f(tentative_f_score, from_start)
            new_node.set_predecessor(predecessor, from_start)
            self._increment_node_priority(from_start)
            open_queue.put((tentative_f_score, self._get_node_priority(from_start), new_node))
            if self.open_nodes:
                self.open_nodes.put(new_point_coordinates)
            self.evaluated_nodes += 1
            self.node_at_coordinates[new_point_coordinates] = new_node
        # elif self._in_closed_set(new_point_coordinates, from_start):
        #     return
        elif already_there.get_f(from_start) > tentative_f_score:
            already_there.set_g(tentative_g_score, from_start)
            already_there.set_f(tentative_f_score, from_start)
            already_there.set_predecessor(predecessor, from_start)
            self._increment_node_priority(from_start)
            open_queue.put((tentative_f_score, self._get_node_priority(from_start), already_there))
            if self.open_nodes:
                self.open_nodes.put(new_point_coordinates)
            self.evaluated_nodes += 1
            path_length = already_there.g_score_from_start + already_there.g_score_from_goal
            if path_length < self.best_path_length:
                self.best_path_length = path_length
                self.touch_node = already_there
    
    def _get_node_priority(self, from_start: bool) -> int:
        """Helper function to get a node's priority

        Parameters
        ----------
        from_start : bool
            if True, we want the node priority from start
            else, we want the node priority from goal
        
        Returns
        -------
        int
            returns the node priority from start/goal 
            based on the value of from start
        """
        return self.node_priority_from_start if from_start else self.node_priority_from_goal
    
    def _increment_node_priority(self, from_start: bool):
        """Helper function to increase the node priority
        so that it can be assigned to the next node to be added
        to the open set from start or open set from goal

        Parameters
        ----------
        from_start : bool
            if True, we increase the node priority from start
            else, we increase the node priority from goal
        """
        if from_start:
            self.node_priority_from_start += 1
        else:
            self.node_priority_from_goal += 1

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
        """constructs the brightest path upon reaching
        the touch node in two steps:
        1. Backtracks its steps from the touch node to the start node
        to insert the coordinates of all the points forming the brightest path
        in the result always at the 0th position
        2. Moves from touchnode to goal node to append the coordinates of
        all the points forming the brightest path
        """
        current_node = self.touch_node

        while not np.array_equal(current_node.point, self.start_point):
            self.result.insert(0, current_node.point)
            current_node = current_node.predecessor_from_start
        
        self.result.insert(0, self.start_point)

        current_node = self.touch_node.predecessor_from_goal

        while not np.array_equal(current_node.point, self.goal_point):
            self.result.append(current_node.point)
            current_node = current_node.predecessor_from_goal
        
        self.result.append(self.goal_point)
