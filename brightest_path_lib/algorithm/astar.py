from collections import defaultdict
import math
import numpy as np
from queue import PriorityQueue
from typing import List

from cost import Reciprocal
from heuristic import Euclidean
from image import ImageStats
from input import CostFunction, HeuristicFunction
from node import Node


class AStarSearch:
    """Class that implements the A-Star Search Algorithm

    Parameters:
    image : numpy ndarray
        the image where A star search is suppossed to run on
    start_point : numpy ndarray
        the coordinates of the start point
    goal_point : numpy ndarray
        the coordinates of the goal point
    scale : float
        the scale of the image.
        defaults to 1.0 if image is not zoomed in/out
    cost_function : numpy ndarray
        the cost function to be used for computing the cost of moving 
        to a new point. For more details, see the cost folder
    heuristic_function : numpy ndarray
        the heuristic function to be used to compute the estimated
        cost of moving from a point to the goal. For more details,
        see the heuristic folder.
    
    """

    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        scale: np.ndarray = np.array([1.0, 1.0]),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN
    ):
        self.image = image
        self.image_stats = ImageStats(image)
        self.start_point = start_point
        self.goal_point = goal_point
        self.scale = scale

        if cost_function == CostFunction.RECIPROCAL:
            self.cost_function = Reciprocal(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = Euclidean(scale=self.scale)
        
        self.result = []  

    def search(self):
        count = 0
        open_set = PriorityQueue()
        start_node = Node(
            point=self.start_point, 
            g_score=0, 
            h_score=self._estimate_cost_to_goal(self.start_point), 
            predecessor=None
            )
        open_set.put((0, count, start_node))
        open_set_hash = {tuple(self.start_point)}
        close_set_hash = set()
        f_scores = defaultdict(self._default_value)
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
        return float("inf")
    
    def _find_neighbors_of(self, node: Node) -> List[Node]:
        if len(node.point) == 2:
            return self._find_2D_neighbors_of(node)
        else:
            return self._find_3D_neighbors_of(node)
    
    def _find_2D_neighbors_of(self, node: Node) -> List[Node]:
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
                neighbor = Node(
                    point=new_point,
                    g_score=g_for_new_point,
                    h_score=h_for_new_point,
                    predecessor=node
                ) 

                neighbors.append(neighbor)

        return neighbors

    def _find_3D_neighbors_of(self, node: Node) -> List[Node]:
        """3D coordinates are of the form (z, x, y)
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
                    neighbor = Node(
                        point=new_point,
                        g_score=g_for_new_point,
                        h_score=h_for_new_point,
                        predecessor=node
                    ) 

                    neighbors.append(neighbor)
                
        return neighbors
    
    def _found_goal(self, point: np.ndarray) -> bool:
        return np.array_equal(point, self.goal_point)

    def _estimate_cost_to_goal(self, point: np.ndarray) -> float:
        return self.cost_function.minimum_step_cost() * self.heuristic_function.estimate_cost_to_goal(
            current_point=point, goal_point=self.goal_point
        )

    def _construct_path_from(self, node: Node):
        while node is not None:
            self.result.insert(0, node.point)
            node = node.predecessor
        