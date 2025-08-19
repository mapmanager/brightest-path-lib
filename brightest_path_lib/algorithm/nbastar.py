""" New Bidirectional A* Search Algorithm (NBA*)
Advanced optimized A* search implementation for finding the brightest path in an image.
This version includes additional performance optimizations beyond the previous version.

Developed by Github user: nipunarora8
"""

import heapq
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any, Optional
import numba as nb
from numba import types, prange, jit

# Import your original modules
from brightest_path_lib.cost import Reciprocal
from brightest_path_lib.heuristic import Euclidean
from brightest_path_lib.image import ImageStats
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node


# Further optimized Numba helper functions
@nb.njit(cache=True, inline='always')
def array_equal(arr1, arr2):
    """Numba-compatible implementation of np.array_equal with maximum optimization"""
    if arr1.shape != arr2.shape:
        return False
    return np.all(arr1 == arr2)


@nb.njit(fastmath=True, cache=True, inline='always')
def euclidean_distance_scaled(current_point, goal_point, scale_x, scale_y, scale_z=1.0):
    """Calculate scaled Euclidean distance between two points with maximum optimizations"""
    if len(current_point) == 2:  # 2D case
        x_diff = (goal_point[1] - current_point[1]) * scale_x
        y_diff = (goal_point[0] - current_point[0]) * scale_y
        return math.sqrt(x_diff * x_diff + y_diff * y_diff)
    else:  # 3D case
        x_diff = (goal_point[2] - current_point[2]) * scale_x
        y_diff = (goal_point[1] - current_point[1]) * scale_y
        z_diff = (goal_point[0] - current_point[0]) * scale_z
        return math.sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff)


# Pre-calculate direction arrays for neighbor finding - improves cache efficiency
directions_2d = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1]
], dtype=np.int32)

directions_3d = np.array([
    [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
    [-1, 0, -1],  [-1, 0, 0],  [-1, 0, 1],
    [-1, 1, -1],  [-1, 1, 0],  [-1, 1, 1],
    
    [0, -1, -1],  [0, -1, 0],  [0, -1, 1],
    [0, 0, -1],               [0, 0, 1],
    [0, 1, -1],   [0, 1, 0],   [0, 1, 1],
    
    [1, -1, -1],  [1, -1, 0],  [1, -1, 1],
    [1, 0, -1],   [1, 0, 0],   [1, 0, 1],
    [1, 1, -1],   [1, 1, 0],   [1, 1, 1]
], dtype=np.int32)

# Pre-calculate distances for 2D neighbors
distances_2d = np.array([
    math.sqrt(2), 1.0, math.sqrt(2),
    1.0,          1.0,
    math.sqrt(2), 1.0, math.sqrt(2)
], dtype=np.float32)

# Pre-calculate distances for 3D neighbors
distances_3d = np.array([
    math.sqrt(3), math.sqrt(2), math.sqrt(3),
    math.sqrt(2), 1.0,          math.sqrt(2),
    math.sqrt(3), math.sqrt(2), math.sqrt(3),
    
    math.sqrt(2), 1.0,          math.sqrt(2),
    1.0,                        1.0,
    math.sqrt(2), 1.0,          math.sqrt(2),
    
    math.sqrt(3), math.sqrt(2), math.sqrt(3),
    math.sqrt(2), 1.0,          math.sqrt(2),
    math.sqrt(3), math.sqrt(2), math.sqrt(3)
], dtype=np.float32)


@nb.njit(cache=True, parallel=False)
def find_2D_neighbors_optimized(node_point, g_score, image, x_min, x_max, y_min, y_max, 
                              min_intensity, max_intensity, reciprocal_min, reciprocal_max, 
                              min_step_cost, scale_x, scale_y, goal_point):
    """Find 2D neighbors using pre-calculated directions and distances"""
    neighbors = []
    max_min_diff = max_intensity - min_intensity
    
    # Use vectorized approach for better cache performance
    for i in range(len(directions_2d)):
        dir_y, dir_x = directions_2d[i]
        new_y = node_point[0] + dir_y
        new_x = node_point[1] + dir_x
        
        # Boundary check
        if new_x < x_min or new_x > x_max or new_y < y_min or new_y > y_max:
            continue
            
        new_point = np.array([new_y, new_x], dtype=np.int32)
        distance = distances_2d[i]
        
        # Calculate h_score
        h_score = min_step_cost * euclidean_distance_scaled(
            new_point, goal_point, scale_x, scale_y)
        
        # Calculate cost of moving (simplified calculation)
        intensity = float(image[new_y, new_x])
        norm_intensity = reciprocal_max * (intensity - min_intensity) / max_min_diff
        norm_intensity = max(norm_intensity, reciprocal_min)
        
        cost = max(1.0 / norm_intensity, min_step_cost)
        new_g_score = g_score + distance * cost
        
        neighbors.append((new_point, new_g_score, h_score))
    
    return neighbors


@nb.njit(cache=True, parallel=False)
def find_3D_neighbors_optimized(node_point, g_score, image, x_min, x_max, y_min, y_max, z_min, z_max,
                              min_intensity, max_intensity, reciprocal_min, reciprocal_max, 
                              min_step_cost, scale_x, scale_y, scale_z, goal_point):
    """Find 3D neighbors using pre-calculated directions and distances"""
    neighbors = []
    max_min_diff = max_intensity - min_intensity
    
    # Use vectorized approach for better cache performance
    for i in range(len(directions_3d)):
        dir_z, dir_y, dir_x = directions_3d[i]
        
        # Skip center point
        if dir_z == 0 and dir_y == 0 and dir_x == 0:
            continue
            
        new_z = node_point[0] + dir_z
        new_y = node_point[1] + dir_y
        new_x = node_point[2] + dir_x
        
        # Boundary check
        if (new_x < x_min or new_x > x_max or 
            new_y < y_min or new_y > y_max or 
            new_z < z_min or new_z > z_max):
            continue
            
        new_point = np.array([new_z, new_y, new_x], dtype=np.int32)
        distance = distances_3d[i]
        
        # Calculate h_score
        h_score = min_step_cost * euclidean_distance_scaled(
            new_point, goal_point, scale_x, scale_y, scale_z)
        
        # Calculate cost of moving (simplified calculation)
        intensity = float(image[new_z, new_y, new_x])
        norm_intensity = reciprocal_max * (intensity - min_intensity) / max_min_diff
        norm_intensity = max(norm_intensity, reciprocal_min)
        
        cost = max(1.0 / norm_intensity, min_step_cost)
        new_g_score = g_score + distance * cost
        
        neighbors.append((new_point, new_g_score, h_score))
    
    return neighbors


# Optimized bidirectional A* search
class NBAStarSearch:
    """Advanced bidirectional A* search implementation
    
    This implementation searches from both start and goal simultaneously,
    which can be much faster for large images.
    """

    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        scale: Tuple = (1.0, 1.0),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN,
        open_nodes=None,
        use_hierarchical: bool = False,
        weight_heuristic: float = 1.0
    ):
        """Initialize bidirectional A* search
        
        Parameters
        ----------
        image : numpy ndarray
            The image to search
        start_point, goal_point : numpy ndarray
            Start and goal coordinates
        scale : tuple
            Image scale factors
        cost_function, heuristic_function : Enum
            Functions to use for cost and heuristic
        open_nodes : Queue, optional
            Queue for visualization
        use_hierarchical : bool
            Whether to use hierarchical search for large images
        weight_heuristic : float
            Weight for heuristic (> 1.0 makes search faster but less optimal)
        """
        self._validate_inputs(image, start_point, goal_point)

        # Convert to int32 for better performance
        self.image = image
        self.image_stats = ImageStats(image)
        self.start_point = np.round(start_point).astype(np.int32)
        self.goal_point = np.round(goal_point).astype(np.int32)
        self.scale = scale
        self.open_nodes = open_nodes
        self.weight_heuristic = weight_heuristic
        self.use_hierarchical = use_hierarchical

        # Configuration for reciprocal cost function
        if cost_function == CostFunction.RECIPROCAL:
            self.cost_function = Reciprocal(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = Euclidean(scale=self.scale)
        
        # State variables
        self.is_canceled = False
        self.found_path = False
        self.evaluated_nodes = 0
        self.result = []

        # For hierarchical search
        if use_hierarchical and max(image.shape) > 1000:
            # Downsampled image for initial path finding
            self.downsampled_image = self._create_downsampled_image()
        else:
            self.downsampled_image = None

    def _validate_inputs(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
    ):
        """Validate input parameters"""
        if image is None or start_point is None or goal_point is None:
            raise TypeError("Image, start_point, and goal_point cannot be None")
        if len(image) == 0 or len(start_point) == 0 or len(goal_point) == 0:
            raise ValueError("Image, start_point, and goal_point cannot be empty")

    def _create_downsampled_image(self, factor=4):
        """Create a downsampled image for hierarchical search"""
        if len(self.image.shape) == 2:  # 2D image
            h, w = self.image.shape
            new_h, new_w = h // factor, w // factor
            downsampled = np.zeros((new_h, new_w), dtype=self.image.dtype)
            
            # Take maximum values to preserve bright paths
            for i in range(new_h):
                for j in range(new_w):
                    y_start, y_end = i*factor, min((i+1)*factor, h)
                    x_start, x_end = j*factor, min((j+1)*factor, w)
                    downsampled[i, j] = np.max(self.image[y_start:y_end, x_start:x_end])
                    
            return downsampled
        else:  # 3D image
            d, h, w = self.image.shape
            new_d, new_h, new_w = d // factor, h // factor, w // factor
            downsampled = np.zeros((new_d, new_h, new_w), dtype=self.image.dtype)
            
            for i in range(new_d):
                for j in range(new_h):
                    for k in range(new_w):
                        z_start, z_end = i*factor, min((i+1)*factor, d)
                        y_start, y_end = j*factor, min((j+1)*factor, h)
                        x_start, x_end = k*factor, min((k+1)*factor, w)
                        downsampled[i, j, k] = np.max(self.image[z_start:z_end, 
                                                             y_start:y_end, 
                                                             x_start:x_end])
            return downsampled

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

    def search(self, verbose: bool = False) -> List[np.ndarray]:
        """Perform bidirectional A* search
        
        This method searches from both the start and goal simultaneously,
        which can dramatically reduce the search space.
        
        Returns
        -------
        List[np.ndarray]
            Path from start to goal
        """
        # If we're using hierarchical search for large images
        if self.use_hierarchical and self.downsampled_image is not None:
            if verbose:
                print("Using hierarchical search...")
            # First find path in downsampled image
            rough_path = self._hierarchical_search()
            if not rough_path:
                # If hierarchical search failed, fall back to normal search
                return self._bidirectional_search(verbose)
            
            # Refine path in original image
            return self._refine_path(rough_path)
        else:
            # Regular bidirectional search
            return self._bidirectional_search(verbose)

    def _hierarchical_search(self):
        """Perform search on downsampled image to get approximate path"""
        # TODO: Implement hierarchical search for initial path estimate
        # This would find a coarse path in the downsampled image
        # The code could be similar to _bidirectional_search but using downsampled
        # coordinates and image
        return None  # For now, we'll just fall back to regular search

    def _refine_path(self, rough_path):
        """Refine a coarse path from hierarchical search"""
        # TODO: Implement path refinement
        # This would take the coarse path and refine it in the original image
        return None  # For now we'll just return the rough path (downsample factor)

    def _bidirectional_search(self, verbose: bool = False) -> List[np.ndarray]:
        """Perform bidirectional A* search from start and goal simultaneously"""
        # Forward search (start to goal)
        open_heap_fwd = []
        count_fwd = [0]  # Use a list for mutable reference
        
        start_node = Node(
            point=self.start_point, 
            g_score=0, 
            h_score=self._estimate_cost_to_goal(self.start_point, self.goal_point), 
            predecessor=None
        )
        
        heapq.heappush(open_heap_fwd, (start_node.f_score, count_fwd[0], start_node))
        open_nodes_dict_fwd = {tuple(self.start_point): (0, start_node.f_score, start_node)}
        closed_set_fwd = set()
        
        # Backward search (goal to start)
        open_heap_bwd = []
        count_bwd = [0]  # Use a list for mutable reference
        
        goal_node = Node(
            point=self.goal_point, 
            g_score=0, 
            h_score=self._estimate_cost_to_goal(self.goal_point, self.start_point), 
            predecessor=None
        )
        
        heapq.heappush(open_heap_bwd, (goal_node.f_score, count_bwd[0], goal_node))
        open_nodes_dict_bwd = {tuple(self.goal_point): (0, goal_node.f_score, goal_node)}
        closed_set_bwd = set()
        
        # Extract parameters for neighbor finding
        scale_x, scale_y = self.scale[0], self.scale[1]
        scale_z = 1.0 if len(self.scale) <= 2 else self.scale[2]
        
        min_intensity = self.image_stats.min_intensity
        max_intensity = self.image_stats.max_intensity
        x_min, x_max = self.image_stats.x_min, self.image_stats.x_max
        y_min, y_max = self.image_stats.y_min, self.image_stats.y_max
        z_min, z_max = self.image_stats.z_min, self.image_stats.z_max
        
        reciprocal_min = self.cost_function.RECIPROCAL_MIN
        reciprocal_max = self.cost_function.RECIPROCAL_MAX
        min_step_cost = self.cost_function.minimum_step_cost()
        
        # Best meeting point found so far
        best_meeting_point = None
        best_meeting_cost = float('inf')
        best_fwd_node = None
        best_bwd_node = None
        
        # Main bidirectional search loop
        while open_heap_fwd and open_heap_bwd and not self.is_canceled:
            # Decide which direction to expand
            # Alternate between forward and backward search
            if len(open_heap_fwd) <= len(open_heap_bwd):
                # Expand forward search
                success = self._expand_search(
                    open_heap_fwd, open_nodes_dict_fwd, closed_set_fwd,
                    open_nodes_dict_bwd, closed_set_bwd,
                    True, count_fwd,
                    x_min, x_max, y_min, y_max, z_min, z_max,
                    min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                    min_step_cost, scale_x, scale_y, scale_z,
                    best_meeting_point, best_meeting_cost, best_fwd_node, best_bwd_node
                )
                if success:
                    best_meeting_point, best_meeting_cost, best_fwd_node, best_bwd_node = success
            else:
                # Expand backward search
                success = self._expand_search(
                    open_heap_bwd, open_nodes_dict_bwd, closed_set_bwd,
                    open_nodes_dict_fwd, closed_set_fwd,
                    False, count_bwd,
                    x_min, x_max, y_min, y_max, z_min, z_max,
                    min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                    min_step_cost, scale_x, scale_y, scale_z,
                    best_meeting_point, best_meeting_cost, best_fwd_node, best_bwd_node
                )
                if success:
                    best_meeting_point, best_meeting_cost, best_fwd_node, best_bwd_node = success
            
            # Check if search is complete
            if best_meeting_point is not None:
                # Check if we should continue searching or terminate
                # terminate if fwd_heap.min + bwd_heap.min >= best_meeting_cost
                min_f_fwd = open_heap_fwd[0][0] if open_heap_fwd else float('inf')
                min_f_bwd = open_heap_bwd[0][0] if open_heap_bwd else float('inf')
                
                if min_f_fwd + min_f_bwd >= best_meeting_cost:
                    if verbose:
                        print(f"Found meeting point at {best_meeting_point} with cost {best_meeting_cost}")
                    self.found_path = True
                    self._construct_bidirectional_path(best_fwd_node, best_bwd_node)
                    break
        
        self.evaluated_nodes = count_fwd[0] + count_bwd[0]
        return self.result

    def _expand_search(self, open_heap, open_nodes_dict, closed_set,
                     other_open_dict, other_closed_set,
                     is_forward, count_ref,
                     x_min, x_max, y_min, y_max, z_min, z_max,
                     min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                     min_step_cost, scale_x, scale_y, scale_z,
                     best_meeting_point, best_meeting_cost, best_fwd_node, best_bwd_node):
        """Expand search in one direction (forward or backward)"""
        if not open_heap:
            return None
            
        # Get node with lowest f_score
        _, _, current_node = heapq.heappop(open_heap)
        current_coordinates = tuple(current_node.point)
        
        # Skip if already processed
        if current_coordinates in closed_set:
            return None
            
        # Remove from open nodes dict
        if current_coordinates in open_nodes_dict:
            del open_nodes_dict[current_coordinates]
        
        # Get target for this search direction
        target_point = self.goal_point if is_forward else self.start_point
        
        # Find neighbors
        if len(current_node.point) == 2:  # 2D
            neighbor_data = find_2D_neighbors_optimized(
                current_node.point, current_node.g_score, self.image,
                x_min, x_max, y_min, y_max, 
                min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                min_step_cost, scale_x, scale_y, target_point
            )
        else:  # 3D
            neighbor_data = find_3D_neighbors_optimized(
                current_node.point, current_node.g_score, self.image,
                x_min, x_max, y_min, y_max, z_min, z_max,
                min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                min_step_cost, scale_x, scale_y, scale_z, target_point
            )
        
        # Store nodes from closed set for meeting point detection
        closed_nodes_with_data = []
        
        # Process neighbors
        for new_point, g_score, h_score in neighbor_data:
            neighbor_coordinates = tuple(new_point)
            
            # Skip if already processed
            if neighbor_coordinates in closed_set:
                continue
            
            # Apply weighted heuristic (makes search faster but less optimal)
            f_score = g_score + self.weight_heuristic * h_score
            
            # Check if we should update this neighbor
            if neighbor_coordinates in open_nodes_dict:
                current_g, current_f, _ = open_nodes_dict[neighbor_coordinates]
                if g_score >= current_g:  # If not a better path, skip
                    continue
            
            # Either a new node or a better path to existing node
            neighbor = Node(
                point=new_point,
                g_score=g_score,
                h_score=h_score,
                predecessor=current_node
            )
            
            # Update open nodes dictionary
            open_nodes_dict[neighbor_coordinates] = (g_score, f_score, neighbor)
            
            # Add to heap - increment the counter
            count_ref[0] += 1
            local_count = count_ref[0]
            heapq.heappush(open_heap, (f_score, local_count, neighbor))
            
            # Update visualization queue if needed
            if self.open_nodes is not None:
                self.open_nodes.put(neighbor_coordinates)
            
            # Check if this node connects the two searches
            if neighbor_coordinates in other_open_dict:
                # We've found a potential meeting point in open set
                other_g, _, other_node = other_open_dict[neighbor_coordinates]
                
                # Calculate total cost of path
                meeting_cost = g_score + other_g
                
                # Check if this is the best meeting point so far
                if meeting_cost < best_meeting_cost:
                    if is_forward:
                        new_best_fwd_node = neighbor
                        new_best_bwd_node = other_node
                    else:
                        new_best_fwd_node = other_node
                        new_best_bwd_node = neighbor
                        
                    return (neighbor_coordinates, meeting_cost, 
                           new_best_fwd_node, new_best_bwd_node)
        
        # Mark as processed
        closed_set.add(current_coordinates)
        
        return None

    def _estimate_cost_to_goal(self, point: np.ndarray, target: np.ndarray) -> float:
        """Estimate heuristic cost between two points"""
        scale = self.scale
        
        if len(point) == 2:  # 2D
            return self.cost_function.minimum_step_cost() * euclidean_distance_scaled(
                point, target, scale[0], scale[1])
        else:  # 3D
            return self.cost_function.minimum_step_cost() * euclidean_distance_scaled(
                point, target, scale[0], scale[1], scale[2] if len(scale) > 2 else 1.0)

    def _construct_bidirectional_path(self, forward_node: Node, backward_node: Node):
        """Construct path from meeting point of bidirectional search"""
        # Forward path (start to meeting point)
        forward_path = []
        current = forward_node
        while current is not None:
            forward_path.append(current.point)
            current = current.predecessor
        
        # Reverse to get start-to-meeting-point order
        forward_path.reverse()
        
        # Backward path (goal to meeting point)
        backward_path = []
        current = backward_node
        while current is not None:
            backward_path.append(current.point)
            current = current.predecessor
        
        # Combine paths (remove duplicate meeting point)
        self.result = forward_path + backward_path[1:]