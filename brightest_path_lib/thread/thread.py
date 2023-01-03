from threading import Thread
import numpy as np
from queue import Queue
from brightest_path_lib.algorithm import AStarSearch
from brightest_path_lib.input import CostFunction, HeuristicFunction

class AStarThread(Thread):
    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        scale: np.ndarray = np.array([1.0, 1.0]),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN
    ):
        super().__init__(daemon=True)
        self.open_nodes = Queue()
        self.result = []

        self.aStarSearch = AStarSearch(
            image,
            start_point,
            goal_point,
            scale,
            cost_function,
            heuristic_function,
            self.open_nodes,
        )
    
    def cancel(self):
        self.aStarSearch.is_canceled = True
    
    def run(self):
        self.result = self.aStarSearch.search()
