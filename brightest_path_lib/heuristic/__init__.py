from .heuristic import Heuristic

DO_TRANSONIC = False
if DO_TRANSONIC:
    from .euclidean_transonic import EuclideanTransonic as Euclidean
else:
    from .euclidean import Euclidean as Euclidean
