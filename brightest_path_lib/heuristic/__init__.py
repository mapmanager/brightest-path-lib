from .heuristic import Heuristic
from .euclidean import Euclidean

DO_TRANSONIC = False
if DO_TRANSONIC:
    from .euclidean_transonic import EuclideanTransonic
else:
    from .euclidean import Euclidean as EuclideanTransonic
