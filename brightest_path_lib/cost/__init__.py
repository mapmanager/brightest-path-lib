from .cost import Cost
from .reciprocal import Reciprocal

DO_TRANSONIC = False
if DO_TRANSONIC:
    from .reciprocal_transonic import ReciprocalTransonic
else:
    from .reciprocal import Reciprocal as ReciprocalTransonic
