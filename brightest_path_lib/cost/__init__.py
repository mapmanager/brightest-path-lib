from .cost import Cost

DO_TRANSONIC = False
if DO_TRANSONIC:
    from .reciprocal_transonic import ReciprocalTransonic as Reciprocal
else:
    from .reciprocal import Reciprocal as Reciprocal
