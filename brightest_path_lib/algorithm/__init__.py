from .astar import AStarSearch

# Try to import the numba-optimized version first, fall back to non-numba version
try:
    import numba
    from .numba_nbastar import NBAStarSearch as NBAStarSearch
except (ModuleNotFoundError, ImportError):
    # numba not available, use the non-numba version
    from .nbastar import NBAStarSearch as NBAStarSearch