try:
    from numba import njit
except ModuleNotFoundError:
    # fallback: define a no-op decorator
     def njit(func=None, *args, **kwargs):
        if not(callable(func)):
            print(f"yes numba")
            return njit
        else:
            print(f"no numba")
            return func
    