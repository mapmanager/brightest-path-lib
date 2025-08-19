from numba import njit, float64
from brightest_path_lib.cost import Cost

# Standalone Numba-optimized function for the cost calculation
@njit(fastmath=True)
def _calculate_cost(intensity_at_new_point, min_intensity, max_intensity, 
                   reciprocal_min, reciprocal_max):
    """Numba-optimized cost calculation function"""
    # Normalize intensity
    intensity_at_new_point = reciprocal_max * (intensity_at_new_point - min_intensity) / (max_intensity - min_intensity)
    
    # Ensure minimum value - use max for better vectorization
    intensity_at_new_point = max(intensity_at_new_point, reciprocal_min)
    
    # Return reciprocal (1/intensity)
    return 1.0 / intensity_at_new_point

class Reciprocal(Cost):
    """Uses the reciprocal of pixel/voxel intensity to compute the cost of moving
    to a neighboring point. Optimized with Numba.

    Parameters
    ----------
    min_intensity : float
        The minimum intensity a pixel/voxel can have in a given image
    max_intensity : float
        The maximum intensity a pixel/voxel can have in a given image

    Attributes
    ----------
    RECIPROCAL_MIN : float
        To cope with zero intensities, RECIPROCAL_MIN is added to the intensities
        in the range before reciprocal calculation
    RECIPROCAL_MAX : float
        We set the maximum intensity <= RECIPROCAL_MAX so that the intensity
        is between RECIPROCAL MIN and RECIPROCAL_MAX
    """

    def __init__(self, min_intensity: float, max_intensity: float) -> None:
        super().__init__()
        if min_intensity is None or max_intensity is None:
            raise TypeError
        if min_intensity > max_intensity:
            raise ValueError
            
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.RECIPROCAL_MIN = float(1E-6)
        self.RECIPROCAL_MAX = 255.0
        self._min_step_cost = 1.0 / self.RECIPROCAL_MAX

    def cost_of_moving_to(self, intensity_at_new_point: float) -> float:
        """calculates the cost of moving to a point

        Parameters
        ----------
        intensity_at_new_point : float
            The intensity of the new point under consideration
        
        Returns
        -------
        float
            the cost of moving to the new point
        
        Notes
        -----
        - To cope with zero intensities, RECIPROCAL_MIN is added to the intensities in the range before reciprocal calculation
        - We set the maximum intensity <= RECIPROCAL_MAX so that the intensity is between RECIPROCAL MIN and RECIPROCAL_MAX
        """
        if intensity_at_new_point > self.max_intensity:
            raise ValueError
            
        # Use the Numba-optimized standalone function
        return _calculate_cost(
            intensity_at_new_point,
            self.min_intensity,
            self.max_intensity,
            self.RECIPROCAL_MIN,
            self.RECIPROCAL_MAX
        )
    
    def minimum_step_cost(self) -> float:
        """calculates the minimum step cost
        
        Returns
        -------
        float
            the minimum step cost
        """
        return self._min_step_cost