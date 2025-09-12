from transonic import boost
from brightest_path_lib.cost import Cost

@boost
class ReciprocalTransonic(Cost):
    """Uses the reciprocal of pixel/voxel intensity to compute the cost of moving
    to a neighboring point

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

    min_intensity: float
    max_intensity: float
    RECIPROCAL_MIN: float
    RECIPROCAL_MAX: float
    _min_step_cost: float

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


    @boost
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

        intensity_at_new_point = self.RECIPROCAL_MAX * (intensity_at_new_point - self.min_intensity) / (self.max_intensity - self.min_intensity)

        if intensity_at_new_point < self.RECIPROCAL_MIN:
            intensity_at_new_point = self.RECIPROCAL_MIN
        
        return 1.0 / intensity_at_new_point
    
    @boost
    def minimum_step_cost(self) -> float:
        """calculates the minimum step cost
        
        Returns
        -------
        float
            the minimum step cost
        """
        return self._min_step_cost
