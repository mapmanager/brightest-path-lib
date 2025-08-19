import numpy as np
import numba as nb

# Numba-optimized functions for faster image stats calculation
@nb.njit(fastmath=True)
def compute_image_intensity_range(image):
    """Efficiently compute min and max intensity of an image using Numba"""
    # Use Numba's optimized implementation rather than np.min/np.max
    # for better performance, especially with large arrays
    
    # Initialize with extreme values
    min_val = np.inf
    max_val = -np.inf
    
    # For 1D arrays (unlikely but handled for completeness)
    if image.ndim == 1:
        for i in range(image.shape[0]):
            val = image[i]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
    
    # For 2D arrays (most common case)
    elif image.ndim == 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                val = image[i, j]
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
    
    # For 3D arrays
    elif image.ndim == 3:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    val = image[i, j, k]
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val
    
    return float(min_val), float(max_val)

@nb.njit
def compute_image_dimensions(image_shape):
    """Compute image dimensions and coordinate ranges"""
    ndim = len(image_shape)
    
    # Initialize with default values
    x_min, y_min, z_min = 0, 0, 0
    x_max, y_max, z_max = 0, 0, 0
    
    if ndim == 2:  # 2D image
        y_max = image_shape[0] - 1
        x_max = image_shape[1] - 1
    elif ndim == 3:  # 3D image
        z_max = image_shape[0] - 1
        y_max = image_shape[1] - 1
        x_max = image_shape[2] - 1
    
    return x_min, x_max, y_min, y_max, z_min, z_max

class ImageStats:
    """Class holding metadata about an image, optimized with Numba

    Parameters
    ----------
    image : numpy ndarray
        the image who's metadata is being stored

    Attributes
    ----------
    min_intensity : float
        the minimum intensity of a pixel/voxel in the given image
    max_intensity : float
        the maximum intensity of a pixel/voxel in the given image
    x_min : int
        the smallest x-coordinate of the given image
    y_min : int
        the smallest y-coordinate of the given image
    z_min : int
        the smallest z-coordinate of the given image
    x_max : int
        the largest x-coordinate of the given image
    y_max : int
        the largest y-coordinate of the given image
    z_max : int
        the largest z-coordinate of the given image
    """
    # Use __slots__ for memory efficiency and faster attribute access
    __slots__ = (
        '_min_intensity', '_max_intensity',
        '_x_min', '_y_min', '_z_min',
        '_x_max', '_y_max', '_z_max'
    )

    def __init__(self, image: np.ndarray):
        # Input validation
        if image is None:
            raise TypeError("Image cannot be None")
        if len(image) == 0:
            raise ValueError("Image cannot be empty")
            
        # Convert image to a numpy array if it isn't already
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
            
        # Compute intensity range using Numba-optimized function
        min_intensity, max_intensity = compute_image_intensity_range(image)
        self._min_intensity = min_intensity
        self._max_intensity = max_intensity
        
        # Compute image dimensions and coordinate ranges
        self._x_min, self._x_max, self._y_min, self._y_max, self._z_min, self._z_max = compute_image_dimensions(image.shape)

    @property
    def min_intensity(self) -> float:
        return self._min_intensity

    @min_intensity.setter
    def min_intensity(self, value: float):
        if value is None:
            raise TypeError("min_intensity cannot be None")
        self._min_intensity = float(value)
    
    @property
    def max_intensity(self) -> float:
        return self._max_intensity
    
    @max_intensity.setter
    def max_intensity(self, value: float):
        if value is None:
            raise TypeError("max_intensity cannot be None")
        self._max_intensity = float(value)
    
    @property
    def x_min(self) -> float:
        return self._x_min
    
    @x_min.setter
    def x_min(self, value: float):
        if value is None:
            raise TypeError("x_min cannot be None")
        self._x_min = float(value)
    
    @property
    def y_min(self) -> float:
        return self._y_min
    
    @y_min.setter
    def y_min(self, value: float):
        if value is None:
            raise TypeError("y_min cannot be None")
        self._y_min = float(value)
    
    @property
    def z_min(self) -> float:
        return self._z_min
    
    @z_min.setter
    def z_min(self, value: float):
        if value is None:
            raise TypeError("z_min cannot be None")
        self._z_min = float(value)
    
    @property
    def x_max(self) -> float:
        return self._x_max
    
    @x_max.setter
    def x_max(self, value: float):
        if value is None:
            raise TypeError("x_max cannot be None")
        self._x_max = float(value)
    
    @property
    def y_max(self) -> float:
        return self._y_max
    
    @y_max.setter
    def y_max(self, value: float):
        if value is None:
            raise TypeError("y_max cannot be None")
        self._y_max = float(value)
    
    @property
    def z_max(self) -> float:
        return self._z_max
    
    @z_max.setter
    def z_max(self, value: float):
        if value is None:
            raise TypeError("z_max cannot be None")
        self._z_max = float(value)