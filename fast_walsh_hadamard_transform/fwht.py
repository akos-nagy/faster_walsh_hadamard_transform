import numpy as np
from .fast_walsh_hadamard_transform import fwht as _fwht_cpp

def fwht(arr):
    """Compute the Fast Walsh-Hadamard Transform.

    Args:
        arr (numpy.ndarray or list): Input array, must be a power of 2 in length.

    Returns:
        numpy.ndarray: FWHT-transformed array.
    """
    # Convert list to NumPy array if needed
    if isinstance(arr, list):
        arr = np.array(arr, dtype=np.int64)
    
    # Ensure NumPy array is of correct type
    if not isinstance(arr, np.ndarray) or arr.dtype != np.int64:
        raise TypeError("Input must be a NumPy array of dtype int64 or a list of integers.")

    # Check if length is a power of 2
    n = arr.shape[0]
    if n & (n - 1) != 0:
        raise ValueError("Input length must be a power of 2.")

    return _fwht_cpp(arr)

__all__ = ["fwht"]
