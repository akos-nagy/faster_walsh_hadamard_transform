# Fast Walsh-Hadamard Transform (FWHT)

A high-performance implementation of the Fast Walsh-Hadamard Transform using C++, OpenMP, and Pybind11.

Note: The current version only works with integer arrays. A floating point implementation will be added later.

## Installation

You can install the package directly from GitHub using pip:

!pip install git+https://github.com/akos-nagy/fast_walsh_hadamard_transform.git

## Usage Example

Below is a simple example of how to use the library:

```python
from fast_walsh_hadamard_transform import fwht
import numpy as np

# Create an array of random integers with length 2^4 = 16
x = (np.random.rand(2**4) * 10).astype(int)
# Example array:
# x = [8, 6, 5, 9, 2, 5, 4, 1, 2, 9, 1, 7, 0, 6, 6, 2]

# Compute the Fast Walsh-Hadamard Transform
y = fwht(x)
# Example output:
# y = [ 73, -17,   3, -11,  21, -13,   3,  21,  7,  13,  1,  11,  11,   9,  -7,   3]
```
