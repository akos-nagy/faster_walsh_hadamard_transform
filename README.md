# Fast(er) Walsh-Hadamard Transform (FWHT)

A high-performance implementation of the Walsh-Hadamard Transform of Josh Alman and Kevin Rao from "Faster Walsh-Hadamard and Discrete Fourier Transforms From Matrix Non-Rigidity" https://dl.acm.org/doi/10.1145/3564246.3585188.

Written in C++, using OpenMP and Pybind11.

Note: The current version only works with integer arrays. Float implementation to be added later.

## Installation

You can install the package directly from GitHub using pip:

```sh
pip install git+https://github.com/akos-nagy/fast_walsh_hadamard_transform.git
```

## Usage Example

Below is a simple example of how to use the library:

```python
from fast_walsh_hadamard_transform import fwht
import numpy as np

# Create an array of random integers with length 2^4 = 16
data = np.random.randint(- 10, 10, 1 << 3)
# Example array:
# data = [ 3,  8,  9,  8, -8,  3,  9,  7]

# Compute the Fast Walsh-Hadamard Transform
result = fwht(data)
# Example output:
# result = [ 39, -13, -27, -19,  17,   5,  15,   7]
print("Original again\t= ", data)  # Should remain unchanged
print("Transformed\t= ", result)  # FWHT-transformed output
```
