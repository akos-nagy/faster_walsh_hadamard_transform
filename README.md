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
x = np.rint(np.random.rand(1 << 4) * 10)
# Example array:
# x = [8, 6, 5, 9, 2, 5, 4, 1, 2, 9, 1, 7, 0, 6, 6, 2]

# Compute the Fast Walsh-Hadamard Transform
y = fwht(x)
# Example output:
# y = [ 73, -17,   3, -11,  21, -13,   3,  21,  7,  13,  1,  11,  11,   9,  -7,   3]
```
