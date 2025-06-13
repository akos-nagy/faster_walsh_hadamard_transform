# Fast(er) Walsh&ndash;Hadamard Transform

A high-performance implementation of Fast Walsh&ndash;Hadamard Transform from the article *Faster Walsh&ndash;Hadamard and Discrete Fourier Transforms From Matrix Non-Rigidity* by Josh Alman and Kevin Rao [DOI](https://dl.acm.org/doi/10.1145/3564246.3585188). For an array for lenght $N = 2^n$, this method uses only $\left( \frac{11}{12} \log_2 (N) + \frac{1}{12} \right) N + o(N)$ additions and $\frac{1}{24} \log_2 (N) N + o(N)$ elementary binary operations. Written in C++, using OpenMP and Pybind11.

Note: The current version only works with integer arrays. Float implementation to be added later. A temporary approximate implementation for float arrays can be found in the **Usage Example** section below.

## Installation

You can install the package directly from GitHub using pip:

```sh
pip install git+https://github.com/akos-nagy/faster_walsh_hadamard_transform.git
```

## Usage Example

Below is a simple example of how to use the library:

```python
from faster_walsh_hadamard_transform import fwht
import numpy as np

# Create an array of random integers with length 2^4 = 16
data = np.random.randint(- 10, 10, 1 << 3)
print("Original data\t= ", data)
# Example array:
# data = [ 3,  8,  9,  8, -8,  3,  9,  7]

# Compute the Walsh–Hadamard Transform
result = fwht(data)
# Example output:
# result = [ 39, -13, -27, -19,  17,   5,  15,   7]
print("Original again\t= ", data)  # Should remain unchanged
print("Transformed\t= ", result)  # Transformed output
```

For float arrays:

```python
from faster_walsh_hadamard_transform import fwht
import numpy as np

def fwht_float(data, scale=1 << 40):
    return fwht(np.rint(data * scale)) / scale

data = np.array([-0.57727239, 0.56031148, 0.6037531, -0.18076892, -0.04983009, -0.23033542, 0.57870502, 0.56614574])
result = fwht_float(data)
# result = np.array([1.27070853, -0.15999724, -1.86496135, -1.75415983, -0.45866198, -0.54612644, 0.98507117, -2.09005194])
```
Testing suggests that the $L^2$ error of ```fwht_float``` for an array of lenght $2^n$ is $O \left( \tfrac{2^n}{\mathrm{scale}} \right)$.
