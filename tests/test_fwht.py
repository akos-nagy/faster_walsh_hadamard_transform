from faster_walsh_hadamard_transform import fwht
import numpy as np

import numpy as np
from faster_walsh_hadamard_transform import fwht
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
