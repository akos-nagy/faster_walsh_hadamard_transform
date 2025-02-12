import fast_walsh_hadamard_transform as fwht

data = [1, 2, 3, 4]
result = fwht.fwht(data)

print("Original:", data)  # Should remain unchanged
print("Transformed:", result)  # FWHT-transformed output
