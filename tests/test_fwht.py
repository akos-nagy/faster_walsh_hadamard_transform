import fast_walsh_hadamard_transform as fwht

data = np.random.randint(-10, 10, 1 << 4)
print("Original\t= ", data)

result = fwht.fwht(data)

print("Original again\t= ", data)  # Should remain unchanged
print("Transformed\t= ", result)  # FWHT-transformed output
