import numpy as np
print("enter values like this: 32 2 4 89 (seperated by spaces)")
x = np.array(input("x values: ").split(), dtype=int)
y = np.array(input("y values: ").split(), dtype=int)
n = x.size

# Use least squares
div = n * np.sum(x*x) - (np.sum(x) ** 2)
a = (np.sum(y) * np.sum(x*x) - np.sum(x) * np.sum(x * y)) / div
b = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / div

print(f"Y = {a} + {b}x")
