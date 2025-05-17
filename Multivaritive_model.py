import numpy as np

# Calculating best model using Linear Algebra's least squares method
# Linear model: y = b0 + x1b1 + x2b2 + ... + xnbn
# Vector B = [b0, b1, b2, ... bn]  &  Vector X = [1, x1, x2, ... xn]
# Y = <B,X>     [<> represents the inner prod]

# Sample Dataset
x_in = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
y_out = np.array([12, 33, 44, 55])

# modify X matrix by adding column
X = np.insert(x_in, 0, values=1, axis=1)

Xt = np.linalg.matrix_transpose(X)
Xt_X_inv = np.linalg.pinv((np.linalg.matmul(Xt, X)))  # Moore-Penrose pseudo-inverse helps tackle with singular matrices
alpha = np.linalg.matmul(Xt_X_inv, Xt)      # Here alpha = (([X]^t[X])^-1)[X]^t
B = np.linalg.matmul(alpha, y_out)
