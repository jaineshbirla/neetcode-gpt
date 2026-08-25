import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert lists to NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # ---------- Forward Pass ----------

        z1 = x @ W1.T + b1

        a1 = np.maximum(0, z1)

        z2 = a1 @ W2.T + b2

        # MSE
        loss = np.mean((z2 - y_true) ** 2)

        # ---------- Backward Pass ----------

        # dL/dz2
        n = y_true.size
        dz2 = 2 * (z2 - y_true) / n

        # dL/dW2
        dW2 = np.outer(dz2, a1)

        # dL/db2
        db2 = dz2

        # dL/da1
        da1 = dz2 @ W2

        # dL/dz1 through ReLU
        dz1 = da1 * (z1 > 0)

        # dL/dW1
        dW1 = np.outer(dz1, x)

        # dL/db1
        db1 = dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }