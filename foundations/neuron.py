import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
        weighted = x @ w
        biased = weighted + b

        if activation == "sigmoid":

            exps = np.exp(-biased)
            sigm = 1 / ( 1 + exps)

            return round(float(sigm), 5)
        elif activation == "relu":
            return round(float(max(0, biased)), 5)