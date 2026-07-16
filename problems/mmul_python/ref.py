"""Python reference: column-major matrix multiply C = A @ B.

Inputs are column-major flat arrays:
  A: M rows x K cols, stored as A_flat[col*M + row]
  B: K rows x N cols, stored as B_flat[col*K + row]  (transposed from CUDA ref)
Output is column-major:
  C: M rows x N cols, stored as C_flat[col*M + row]

Computes C[row,col] = sum_k A[row,k] * B[k,col]
"""

import numpy as np


def gemm_reference(scalars, buffers):
    M = scalars["kSizeM"]
    N = scalars["kSizeN"]
    K = scalars["kSizeK"]
    A = np.frombuffer(buffers["mat_a"].tobytes(), dtype=np.float32).reshape(M, K, order="F")
    B = np.frombuffer(buffers["mat_b"].tobytes(), dtype=np.float32).reshape(K, N, order="F")
    C = A @ B
    return C.ravel(order="F")
