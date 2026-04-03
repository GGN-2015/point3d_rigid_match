import numpy as np
from scipy.optimize import linear_sum_assignment

def distance_signature(points, idx):
    """
    计算排序+归一化的距离指纹（刚性不变、抗噪）
    """
    d = np.linalg.norm(points[idx] - points, axis=1)
    d_sorted = np.sort(d)                  # 排序（关键！）
    d_norm = d_sorted / (d_sorted.sum() + 1e-8)  # 归一化
    return d_norm

def build_cost_matrix(A, B):
    N = len(A)
    cost = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        sigA = distance_signature(A, i)
        for j in range(N):
            sigB = distance_signature(B, j)
            # 用L2误差（比L1更适合连续匹配）
            cost[i, j] = np.sqrt(np.sum((sigA - sigB) ** 2))
    return cost

def find_correspondence(raw_A:list[list[float]], raw_B:list[list[float]]):
    A = np.array(raw_A)
    B = np.array(raw_B)
    N = len(raw_A)
    if len(A.shape) != 2 or A.shape[1] != 3:
        raise ValueError("A should be N * 3 matrix.")
    if len(B.shape) != 2 or B.shape[1] != 3:
        raise ValueError("B should be N * 3 matrix.")
    cost = build_cost_matrix(A, B)
    row_ind, col_ind = linear_sum_assignment(cost)
    corr = np.zeros(N, dtype=int)
    for i, j in zip(row_ind, col_ind):
        corr[i] = j
    return corr

if __name__ == "__main__":
    A = [
        [-0.8583011244293051, 1.664188145937392, -5.255867327332927],
        [-0.10348646934862409, 2.087722438630186, -3.5977804293199576],
        [-0.037025558771713755, 1.502559878964906, -5.803026123816429],
        [1.314519244586987, 1.8135837080551638, -4.756535447571685]
    ]

    B = [
        [0.9444699906120909, -0.00025015438771101525, 0.0018484750111346566],
        [-0.22389601437774168, -1.201344263998704, -0.0010553875203044244],
        [0.5505852240909018, 0.9189030480397202, -0.0016416782728876612],
        [-1.2711589253295885, 0.2826915078445261, 0.0008482241211740123]
    ]

    N = len(A)
    corr = find_correspondence(A, B)
    for i, j in enumerate(corr):
        print(f"A[{i}] -> B[{j}]")
