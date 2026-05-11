import numpy as np
from itertools import permutations

# ------------------- 新增：暴力k优解替换linear_sum_assignment -------------------
def k_best_linear_sum_assignment(cost_matrix, rank_idx=0):
    """
    暴力枚举所有完美匹配，返回第 rank_idx 优解（0=最优）
    """
    n = cost_matrix.shape[0]
    # 生成所有可能的列分配（完美匹配）
    all_cols = list(permutations(range(n)))
    # 计算每个匹配的总代价
    solutions = []
    for cols in all_cols:
        cost = cost_matrix[np.arange(n), cols].sum()
        solutions.append((cost, cols))
    # 按代价升序排序
    solutions.sort(key=lambda x: x[0])
    # 取第 rank_idx 优解
    best_cost, best_cols = solutions[rank_idx]
    row_ind = np.arange(n)
    col_ind = np.array(best_cols)
    return row_ind, col_ind
# --------------------------------------------------------------------------------------

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

# 改：增加 rank_idx 参数，默认0=最优
def find_correspondence(raw_A:list[list[float]], raw_B:list[list[float]], rank_idx=0):
    A = np.array(raw_A)
    B = np.array(raw_B)
    N = len(raw_A)
    if len(A.shape) != 2 or A.shape[1] != 3:
        raise ValueError("A should be N * 3 matrix.")
    if len(B.shape) != 2 or B.shape[1] != 3:
        raise ValueError("B should be N * 3 matrix.")
    cost = build_cost_matrix(A, B)
    # 替换为支持k优解的版本
    row_ind, col_ind = k_best_linear_sum_assignment(cost, rank_idx=rank_idx)
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

    # 先构建代价矩阵（用于计算误差）
    A_np = np.array(A)
    B_np = np.array(B)
    cost_matrix = build_cost_matrix(A_np, B_np)
    
    # 测试：最优解(rank_idx=0)、次优解(rank_idx=1) + 误差衡量
    for rank in range(5):
        print(f"\n===== 第 {rank} 优匹配结果 =====")
        corr = find_correspondence(A, B, rank_idx=rank)
        
        # 计算当前匹配的总误差 & 平均误差
        total_error = 0.0
        for i, j in enumerate(corr):
            err = cost_matrix[i, j]
            total_error += err
            print(f"A[{i}] -> B[{j}] | 单点匹配误差: {err:.6f}")
        
        avg_error = total_error / len(corr)
        print(f"\n第 {rank} 优解 误差统计：")
        print(f"总匹配代价误差 = {total_error:.6f}")
        print(f"平均单点匹配误差 = {avg_error:.6f}")