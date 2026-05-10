---
layout: post
title: GPU加速ADMM求解大规模二次规划
date: 2026-05-10
description: 基于OSQP的CUDA实现，利用预条件共轭梯度法替代直接求解器，实现高达两个数量级的性能提升
categories:
  - 优化算法
  - GPU计算
---

## 1. 引言

在许多工程领域（包括控制、信号处理、统计学、金融和机器学习），凸优化已成为标准工具。在某些应用中，优化问题的维度可能非常大。对于这类问题，经典的优化算法（如内点法）可能无法提供解决方案。

近十年来，算子分裂方法（如近似梯度法（proximal gradient method）和交替方向乘子法（ADMM））在众多应用领域中获得了越来越多的关注。这些方法具有良好的可扩展性，能够高效地利用问题数据中的稀疏性，而且通常易于并行化。

图形处理单元（GPU）是硬件加速器，以相对较低的价格提供无与伦比的并行计算能力。它们提供的内存带宽远大于传统CPU系统，这在处理大量数据的应用中是尤其有益的。

本文探讨利用GPU的大规模并行化能力来加速大规模二次规划（QP）的求解。我们在OSQP求解器的基础上构建我们的求解器，实验表明我们的开源CUDA实现比CPU实现快达两个数量级。

> **论文来源**：Schubiger M, Banjac G, Lygeros J. GPU acceleration of ADMM for large-scale quadratic programming[J]. Journal of Parallel and Distributed Computing, 2020, 144: 55-67.

## 2. 问题描述

考虑如下二次规划（QP）问题：

$$
\begin{align}
\text{minimize} \quad & \frac{1}{2}x^TPx + q^Tx \\
\text{subject to} \quad & l \leq Ax \leq u
\end{align}
$$

其中：
- $x \in \mathbb{R}^n$ 是优化变量
- $P \in \mathbb{S}_+^n$ 是半正定矩阵
- $q \in \mathbb{R}^n$ 是线性项系数向量
- $A \in \mathbb{R}^{m \times n}$ 是约束矩阵
- $l, u \in \mathbb{R}^m \cup \{-\infty, +\infty\}$ 分别是下界和上界向量

通过引入变量 $z \in \mathbb{R}^m$，我们可以将问题重写为等价形式：

$$
\begin{align}
\text{minimize} \quad & \frac{1}{2}x^TPx + q^Tx \\
\text{subject to} \quad & Ax = z, \quad l \leq z \leq u
\end{align}
$$

## 3. OSQP求解器

OSQP是一个基于ADMM的开源QP数值求解器，已被证明具有竞争力甚至比商业QP求解器更快。

### 3.1 ADMM算法

OSQP的ADMM迭代算法如下：

**Algorithm 1: OSQP算法**

```
given initial values x⁰, z⁰, y⁰ and parameters σ > 0, R ∈ Sⁿ⁺⁺, α ∈ ]0,2[
Set k = 0
repeat
    (x̃ᵏ⁺¹, z̃ᵏ⁺¹) ← argmin_{x̃,z̃:Ax̃=z̃} ½x̃ᵀPx̃ + qᵀx̃ + (σ/2)∥x̃ - xᵏ∥²₂ + ½∥z̃ - zᵏ + R⁻¹yᵏ∥²₂
    xᵏ⁺¹ ← αx̃ᵏ⁺¹ + (1-α)xᵏ
    zᵏ⁺¹ ← Π[l,u](αz̃ᵏ⁺¹ + (1-α)zᵏ - R⁻¹yᵏ)
    yᵏ⁺¹ ← yᵏ + R(zᵏ⁺¹ - αz̃ᵏ⁺¹ - (1-α)zᵏ)
    k ← k + 1
until termination criterion is satisfied
```

其中标量 $\alpha \in (0,2)$ 称为松弛参数，$\sigma > 0$ 和 $R \in \mathbb{S}_+^n$ 是惩罚参数。OSQP使用对角正定矩阵 $R$，这使得 $R^{-1}$ 易于计算。

投影到盒约束 $[l, u]$ 上的欧几里得投影具有简单的闭式解：

$$
\Pi_{[l,u]}(z)_i = \min(\max(z_i, l_i), u_i), \quad i = 1, \ldots, m
$$

### 3.2 最优性条件

问题(2)的最优性条件为：

$$
\begin{align}
Ax - z &= 0 \tag{3a} \\
Px + q + A^T y &= 0 \tag{3b} \\
l \leq z \leq u \tag{3c} \\
y_{\mathcal{I}_-(z-l)} = (z-l)_{\mathcal{I}_-(z-l)}^T &= 0, \quad y_{\mathcal{I}_+(z-u)} = (z-u)_{\mathcal{I}_+(z-u)}^T = 0 \tag{3d}
\end{align}
$$

### 3.3 求解KKT系统

Algorithm 1的第4步需要求解等式约束QP，等价于求解以下线性系统：

$$
\begin{bmatrix} P + \sigma I & A^T \\ A & -R^{-1} \end{bmatrix} \begin{bmatrix} \nu^{k+1} \\ x^{k+1} \end{bmatrix} = \begin{bmatrix} z^k - \sigma x^k - q \\ -R^{-1}y^k \end{bmatrix} \tag{7}
$$

我们称(7)中的矩阵为KKT矩阵。OSQP使用直接方法，通过首先计算KKT矩阵的因式分解来计算精确解。由于KKT矩阵不依赖于迭代计数器 $k$，OSQP在算法开始时执行因式分解，并在后续迭代中重用这些因子。

## 4. GPU加速方案：预条件共轭梯度法

### 4.1 共轭梯度法（CG）

共轭梯度法是一种迭代方法，用于求解线性系统：

$$
Kx = b \tag{10}
$$

其中 $K \in \mathbb{S}_+^n$ 是对称正定矩阵。该方法最多在 $n$ 次迭代内计算出线性系统的解。

求解(10)等价于求解以下无约束优化问题：

$$
\text{minimize} \quad f(x) := \frac{1}{2}x^T Kx - b^T x
$$

### 4.1.1 共轭方向

一组非零向量 $\{p_0, \ldots, p_{n-1}\}$ 称为关于 $K$ 共轭的，如果：

$$
p_i^T K p_j = 0, \quad \forall i \neq j
$$

沿共轭方向 $p_k$ 进行逐次最小化：

$$
\alpha_k = \arg\min_{\alpha} f(x_k + \alpha p_k) \tag{11a}
$$

更新解：

$$
x_{k+1} = x_k + \alpha_k p_k \tag{11b}
$$

### 4.1.2 预条件共轭梯度法（PCG）

**Algorithm 3: PCG方法**

```
Require: initial value x⁰, preconditioner M
1: r⁰ ← Kx⁰ - b, y⁰ ← M⁻¹r⁰, p⁰ ← -y⁰
2: Set k = 0
3: while ∥rᵏ∥ > ε∥b∥ do
4:     αᵏ ← -(rᵏ)ᵀyᵏ / (pᵏ)ᵀKpᵏ
5:     xᵏ⁺¹ ← xᵏ + αᵏpᵏ
6:     rᵏ⁺¹ ← rᵏ + αᵏKpᵏ
7:     yᵏ⁺¹ ← M⁻¹rᵏ⁺¹
8:     βᵏ⁺¹ ← (rᵏ⁺¹)ᵀyᵏ⁺¹ / (rᵏ)ᵀyᵏ
9:     pᵏ⁺¹ ← -yᵏ⁺¹ + βᵏ⁺¹pᵏ
10:    k ← k + 1
11: end while
```

### 4.2 缩减KKT系统

从(7)中消去 $\nu^{k+1}$，得到缩减KKT系统：

$$
(P + \sigma I + \rho\bar{I}A^TA)\tilde{x}^{k+1} = \sigma x^k - q + A^T(\rho\bar{z}^k - y^k) \tag{14}
$$

注意上述系数矩阵不必显式形成。矩阵-向量积：

$$
r \leftarrow (P + \sigma I + \rho\bar{I}A^TA)x
$$

可以通过以下方式计算：

$$
\begin{aligned}
z &\leftarrow \rho\bar{A}x \\
r &\leftarrow Px + \sigma x + A^T z
\end{aligned}
$$

### 4.3 预条件子

我们使用Jacobi预条件子，求解 $My = r$ 相当于简单的对角矩阵-向量乘积。Jacobi预条件子对(14)的对角线可计算为：

$$
\text{diag}(M) = \text{diag}(P) + \sigma \mathbf{1} + \rho\bar{I} \cdot \text{diag}(A^TA)
$$

### 4.4 参数更新

cuOSQP每10次迭代更新 $\bar{\rho}$，这有助于减少ADMM迭代总数。通过使用预条件子，参数更新计算变得非常便宜，因为只需更新预条件子 $M$。

### 4.5 终止准则与热启动

PCG方法的解无需精确完成Algorithm 1的收敛。我们使用基于ADMM残差的自适应终止准则：

$$
\varepsilon \leftarrow \max\left(\lambda \sqrt{\|r_{\text{prim}}^k\|_\infty \|r_{\text{dual}}^k\|_\infty}, \varepsilon_{\min}\right)
$$

其中 $r_{\text{prim}}^k$ 和 $r_{\text{dual}}^k$ 是缩放后的原始和对偶残差。参数 $\lambda \in (0,1)$ 确保 $\varepsilon$ 始终低于原始和对偶残差的几何均值。

通过热启动（warm-starting）技术，将 $x^0$ 初始化为上一次ADMM迭代计算得到的解 $\tilde{x}^k$，可以进一步加速PCG收敛。

## 5. CUDA实现要点

### 5.1 稀疏矩阵存储格式

#### COO格式（Coordinate）

COO格式是最简单的稀疏矩阵格式之一。它存储三个数组：Value、RowIndex和ColumnIndex。适用于矩阵操作（如转置、拼接）作为中间格式。

例如，矩阵：

$$
A = \begin{bmatrix} 1 & 0 & 0 & 0 & 4 \\ 0 & 5 & 1 & 0 & 0 \\ 0 & 2 & 0 & 0 & 1 \\ 7 & 0 & 1 & 0 & 0 \end{bmatrix}
$$

的COO表示为：

```
Value = [1, 4, 5, 1, 2, 1, 7, 1]
RowIndex = [0, 0, 1, 1, 2, 2, 3, 3]
ColumnIndex = [0, 4, 1, 2, 1, 4, 0, 2]
```

#### CSR格式（Compressed Sparse Row）

CSR格式与COO格式的区别仅在于RowIndex数组被压缩。压缩过程包括：
1. 从RowIndex确定每行非零元素个数
2. 计算累积和并在开头插入0
3. 结果数组记为RowPointer

CSR格式在GPU上具有更优的稀疏矩阵-向量乘法（SpMV）性能。

#### CSC格式（Compressed Sparse Column）

CSC格式与CSR类似，但按列压缩存储。cuOSQP同时存储 $A$ 和 $A^T$ 的CSR格式，避免了转置运算的开销。

### 5.2 核函数设计

#### 向量运算（axpy）

```cuda
__global__ void axpy_gpu(float* y, float* x, float a, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        y[idx] += a * x[idx];
    }
}
```

#### 分段归约（Segmented Reduction）

归约操作将向量 $x \in \mathbb{R}^n$ 和结合二元运算符 $\oplus$ 作为输入，返回标量。在CSR格式中，RowPointer数组定义了segments，可高效并行计算行范数：

$$
\|A\|_{2,i}^2 = \|A_{\cdot i}\|_2^2
$$

### 5.3 cuBLAS和cuSPARSE库

cuOSQP使用NVIDIA提供的CUDA库：
- **Thrust**: CUDA C++模板库，提供高级并行原语
- **cuBLAS**: BLAS的CUDA实现，用于点积、axpy、范数等level-1 BLAS函数
- **cuSPARSE**: 处理稀疏矩阵的线性代数子程序，用于SpMV等操作

### 5.4 OSQP与cuOSQP的主要区别

| 特性 | OSQP (CPU) | cuOSQP (GPU) |
|------|------------|--------------|
| Step 4求解 | 线性系统(7) | 线性系统(14) |
| 求解方法 | LDLT分解 | PCG方法 |
| 参数矩阵R | $R = \text{diag}(\rho_1, \ldots, \rho_m)$ | $R = \bar{\rho}I$ |
| $\bar{\rho}$更新频率 | 很少 | 每10次迭代 |
| 矩阵存储格式 | CSC格式 | CSR格式 |
| | 上三角P | 完整P |
| | 仅存储A | 同时存储A和Aᵀ |
| 终止条件检查 | 每25次迭代 | 每5次迭代 |

## 6. 数值结果

### 6.1 测试环境

- CPU: Intel i9-9900K @ 3.6 GHz (8核)
- RAM: 64 GB DDR4 3200MHz
- GPU: NVIDIA GeForce RTX 2080 Ti (11 GB VRAM)

### 6.2 测试问题类别

实验使用了7类基准问题：

1. **控制问题（Control）**：线性时不变系统的有限时域最优控制问题
2. **等式约束（Equality）**：仅含等式约束的QP
3. **Huber拟合**：鲁棒最小二乘回归问题
4. **Lasso**：稀疏线性回归问题
5. **投资组合优化（Portfolio）**：金融领域的风险调整收益最大化问题
6. **随机QP（Random）**：随机生成的QP
7. **支持向量机（SVM）**：分类问题的QP形式

### 6.3 性能对比

实验结果表明：
- 对于问题规模小于 $10^5$ 非零元素的情况，OSQP比cuOSQP更快
- 对于大规模问题实例，cuOSQP显著更快
- 最大加速比达到270倍（Equality类问题）
- SVM类问题的加速比为107倍

```
表：OSQP与cuOSQP性能对比（典型结果）

问题类别    | 最大加速比 | OSQP最大运行时间 | cuOSQP运行时间
------------|------------|------------------|----------------
Equality    | 270倍      | 6.5分钟          | 1.4秒
SVM         | 107倍      | 5.6分钟          | 3.2秒
Lasso       | 50倍       | 4.2分钟          | 5秒
Portfolio   | 30倍       | 3分钟            | 6秒
Control     | 15倍       | 2分钟            | 8秒
```

### 6.4 浮点精度影响

cuOSQP支持单精度和双精度浮点表示。使用双精度相比单精度的计算时间惩罚小于2倍。这是因为SpMV等操作是内存带宽限制的，而非计算限制。

### 6.5 迭代次数分析

更新 $\bar{\rho}$ 每10次迭代有助于减少Equality、Lasso和SVM类的ADMM迭代总数。对于Control、Portfolio和Random类，频繁更新 $\bar{\rho}$ 并未显示明显优势。

## 7. 结论

本文探索了利用GPU的大规模并行化能力来加速大规模QP的求解。我们的实现cuOSQP建立在OSQP基础上，这是一个基于ADMM的最先进QP求解器。显著加速是通过使用PCG方法求解ADMM中的线性系统以及并行化所有向量和矩阵运算实现的。

**主要贡献**：
1. PCG方法替代直接线性系统求解器，避免了昂贵的矩阵分解
2. GPU并行化稀疏矩阵运算和向量归约
3. 自适应PCG终止准则，根据ADMM收敛状态动态调整精度
4. 实现高达两个数量级的性能提升

**局限性**：
- 小规模问题无法充分利用GPU，CPU实现通常更快
- 问题规模受限于可用GPU内存
- 需要优化小问题场景的kernel launch开销

**未来方向**：
- 统一内存（Unified Memory）方法合并系统内存和GPU内存
- 多GPU架构分布式求解超大规模问题

## 参考文献

[1] Schubiger M, Banjac G, Lygeros J. GPU acceleration of ADMM for large-scale quadratic programming[J]. Journal of Parallel and Distributed Computing, 2020, 144: 55-67.

[2] Stellato B, Banjac G, Goulart P, et al. OSQP: an operator splitting solver for quadratic programs[J]. Mathematical Programming Computation, 2020, 12(4): 637-667.
