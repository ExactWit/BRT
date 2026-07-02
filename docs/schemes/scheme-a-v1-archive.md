# Scheme A v1 归档

**归档 ID:** `scheme-a-v1`  
**Git ref:** `4228e4d`（`scheme-a-boundary-mp` @ 2026-06 实验冻结点）  
**模型实现提交:** `20e0044`（`feat(scheme-a): replace TopoEncoder with boundary-operator message passing`）  
**状态:** `archived` — 性能良好，但与理论 $\partial_2$ 构象有显著偏差；后续迭代前以此点为复现基线。

## 实验结果（Fusion360 seg, best ckpt）

| 指标 | Baseline (`main` / TopoEncoder) | Scheme A v1 |
|------|----------------------------------|-------------|
| test IoU | 74.5 | **78.2** (+3.7) |
| test acc | 93.6 | **94.5** (+0.9) |

> 以上为当前最佳 run 的汇总；具体 run 目录见 `results/fusion360_seg/<date>/schemea/`，metadata 中 `model.id == scheme-a-v1` 或 `git.branch == scheme-a-boundary-mp`。

## 理论构象 vs 当前实现

方案 A 的**设计动机**是显式使用边界算子 $\partial_2$：

$$
\partial_2 f = \sum_{e \in \partial f} [f:e]\, h_e,\quad [f:e] \in \{-1,+1\}
$$

| 维度 | 理论 $\partial_2$ | 当前实现 (`BoundaryOperatorTopoEncoder`) |
|------|-------------------|------------------------------------------|
| 边界 incidence 符号 | 带符号 $\sum [f:e]\cdot h_e$ | **无符号**：`index_add` 纯累加 |
| 聚合归一化 | 线性算子，不求平均 | **求平均**：`agg / counts` |
| 面更新输入 | 仅边界边特征 | 边界边 + **邻居面** `adj_agg`（`adj_face` MLP） |
| 层数 | 单层线性边界算子 | **3 层**迭代 MP + 残差 + LayerNorm |
| 边状态 | 可选对偶 $\partial_2^\top$ | **有** edge 更新（incident faces） |

**结论：** 当前模块更接近「多层胞腔消息传递 + 邻接面聚合」，而非严格的带符号 $\partial_2$。名称保留 `BoundaryOperatorTopoEncoder` 以记录设计意图；性能提升可能主要来自下表因素，而非边界算子结构本身。

## 因素归因（预估 vs 实测）

| 因素 | 预估 IoU 提升 | 理由 |
|------|---------------|------|
| 多层迭代（3 层 vs 1 层） | +2.0 ~ +2.5 | 主要收益来源 |
| 残差 + LayerNorm | +0.5 ~ +1.0 | 训练稳定性 |
| Edge 状态更新 | +0.5 ~ +1.0 | 信息双向流动 |
| adj_face MLP 聚合 | +0.3 ~ +0.5 | 比 pooling 灵活 |
| 「边界算子」结构（无符号+平均） | ≈ 0 | 非真正 $\partial_2$ |
| **合计** | +3.7 ~ +5.0 | **实测 +3.7**（74.5→78.2） |

## 关键代码文件（`4228e4d`）

| 文件 | 说明 |
|------|------|
| `models/boundary_topo_encoder.py` | `BoundaryOperatorTopoEncoder`、scatter 聚合 |
| `models/brt.py` | topo 层切换为 Scheme A |
| `tests/test_scheme_a_forward.py` | forward/backward smoke test |

## 复现

```bash
# 交互式（推荐）
bash scripts/branch.sh
# → 选择 model: scheme-a-v1 @ 4228e4d
# → 360 → train / test / viz

# 或手动 checkout
git checkout scheme-a-boundary-mp
git checkout 4228e4d
```

`branch.sh` 会 checkout 到 `4228e4d`，并在 `experiment_metadata.json` 写入 `model.id` / `model.commit`。

## 后续迭代方向（备忘，未实施）

1. 真·带符号 $\partial_2$（wire 遍历方向 / coedge orientation）
2. 去掉 `counts` 平均或改为可学习缩放
3. 消融：仅边界边 vs +adj_face、层数、edge 更新
4. 与 Scheme B（Hodge 块）在统一 MP 框架上组合
