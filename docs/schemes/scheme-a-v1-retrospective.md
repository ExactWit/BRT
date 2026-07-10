# Scheme A v1 阶段性复盘

> **模型 ID:** `scheme-a-v1` · **架构提交:** `20e0044` · **复现锚点:** `4228e4d`（`scheme-a-boundary-mp`）  
> **对照基线:** `baseline` @ `ad21473`（`main` / `TopoEncoder`）  
> **状态:** 2026-07 阶段总结；A1 已归档，A2 在 `bb14b54` 继续迭代。

---

## 1. Problem（问题）

BRT 在 B-Rep 面分割任务上依赖一条固定流水线：三角 Bézier **面编码**、边/顶点编码、**拓扑编码**、再经面序列 Transformer 做逐面分类。其中拓扑层 `TopoEncoder`（`models/brt.py`）是连接几何嵌入与全局推理的瓶颈：

1. **Wire RNN 聚合边界**：沿每条 wire 用 `nn.RNN` 编码边序列，再对面内多条 wire 做 **无向求和**（`torch.sum(wires_per_face, dim=1)`），不区分边的遍历方向，也不显式对应「面的边界」这一拓扑对象。
2. **单层、无对偶更新**：边特征经 RNN 后不再更新；面只接收 wire 汇总与邻接面 MLP 聚合，缺少边—面双向消息传递。
3. **与胞腔复形视角脱节**：B-Rep 天然是 0/1/2-胞腔（顶点 / 边 / 面）及其边界关系；baseline 用启发式索引（`wire_index`、`edge_index`、`adj_face_index`）拼特征，**未把「面的边界 = 边链」写成可解释的算子**。

在 **Fusion 360 Gallery Segmentation**（8 类 per-face seg）上，该设计足以训练，但 test macro-IoU 长期卡在 **~74.5%**（`baseline` @ `ad21473`，见 `results/fusion360_seg/0629/193848`）。在 **TMCAD / mechcad** 上，同一拓扑层在「零件类广播到每个面」的 seg 设定下，test accuracy 约 **74.8%**（`results/mechcad_seg/0705/baseline`），同样存在提升空间。

---

## 2. Motivation（动机）

我们从**胞腔复形上的边界算子**出发重新设计拓扑层，核心动机有三点：

| 动机 | 说明 |
|------|------|
| **几何—拓扑一致** | 面的语义应由其**定向边界**上的边（coedge）特征决定，而非 wire 上无序 RNN 隐状态。 |
| **可堆叠的消息传递** | 单层 RNN 表达能力有限；多层边↔面迭代 + 残差更接近 GNN / MPNN 在图上的成功范式。 |
| **最小侵入数据管线** | 不改动三角化与 `.bin` 拓扑格式（仍用 `wire_index`、`edge_index`、`adj_face_index`），仅替换 `BRT.topo_layer`。 |

设计目标形式化：用边界算子 $\partial_2$ 将边（1-胞腔）特征聚合到面（2-胞腔）：

$$
\partial_2 f = \sum_{e \in \partial f} [f:e]\, h_e, \quad [f:e] \in \{-1,+1\}
$$

并在实现上允许对偶方向 $\partial_2^\top$（边接收 incident 面特征）。**动机是拓扑正确的消息传递**；是否与严格带符号 $\partial_2$ 一致，见 Solution 与 Limitation。

---

## 3. Solution（方案与实现）

### 3.1 总体结构（相对 baseline 不变部分）

```
三角 Bézier → FaceEncoder → face_emb ─┐
边 Bézier + 顶点 → EdgeEncoder → edge_emb ─┤
                                          ├→ BoundaryOperatorTopoEncoder (×3 层) → topo_emb
wire / adj 索引 ──────────────────────────┘
topo_emb → 按 solid 分 batch → Transformer → 逐面 logits
```

切换点（`models/brt.py` @ `4228e4d`）：

```python
from .boundary_topo_encoder import BoundaryOperatorTopoEncoder

self.topo_layer = BoundaryOperatorTopoEncoder(
    edge_dim=dmodel, face_dim=dmodel, hidden_dim=2 * dmodel, dropout=dropout
)
```

### 3.2 `BoundaryOperatorTopoEncoder` 做了什么（`models/boundary_topo_encoder.py`）

**（1）从 wire 建 (face, edge) 对**

`build_face_edge_pairs` 遍历每个面的 wire 列表，再沿 wire 取 `edge_index`，得到面边界上的 `(face_id, edge_id)`。当前数据**未使用 coedge 符号**，等价于 incidence $+1$。

**（2）面更新 — 类 $\partial_2$ 聚合**

```python
def scatter_boundary_to_faces(h_edge, face_ids, edge_ids, n_faces):
    agg.index_add_(0, face_ids, h_edge[edge_ids])
    return agg / counts.clamp(min=1.0)   # 按边界边数平均
```

面状态更新（每层）：

$$
h_f \leftarrow \mathrm{LN}\bigl(h_f + \mathrm{MLP}([h_f,\; \mathrm{mean}_{e\in\partial f} h_e,\; \mathrm{AdjMLP}(h_f, \mathcal{N}(f))])\bigr)
$$

其中 `AdjMLP` 即 `_aggregate_adj_faces`：对 `adj_face_index` 邻居面做 MLP 后求和（与 baseline 类似，但并入 MP 循环）。

**（3）边更新 — 类 $\partial_2^\top$**

```python
def scatter_incident_faces_to_edges(h_face, face_ids, edge_ids, n_edges):
    agg.index_add_(0, edge_ids, h_face[face_ids])
    return agg / counts.clamp(min=1.0)
```

边状态：$h_e \leftarrow \mathrm{LN}(h_e + \mathrm{MLP}([h_e, \mathrm{mean}_{f\ni e} h_f]))$。

**（4）深度与输出**

- 默认 **3 层**上述边—面交替更新（`num_layers=3`）。
- 最后再聚合一次邻接面，经 `output_layer` 映射回 `edge_dim`，供后续 Transformer 使用。

**（5）与 baseline `TopoEncoder` 对照**

| 维度 | Baseline `TopoEncoder` | Scheme A v1 |
|------|------------------------|-------------|
| 边界编码 | WireNet（RNN，取最后隐状态） | 边界边特征直接 `index_add` 到面 |
| 层数 | 1 | 3（残差 + LayerNorm） |
| 边状态 | 训练后固定 | 每层更新 |
| 邻接面 | 单次 `adj_face_layer` + sum | 每层 + 最终输出前各一次 |
| 参数量 | 较小 | 明显增加（3× face/edge MLP） |

### 3.3 训练与评测设定

- **Fusion 360**：`processed/brt`，8 类，`segmentation.py`，监控 `val_iou`，与 BRepNet 划分对齐（`datasplit.json` / `dataset.json`）。
- **TMCAD / mechcad**：10 类零件 ID 广播为 per-face 标签（`datasets/brt_dataset.py` `load_one_sample`）；seg 任务下 **macro accuracy 为主要可读指标**（每面同类，IoU 与 acc 高度相关）。
- 公共 infra：`scripts/branch.sh` + `scripts/model_registry.tsv`；模型锚点见仓库根 `models.tsv`（exp_launcher）与 `scripts/model_registry.tsv`（branch.sh）。

---

## 4. Contribution（贡献与结果）

### 4.1 方法贡献

1. **首个在 BRT 管线内可复现的「胞腔消息传递」拓扑层**，以 `BoundaryOperatorTopoEncoder` 替换 Wire RNN，并在 **不改编码器与 Transformer** 的前提下取得显著提升。
2. **可审计的实现路径**：`build_face_edge_pairs` + scatter 聚合 + 多层残差，配套 `tests/test_scheme_a_forward.py`（forward/backward smoke）。
3. **工程化复现**：`scheme-a-v1` 写入模型注册表（`4228e4d`），`experiment_metadata.json` / `test_metadata.json` 记录 commit 与指标；归档说明见 [`scheme-a-v1-archive.md`](./scheme-a-v1-archive.md)。

### 4.2 定量结果（best checkpoint，test split）

#### Fusion 360 Gallery Segmentation

| 指标 | Baseline (`ad21473`) | Scheme A v1 (`4228e4d`) | Δ |
|------|----------------------|-------------------------|---|
| test macro-IoU | 74.6% | **78.2%** | **+3.6 pp** |
| test accuracy | 93.7% | **94.6%** | +0.9 pp |

来源：`results/fusion360_seg/0629/193848`（baseline）vs `results/scheme-a-boundary-mp_360/0701/153039`（scheme-a，`git` @ `6b45ace`，与注册表 `4228e4d` 同系列提交）。

#### TMCAD / mechcad（seg，标签广播）

| 指标 | Baseline (`ad21473`) | Scheme A 分支 (`1cf72df` run) | Δ |
|------|----------------------|-------------------------------|---|
| test accuracy | 74.78% | **75.21%** | +0.43 pp |
| test macro-IoU | 60.56% | **61.14%** | +0.58 pp |

来源：`results/mechcad_seg/0705/baseline` vs `results/mechcad_seg/0706/schemea`。  
**说明：** 该 mechcad run 登记为 `scheme-a-v1` 但 checkout 在 `1cf72df`（含数据集 label 修复；分支 HEAD 已合并 A2 拓扑代码）。严格 **A1-only** 的 mechcad 对照建议在 `4228e4d` + label fix 上补跑；Fusion360 上的 A1 结论更干净。

### 4.3 归因（与归档文档一致）

| 因素 | 对 Fusion360 IoU 的预期贡献 |
|------|---------------------------|
| 3 层迭代 MP（vs 单层 RNN） | 主要（~+2.0–2.5 pp） |
| 残差 + LayerNorm | ~+0.5–1.0 pp |
| 边状态双向更新 | ~+0.5–1.0 pp |
| adj_face MLP（保留） | ~+0.3–0.5 pp |
| 「无符号 + 平均」的 ∂₂ 形式 | ≈ 0（非严格边界算子） |

实测 **+3.6 pp IoU** 与上述分解一致（详见 [`scheme-a-v1-archive.md`](./scheme-a-v1-archive.md) 因素表）。

---

## 5. Limitation（局限与后续）

### 5.1 理论与实现 gap

| 设计意图 | A1 实际代码 |
|----------|-------------|
| 带符号 $\partial_2$ | **无符号** `index_add`，未用 $[f:e]$ |
| 线性边界算子 | 边界聚合后 **除以边数**（`agg / counts`） |
| 仅边界驱动面更新 | 仍强依赖 **adj_face** 邻居聚合 |
| 单层算子 | **3 层非线性 MLP** + 残差 |

因此 A1 应理解为 **「受边界算子启发的多层胞腔 MP」**，而非离散 Hodge 理论中的严格 $\partial_2$。性能增益更可能来自 **深度 MP + 稳定训练**，而非符号拓扑本身。

### 5.2 数据与任务局限

- **mechcad seg** 是零件分类的退化设定（每面同标），指标增益小于 Fusion360，且缺少纯 `4228e4d` 的独立 run。
- **FaceEncoder** 仍为 flatten + patch 内 Transformer + mean pool，未利用三角 Bézier 重心结构（方案 D 待做）。
- 面序列 **Transformer 顺序敏感** 问题未解决（方案 E 待做）。
- 未系统报告参数量、训练耗时、收敛 epoch 对比。

### 5.3 已启动的后续（Scheme A2）

在 A1 框架上，**A2**（`BoundaryOperatorTopoEncoderA2` @ `bb14b54`）仅改边界聚合：

- 带符号 coedge incidence（`coedge_sign`，外环 +1 / 内环 −1）；
- **线性求和、不平均**（`scatter_boundary_to_faces_signed`）。

Fusion360 上 A2 与 A1 接近（~77.6–77.8% IoU），说明 A1 的主要收益**不来自**严格 $\partial_2$ 符号，与 5.1 一致。详见 [`scheme-a2.md`](./scheme-a2.md)。

### 5.4 复现命令

```bash
# 在 main 上启动 branch.sh（读 scripts/model_registry.tsv）
bash scripts/branch.sh
# → model: scheme-a-v1 @ 4228e4d
# → 360 / mechcad → train | test | viz

# exp_launcher（仓库根 models.tsv + run.sh）
./run.sh capabilities
```

---

## 6. 小结

Scheme A v1 用 **多层边界启发式消息传递** 替换了 BRT baseline 的 Wire RNN 拓扑层，在 **Fusion 360 面分割** 上将 test IoU 从 **~74.6% 提升到 ~78.2%**，在 **TMCAD seg** 上取得小幅但一致的 accuracy 提升。其核心价值在于：**在现有 BRT 数据与训练栈上验证了「显式建模面—边拓扑关系 + 深层 MP」的有效性**；同时诚实地承认当前实现与理论 $\partial_2$ 仍有差距，这一差距正由 A2 及后续方案 B/C 继续闭合。

**相关文档：** [`brt_innovation.md`](../brt_innovation.md) · [`scheme-a-v1-archive.md`](./scheme-a-v1-archive.md) · [`scheme-a2.md`](./scheme-a2.md)
