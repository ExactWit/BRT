# BRT 结构创新方案

在现有 BRT 数据管线（三角 Bézier 面 + 边 Bézier + 拓扑 `.bin`）基础上，以**胞腔复形 / 边界算子**视角迭代神经网络结构。每个方案对应独立 git branch，从 `main`（base）分出。

## 当前 BRT 架构（baseline）

```
三角 Bézier patch → FaceEncoder (MLP + patch 内 Transformer + mean pool) → face_emb
边 Bézier + 端点   → EdgeEncoder + VertexEncoder                    → edge_emb
wire / adj 索引    → TopoEncoder (Wire RNN + 邻面 sum，单层)         → topo_emb
face 序列          → Transformer                                     → 逐面分类
```

主要缺口：拓扑层未显式使用边界算子 $\partial_2,\partial_1$；FaceEncoder 展平后 mean pool 丢失三角 Bézier 重心结构。

## Branch 规划

| Branch | 方案 | 状态 |
|--------|------|------|
| `main` | baseline + 本文档 | base |
| `scheme-a-boundary-mp` | A. 边界算子消息传递 | 进行中 |
| `scheme-b-hodge-block` | B. Hodge 分解块 | 待做 |
| `scheme-c-coedge-mp` | C. Coedge 双流形传播 | 待做 |
| `scheme-d-barycentric-face` | D. 重心感知 FaceEncoder | 待做 |
| `scheme-e-graph-transformer` | E. 面邻接图 Transformer | 待做 |

---

## 方案 A：边界算子消息传递（Boundary-Operator MP）

**Branch:** `scheme-a-boundary-mp`

**动机：** $\partial_2 f = \sum_{e \in \partial f} [f:e]\, e$，面特征应由**带符号的边界边**聚合，而非 Wire RNN 的无向 sum。

**结构：**

1. 维护 1/2-胞腔状态 $h_e, h_f$（由 EdgeEncoder / FaceEncoder 初始化）
2. **面更新（$\partial_2$）：** $h_f' = \mathrm{MLP}(h_f,\; \sum_{e \in \partial f} \sigma_{f,e}\, \phi(h_e))$，$\sigma_{f,e} \in \{-1,+1\}$ 由 wire 遍历方向决定（当前数据可推 +1）
3. **边更新（对偶 /  incident faces）：** $h_e' = \mathrm{MLP}(h_e,\; \mathrm{Agg}_{f \ni e} h_f)$
4. 堆叠 $L$ 层（默认 3），替代单层 `TopoEncoder`

**改动文件：** `models/boundary_topo_encoder.py`（新）、`models/brt.py`（切换 topo 层）

**验证：** `tests/test_scheme_a_forward.py` — forward + backward + 梯度非零

---

## 方案 B：Hodge 分解块（Hodge Cell Block）

**Branch:** `scheme-b-hodge-block`（待建）

**动机：** $\Omega^k = \mathrm{im}\, d_{k-1} \oplus \mathrm{im}\,\delta_k \oplus \mathcal{H}^k$

**结构：** 每层用固定稀疏 $\partial,\delta$ + 可学习线性变换近似 gradient / curl / harmonic 三路更新。

**依赖：** 方案 A 的胞腔消息传递框架

---

## 方案 C：Coedge 双流形传播（Manifold Coedge MP）

**Branch:** `scheme-c-coedge-mp`（待建）

**动机：** 流形 B-Rep 在 coedge 上有 $next/mate$ 环，比 wire RNN 更贴近 $\partial F$。

**数据扩展：** `solid_to_brt` 增加 `next, mate, face, edge` 索引（不动三角化）

---

## 方案 D：重心感知 FaceEncoder（Barycentric Bézier Encoder）

**Branch:** `scheme-d-barycentric-face`（待建）

**动机：** degree-6 三角 Bézier 的 28 个控制点有固定重心坐标，不应 flatten + mean pool。

**结构：** 控制点 barycentric PE + patch 内 Set/Point Transformer + attention pooling（尊重 `in_mask`）

---

## 方案 E：面邻接图 Transformer

**Branch:** `scheme-e-graph-transformer`（待建）

**动机：** 面序列 Transformer 顺序敏感；`adj_face_index` 已是 2-胞腔邻接图。

**结构：** 用 Graph Transformer 替代/减弱末端 face 序列 Transformer

---

## 方案 F：边界一致性正则（训练侧，可选）

$$\mathcal{L}_{\mathrm{bdry}} = \sum_f \Big\| \psi(h_f) - \sum_{e \in \partial f} \sigma_{f,e}\, \phi(h_e) \Big\|^2$$

可作为方案 A/C 的辅助 loss，推理结构不变。

---

## 推荐实施顺序

1. **A + D** — 不动数据，改动集中
2. **+ E** — 换全局聚合
3. **+ C** — 扩展 topo 预处理
4. **+ B + F** — 论文深度 / 正则

## 实验（训练阶段再做）

- 主任务：Fusion360 面分割（与 BRepNet 同 split）
- 消融：baseline → +A → +A+D → +A+D+E → +C
- 指标：mIoU、per-class IoU、参数量
