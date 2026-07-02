# Scheme A2 — 带符号 ∂₂（无平均）

**模型 ID:** `scheme-a2-wip`  
**对比基线:** `scheme-a-v1` @ `4228e4d`（iou 78.2）

## 相对 A1 的改动

| 维度 | A1 (v1) | A2 |
|------|---------|-----|
| 边界聚合 | 无符号 `index_add` + **平均** | **带符号** `sign * h_coedge` + **线性求和** |
| 对偶聚合 ∂₂ᵀ | 无符号 + 平均 | 带符号 + 线性求和 |
| 层数 / 残差 / adj_face | 3 层 + LN + adj MLP | 同 A1 |
| 定向来源 | 隐含 +1 | `coedge_sign`（预处理）或 **首 wire 外环 +1 / 内环 -1** 启发式 |

## 实现

- `BoundaryOperatorTopoEncoderA2` in `models/boundary_topo_encoder.py`
- `build_face_coedge_pairs` → `(face_id, coedge_id, sign)`
- `process/solid_to_brt.py` 新样本写入 `coedge_sign`（`outer_wire` vs inner wire）
- 旧 `.bin` 无 `coedge_sign` 时自动用 wire 槽位启发式

## 实验矩阵（备忘）

| 版本 | 描述 |
|------|------|
| A1 | 无符号 + 平均 + 多层 → 78.2 |
| **A2** | **带符号 + 不平均 + 多层** |
| A3 | 无符号 + 不平均 + 多层 |
| A4 | 带符号 + 平均 + 多层 |

**关键对比:** A2 vs A3（符号是否有用）；A2 vs A4（不平均是否有用）。

## Results

`results/fusion360_seg/<MMDD>/schemea2/`
