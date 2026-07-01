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

## 分支工作流（公共 infra vs 方案代码）

| 分支 | 职责 |
|------|------|
| `main` | 公共 infra：`scripts/branch.sh`、`segmentation.py`、`utils/`、数据集脚本、`docs/` 等 |
| `scheme-*` | **仅**方案相关：`models/*_encoder.py`、切换 `models/brt.py` 中的 topo 层、对应 `tests/` |

**约定：**

1. 新方案分支从 **main 最新公共 infra** 分出，不要从旧 scheme 分支再分。
2. scheme 分支需要公共更新时，在分支上执行 `git merge main`（不要 cherry-pick 多文件）。
3. 公共 infra 锚点（含 branch.sh、experiment metadata、viz）：`82f4ba0`（`feat: branch control`）。

```bash
# 新建方案 C
git checkout main
git checkout -b scheme-c-coedge-mp
# 只改 models/ 与 tests/，然后 train
bash scripts/branch.sh
```

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

---

## 交互式实验脚本 `scripts/branch.sh`

在 **main** 分支提供的统一入口，自动切换 git 分支、选择数据集并执行 train / test / viz。

```bash
cd ~/workspace/repo/BRT
bash scripts/branch.sh
```

流程：

1. 列出本地 git 分支并 `checkout` 到所选分支
2. 选择数据集：`360` | `mechcad`
3. 选择操作：`train` | `test` | `viz`

### 实验元数据

每次 **train** 都会在 run 目录写入：

`results/<experiment_name>/<log_name>/<log_version>/experiment_metadata.json`

（通过 `branch.sh`、`train_360.sh` 或直接 `python segmentation.py train` 均可；未传 `--git_branch` 时会自动从当前仓库读取。）

**train** 元数据示例（`schema_version: 1`）：

```json
{
  "schema_version": 1,
  "run": {
    "experiment_name": "scheme-a-boundary-mp_360",
    "log_name": "0701",
    "log_version": "120000",
    "run_dir": "/abs/path/to/results/...",
    "created_at": "2026-07-01T12:00:00"
  },
  "git": {
    "branch": "scheme-a-boundary-mp",
    "commit": "908a912",
    "commit_full": "...",
    "dirty": false,
    "remote": "..."
  },
  "dataset": {
    "id": "360",
    "processed_dir": "/data/hdd/datasets/s2.0.0/processed/brt",
    "num_classes": 8,
    "num_control_pts": 28
  },
  "datasplit": {
    "datasplit_json": ".../datasplit.json",
    "sha256": "...",
    "counts": {"train": 19675, "val": 8426, "test": 4978},
    "split_source_json": ".../processed/dataset.json",
    "sidecar": { "...": "来自 datasplit_meta.json" }
  },
  "train": {
    "batch_size": 16,
    "max_epochs": 1000,
    "gpu": 0,
    "resume_from": null
  },
  "note": "用户备注（branch.sh train 时会提示输入）"
}
```

预处理划分时会在 `processed_dir` 旁写入 `datasplit_meta.json`（划分来源、各 split 样本数、跳过数等），train 时会一并记入 `datasplit.sidecar`。

**test** 结束后在同目录写入 `test_metadata.json`（含 metrics、datasplit 摘要、关联的 `experiment_metadata_path`）。

`test` / `viz` 会按当前 **branch + dataset** 过滤 `results/` 下的实验供选择；列表中会显示备注摘要（`note=...`）。

### 数据集默认路径

| dataset | processed | STEP 根目录 | num_classes |
|---------|-----------|-------------|-------------|
| `360` | `/data/hdd/datasets/s2.0.0/processed/brt` | `.../breps/step` | 8 |
| `mechcad` | `/data/hdd/datasets/mechcad/processed` | `.../mechcad` | 25 |

### viz 输出

`viz` 从 test 划分选样本，调用 `scripts/viz_segmentation.py`：

- **ply**：`{stem}_pred.ply`、`{stem}_gt.ply`（面片着色网格）
- **stp**：`{stem}_pred.stp`、`{stem}_gt.stp`（XCAF 面色 STEP）

默认输出到 `<run_dir>/viz/<stem>/`，可用环境变量 `VIZ_OUTPUT_DIR` 覆盖。

### 环境变量（可选）

```bash
BATCH_SIZE=8 GPU=0 MAX_EPOCHS=100 EXPERIMENT_NAME=my_exp bash scripts/branch.sh
```
