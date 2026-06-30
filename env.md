# BRT Conda 环境配置说明

本文档记录 `process/` 数据预处理脚本（拓扑提取、三角面片提取、数据集划分）所需的 conda 环境配置方法，以及直接 `conda env create -f environment.yml` 可能失败的原因与修复方式。

## 背景：为什么只跑 environment.yml 不够

原始 `environment.yml` 缺少预处理脚本实际依赖的包，且未处理一个常见的 Python 路径冲突：

| 问题 | 现象 | 原因 |
|------|------|------|
| 缺少 `occwl` | `ModuleNotFoundError: No module named 'occwl'` | `process/` 下脚本大量依赖 `occwl`（OpenCascade 的 Python 封装），但原 yml 未声明 |
| NumPy 版本冲突 | `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6` | 用户目录 `~/.local/lib/python3.10/site-packages` 中安装了 numpy 2.x，优先级高于 conda 环境内的 1.23.5，导致 PyTorch 加载失败 |
| 缺少 `tqdm` | 部分环境可能未装 | 预处理脚本使用进度条，建议显式安装 |

`occwl` 只能通过 conda 安装（来自 `lambouj` channel），不能 pip 安装。

## 推荐：从零创建环境

### 1. 创建 conda 环境

```bash
cd /path/to/BRT
conda env create -f environment.yml
conda activate brt
```

当前 `environment.yml` 已补充：
- channel：`lambouj`（提供 `occwl`）
- 依赖：`occwl`、`tqdm`

### 2. 修复 NumPy 用户目录冲突（重要）

若本机 `~/.local` 下曾用 pip 装过 numpy 2.x，激活 `brt` 后仍可能误加载用户级包。在环境中加入激活脚本：

```bash
# 创建目录
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

# activate：禁用用户 site-packages
cat > "$CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh" << 'EOF'
#!/bin/sh
export _BRT_OLD_PYTHONNOUSERSITE="${PYTHONNOUSERSITE-}"
export PYTHONNOUSERSITE=1
EOF

# deactivate：恢复原有设置
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/no_user_site.sh" << 'EOF'
#!/bin/sh
if [ -n "${_BRT_OLD_PYTHONNOUSERSITE+x}" ]; then
    export PYTHONNOUSERSITE="${_BRT_OLD_PYTHONNOUSERSITE}"
    unset _BRT_OLD_PYTHONNOUSERSITE
else
    unset PYTHONNOUSERSITE
fi
EOF

chmod +x "$CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh"
chmod +x "$CONDA_PREFIX/etc/conda/deactivate.d/no_user_site.sh"
```

重新激活后验证 numpy 路径应指向 conda 环境：

```bash
conda deactivate && conda activate brt
python -c "import numpy; print(numpy.__version__, numpy.__file__)"
# 期望：1.23.5 .../envs/brt/lib/python3.10/site-packages/numpy/__init__.py
```

若仍指向 `~/.local`，也可临时手动设置：

```bash
export PYTHONNOUSERSITE=1
```

或卸载用户级 numpy（影响其他非 conda 项目，慎用）：

```bash
pip uninstall numpy  # 在 --user 安装的环境下执行
```

### 3. 已有环境时的增量修复

若环境已存在但缺包或报错，可执行：

```bash
conda activate brt
conda install -c lambouj -c conda-forge occwl tqdm -y
# 然后按上一节配置 no_user_site.sh
```

## 验证环境

```bash
conda activate brt
python -c "
import torch, numpy, scipy, occwl, OCC, tqdm
print('torch', torch.__version__)
print('numpy', numpy.__version__)
print('cuda', torch.cuda.is_available())
print('occwl ok')
"
```

单文件冒烟测试（可选）：

```bash
cd process
mkdir -p logs

# 准备测试数据
mkdir -p /tmp/mechcad_one/bearing
cp /path/to/one/file.stp /tmp/mechcad_one/bearing/0.stp

python -c "
from solid_to_triangles2 import main
main(['/tmp/mechcad_one/bearing', '/tmp/mechcad_one_out/topo/brt/bearing',
      '--num_processes', '1', '--no_random_name', '--method', '10', '--no_label'])
main(['/tmp/mechcad_one/bearing', '/tmp/mechcad_one_out/triangles/triangles/bearing',
      '--num_processes', '1', '--no_random_name', '--method', '8', '--no_label'])
"

python split_dataset.py \
  /tmp/mechcad_one_out/triangles/triangles \
  /tmp/mechcad_one_out/topo/brt \
  /tmp/mechcad_one_out/datasplit.json
```

## 运行 TMCAD 预处理

```bash
conda activate brt
cd process

# 提取拓扑
python gen_tmcad_topo.py /data/hdd/datasets/mechcad/mechcad/ /data/hdd/datasets/mechcad/processed/topology

# 提取面几何（三角 Bézier 面片）
python gen_tmcad_triangles.py /data/hdd/datasets/mechcad/mechcad/ /data/hdd/datasets/mechcad/processed/triangles

# 划分数据集（路径需指向含类别子目录的实际输出层）
python split_dataset.py \
  /data/hdd/datasets/mechcad/processed/triangles/triangles \
  /data/hdd/datasets/mechcad/processed/topology/brt \
  /data/hdd/datasets/mechcad/processed/datasplit.json
```

脚本默认 `process_num=30` 多进程，全量数据耗时较长；需 GPU 时 PyTorch CUDA 12.1 即可，预处理本身主要依赖 CPU + OpenCascade。

## 关键依赖一览

| 包 | 用途 | 安装来源 |
|----|------|----------|
| pythonocc-core | OpenCascade Python 绑定 | conda-forge |
| occwl | B-rep 几何/拓扑高层 API | lambouj |
| torch 2.2.1 + cuda 12.1 | BRT 图结构构建等 | pytorch / nvidia |
| numpy 1.23.5 | 数值计算（须 <2，与 torch 兼容） | pip（yml 指定） |
| scipy, geomdl | 几何/样条计算 | pip |
| tqdm | 进度条 | conda-forge |
| dgl 2.1.0+cu121 | 训练阶段图神经网络（预处理可不严格依赖） | dglteam |

## 常见问题

**Q: `conda env create` 很慢或解析失败？**  
A: 可改用 mamba：`mamba env create -f environment.yml`；`occwl` 单独装：`mamba install -c lambouj -c conda-forge occwl`。

**Q: 只有预处理，是否必须装 dgl / pytorch-cuda？**  
A: 预处理脚本会 `import torch`，且 `solid_to_brt.py` 会用到；建议按完整 yml 安装。无 GPU 机器可装 CPU 版 PyTorch，但需自行调整 yml。

**Q: 如何完全重建环境？**  
```bash
conda env remove -n brt -y
conda env create -f environment.yml
conda activate brt
# 再执行「修复 NumPy 用户目录冲突」一节
```

---

## 附录：bash `set` 用法大全

`set` 是 bash 内建命令，用于**查看或修改当前 shell 的行为选项**，以及**重置位置参数**（`$1` `$2` …）。脚本开头写 `set -e` 等，就是在声明「出错时怎么处理」。

### 基本语法

| 写法 | 含义 |
|------|------|
| `set 选项` / `set -选项` | 开启（enable）某选项 |
| `set +选项` | 关闭（disable）某选项 |
| `set` | 打印当前所有 `-` 开头的已开启选项 |
| `set -o` | 列出所有选项及开/关状态 |
| `set -o 选项名` | 查看单个选项是否开启，如 `set -o pipefail` |
| `set -- arg1 arg2 …` | 重置位置参数 `$1` `$2` …（与选项无关的另一用法） |

规律：**`-` 开启，`+` 关闭**（与常见命令行 `-v` / `+v` 类似）。

### 脚本中最常用的选项

| 短选项 | 长选项名 | 开启 `set -…` | 关闭 `set +…` | 作用 |
|--------|----------|---------------|---------------|------|
| `-e` | `errexit` | 遇错即退 | 忽略非零退出码 | 任一简单命令失败（退出码 ≠ 0）时，shell 立即退出 |
| `-u` | `nounset` | 未定义即错 | 允许未定义变量 | 引用未赋值变量时报错退出 |
| `-o pipefail` | `pipefail` | 管道任一失败即失败 | 只看管道最后一个命令 | `cmd1 \| cmd2` 中任一失败，整条管道退出码非 0 |
| `-x` | `xtrace` | 跟踪执行 | 关闭跟踪 | 执行前打印每条命令（调试脚本） |
| `-v` | `verbose` | 回显输入 | 关闭回显 | 读取时先打印命令行再执行 |
| `-n` | `noexec` | 只检查不执行 | 正常执行 | 语法检查模式，不真正跑命令 |
| `-C` | `noclobber` | 禁止覆盖 | 允许覆盖 | 重定向 `>` 时若目标文件已存在则失败 |
| `-E` | — | ERR trap 继承 | 不继承 | 在函数/子 shell 中仍触发 `trap ERR`（bash 特有） |
| `-H` | `histexpand` | 启用 `!` 历史展开 | 禁用 | 交互式里 `!` 历史替换；脚本里常 `set +H` |
| `-f` | `noglob` | 禁用通配符 | 启用通配符 | `*` `?` 等不再展开 |
| `-a` | `allexport` | 自动 export | 关闭 | 之后定义的变量自动 `export` 到子进程 |
| `-b` | `notify` | 异步作业立即通知 | 等提示符再通知 | 主要影响交互式 job 控制 |
| `-T` | — | 继承 DEBUG/RETURN/ERR trap | 不继承 | 函数内 trap 行为（bash 4.4+） |

### 常见组合写法

| 组合 | 含义 | 典型场景 |
|------|------|----------|
| `set -e` | 命令失败即退出 | 最基础的严格脚本 |
| `set -eo pipefail` | 失败即退 + 管道内失败也算失败 | **本项目 `scripts/preprocess.sh` 使用** |
| `set -eu` | 失败即退 + 禁止未定义变量 | 较严格；与 conda 等第三方脚本可能冲突 |
| `set -euo pipefail` | 上述三者全开 | 俗称「严格模式」；很多项目默认写法 |
| `set -ex` | 失败即退 + 打印每条命令 | 调试脚本 |
| `set -euxo pipefail` | 严格 + 跟踪 | 调试严格脚本 |

### 与 `$` 变量语法的对比（易混点）

| 写法 | 类型 | 说明 |
|------|------|------|
| `VAR=value` | 变量**赋值** | 正确；**不要**写成 `$VAR=value` |
| `$VAR` / `${VAR}` | 变量**引用** | 读取已定义变量的值 |
| `set -e` | shell **选项** | 与变量无关；`-e` 是 errexit 开关 |
| `set -- "$@"` | 重置**位置参数** | 常用于 getopts 或参数重组 |

### `set -e` 的细节与例外

| 情况 | `set -e` 下是否退出 |
|------|----------------------|
| 简单命令返回非 0 | 是 |
| `if cmd; then …; fi` 中的条件命令 | 否（条件判断允许失败） |
| `cmd1 \|\| cmd2` | 否（`\|\|` 本身在处理失败） |
| `cmd1 && cmd2` | 仅当最终逻辑需要失败时 |
| 管道 `a \| b`（无 `pipefail`） | 否（默认只看最后一个 `b`） |
| 管道 `a \| b`（有 `pipefail`） | 任一段失败则退出 |
| 子 shell `( … )` / 命令替换 | 子 shell 内 `-e` 仍生效 |

### 位置参数用法（`set --`）

| 命令 | 效果 |
|------|------|
| `set -- foo bar` | `$1=foo`，`$2=bar`，`$#=2` |
| `set -- "$@" extra` | 在原有参数后追加 `extra` |
| `set --` | 清空位置参数（`$#=0`） |
| `shift` | 去掉 `$1`，后面参数前移 |

### 查看与临时开关示例

```bash
set -o              # 列出所有选项状态
set -o pipefail     # 查看 pipefail 是否开启
set +x              # 调试结束后关闭 xtrace

set +e              # 临时允许失败
some_may_fail
set -e              # 恢复遇错即退
```

### 与本项目相关的注意点

| 场景 | 建议 |
|------|------|
| `scripts/preprocess.sh` | 使用 `set -eo pipefail`（**不加 `-u`**） |
| `conda activate` 报「未绑定的变量」 | 多为 `set -u` 与 conda 的 `deactivate.d/*.sh` 冲突；去掉 `-u` 或在 `conda activate` 前后 `set +u` / `set -u` |
| 长任务预处理 | `-e` 保证某步失败后不继续写脏数据；`pipefail` 保证 `\| tee log` 里 python 失败时脚本也失败 |
| 调试预处理脚本 | 临时改为 `set -exo pipefail` 看每条命令 |

### 快速查阅：`set -o` 常见输出对照

| `set -o` 显示名 | 短选项 | 一句话 |
|-----------------|--------|--------|
| `errexit` | `-e` | 失败就停 |
| `nounset` | `-u` | 未定义变量不能用 |
| `pipefail` | `-o pipefail` | 管道里谁失败都算失败 |
| `xtrace` | `-x` | 打印执行的命令 |
| `verbose` | `-v` | 打印读入的命令行 |
| `noclobber` | `-C` | 禁止 `>` 覆盖已有文件 |
| `noglob` | `-f` | 关闭 `*` 通配符 |
| `allexport` | `-a` | 变量自动 export |
| `hashall` | `-h` | 记住命令路径（默认开） |
| `monitor` | `-m` | 启用 job 控制（交互式） |
