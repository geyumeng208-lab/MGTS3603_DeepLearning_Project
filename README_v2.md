# 基于淘宝用户行为数据的购买预测序列建模

> **课程：** 深度学习（2026 春季）  
> **项目地址：** https://github.com/geyumeng208-lab/MGTS3603_DeepLearning_Project

---

## 项目描述

本项目聚焦电商场景中的用户行为序列建模问题：根据用户当前 session 及历史行为序列，预测用户是否会发生购买行为，并输出购买概率。在真实电商平台中，用户会产生浏览、收藏、加购、购买等多种行为，不同行为类型、行为时间间隔、当前 session 短期兴趣和长期历史兴趣都会影响最终转化。

项目以 LSTM 作为序列建模基线，并尝试 SIM、ETA、TWIN 等长序列检索/注意力模型作为对比。在此基础上，最终选择 HyFormer 系列模型作为主方案，因为 HyFormer 的多序列结构更适合融合电商场景中的异构信息，例如：

- 当前 session 短期行为序列
- 长期历史行为序列
- brand / cate 异构序列
- `pv`、`fav`、`cart`、`buy` 等行为类型
- 时间间隔与近期行为衰减
- 用户画像和商品侧静态特征

最终最优模型为 **`HyFormer-Hierarchical`**，它在 HyFormer-Session 和 HyFormer-Static 的基础上进一步引入长序列分层压缩：当前 session 保留细粒度行为，最近历史保留最近 100 条行为，更早历史通过 chunk pooling 压缩为长期兴趣序列。在用户级验证切分下，最终达到 **AUC 0.7988、GAUC 0.7047**。

此外，项目还进行了 4 个方向的探索性实验：TWIN 优化、知识蒸馏、模型压缩与部署方案，详见各实验记录文档。

---

## 团队成员与贡献

| 成员 | 主要贡献 | 产出 |
|------|---------|------|
| **geyumeng208-lab** | 项目架构搭建与维护、数据处理、HyFormer 系列全部模型实现、外部数据集验证、时延基准测试、整体文档编写 | 17 个基线与 HyFormer 模型、完整的数据管线、时延测试脚本、README 文档 |
| **LovelyShark35** | 探索性实验（TWIN 优化、蒸馏、压缩、部署） | 5 个 TWIN 变体、蒸馏训练器、量化/剪枝/流水线脚本 |
| **MarcoCola** | 数据集收集与预处理、基线模型搭建（LSTM / LSTM-Attention / Transformer-Baseline）、模型性能跨数据集对比分析 | 基线模型实现、数据集构建脚本、跨数据集分析报告 |
| **a2005** | 自适应 Session 阈值、硬负采样、多任务学习 | Data/Trainer/Config 改造，多任务 btag 损失 |

---

## 实验环境

```bash
conda activate foodvision
pip install -r requirements.txt

# HyFormer 需要外部依赖（不是 pip 包）：
cd external && git clone https://github.com/WestbrookLong/Hyformer_Pytorch.git
```

### 默认配置

完整配置详见 `configs/default.yaml`。关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_path` | （空）→ 合成数据 | 淘宝 CSV 数据路径 |
| `model` | `twin` | 模型名称 |
| `epochs` | 3 | 训练轮数 |
| `max_seq_len` | 1000 | 最大历史序列长度 |
| `batch_size` | 128 | 批大小 |
| `embedding_dim` | 32 | Embedding 维度 |
| `top_k` | 50 | Top-K 筛选数 |
| `pos_weight` | 0.0 | 正样本加权 |
| `session_gap_minutes` | 30.0 | Session 切分间隔 |
| `adaptive_session_gap` | False | 自适应 Session 阈值 |
| `multitask_loss_weight` | 0.0 | 多任务 btag 损失权重 |
| `auto_pos_weight` | False | 自动计算正样本权重 |

---

## 模型家族

### 基线模型

| 模型 | 核心思想 | 文件 |
|------|---------|------|
| `base` / `LSTM` | LSTM 编码行为序列，取最后隐状态 | `src/models/lstm.py` |
| `LSTM-Attention` | LSTM 输出上加 masked attention pooling | `src/models/sequence_baselines.py` |
| `Transformer-Baseline` | 标准 Transformer encoder 建模完整历史 | `src/models/sequence_baselines.py` |

### 检索式长序列模型

| 模型 | 核心思想 | 文件 |
|------|---------|------|
| `SIM` | 类别硬筛 GSU + Target Attention ESU | `src/models/sim.py` |
| `ETA` | SimHash 二进制指纹 + 汉明距离 Top-K | `src/models/eta.py` |
| `TWIN` | 共享多头注意力 CP-GSU Top-K + ESU | `src/models/twin.py` |

### HyFormer 系列

参考 [HyFormer (arXiv 2601.12681)](https://arxiv.org/abs/2601.12681)，基于 [WestbrookLong/Hyformer_Pytorch](https://github.com/WestbrookLong/Hyformer_Pytorch) 主干。

| 模型 | 核心思想 | 文件 |
|------|---------|------|
| `HyFormer` | 序列编码器 + query tokens + non-sequence tokens | `src/models/hyformer.py` |
| `HyFormer-Opt` | 两条异构序列（brand + cate） | `src/models/hyformer_optimized.py` |
| `HyFormer-Time` | 时间间隔、时间桶、衰减 gate | `src/models/hyformer_time.py` |
| `HyFormer-Event` | 行为类型 embedding | `src/models/hyformer_event.py` |
| `HyFormer-MultiGranularity` | 按行为类型拆分为多条序列 | `src/models/hyformer_multigrain.py` |
| `HyFormer-Session` | 相邻时间间隔切分 session | `src/models/hyformer_session.py` |
| `HyFormer-Static` | + 用户画像和商品统计特征 | `src/models/hyformer_static.py` |
| **`HyFormer-Hierarchical`** | **最优模型**，长序列分层压缩 | `src/models/hyformer_hierarchical.py` |
| `HyFormer-Dynamic` | 动态 recent 长度（低活跃用户保留更多） | `src/models/hyformer_dynamic.py` |
| `HyFormer-TopK` | 长期历史 Top-K 筛选 | `src/models/hyformer_topk.py` |
| `HyFormer-OfflineLong` | 长期兴趣离线预计算缓存 | `src/models/hyformer_offline_long.py` |

### 探索性实验 TWIN 变体

| 变体 | 方案 | 文件 |
|------|------|------|
| `twin_nonlinear_sim` | MLP 替代点积  | `src/models/twin_nonlinear_sim.py` |
| `twin_gate_fusion` | GSU-ESU 门控融合 |  `src/models/twin_gate_fusion.py` |
| `twin_shared_emb` | item/cate 共享 embedding | `src/models/twin_shared_emb.py` |
| **`twin_gumbel_topk`** | Gumbel-Softmax 可微 TopK |  `src/models/twin_gumbel_topk.py` |

---

## 快速开始

### 训练基线

```bash
# 使用合成数据快速验证
python train.py --model twin

# 使用淘宝数据
python train.py --model twin --data_path data/purchase_sequence_100k_static_long500.csv --epochs 3 --max_seq_len 100

# FP16 混合精度（NVIDIA GPU 预期加速 20-30%）
python train.py --model twin --data_path ... --fp16
```

### 训练 HyFormer 系列

```bash
# 最优模型
python train.py --model hyformer_hierarchical --data_path data/purchase_sequence_100k_static_long500.csv --epochs 6 --max_seq_len 500

# 会话切分变体
python train.py --model hyformer_session --data_path ... --epochs 3 --max_seq_len 100

# 离线缓存长期兴趣（更适合在线部署）
python train.py --model hyformer_offline_long --data_path ... --epochs 3 --max_seq_len 500
```

### 训练 TWIN 变体

```bash
# 最佳 TWIN 变体
python train.py --model twin_gumbel_topk --data_path ... --epochs 3 --max_seq_len 100

# 参数共享
python train.py --model twin_shared_emb --data_path ... --epochs 3 --max_seq_len 100
```

### 知识蒸馏

```bash
python train_distill.py --model twin --teacher hyformer_hierarchical --alpha 0.5 --T 2.0 --data_path ... --epochs 3 --max_seq_len 100
```

### 模型压缩

```bash
# INT8 权重量化
python scripts/quantize_twin.py --model twin --data_path ...

# Embedding 剪枝
python scripts/prune_embedding.py --model twin --data_path ... --prune_ratio 0.25 --finetune_epochs 2
```

### 两阶段推理流水线

```bash
# 1. 先训练 HyFormer
python train.py --model hyformer_hierarchical --data_path ... --epochs 6 --max_seq_len 100

# 2. 运行两阶段流水线（TWIN 粗排 + HyFormer 精排）
python scripts/two_stage_benchmark.py --data_path ... --top_k_ratio 0.5
```

---

## 实验结果汇总

### 淘宝主数据结果

统一设置：`--epochs 3 --max_seq_len 100`；`*` 标记为 6 epoch, seq_len=500

| 模型 | AUC | GAUC | 延迟 | 说明 |
|------|-----|------|------|------|
| LSTM | 0.6610 | 0.6170 | ~3ms | 循环基线 |
| LSTM-Attention | 0.6940 | 0.6640 | ~3ms | + 注意力 |
| Transformer | 0.7030 | 0.6700 | ~5ms | 标准 Transformer |
| SIM | 0.7370 | 0.6980 | ~6ms | 搜索兴趣池化 |
| ETA | 0.6460 | 0.5910 | ~6ms | 哈希化注意力 |
| TWIN | 0.7116 | 0.6384 | **~7.6ms** | 轻量检索 |
| HyFormer-Session | 0.7860 | 0.6860 | ~17ms | Session 切分 |
| HyFormer-Static | 0.7943 | 0.6824 | ~17ms | + 静态特征 |
| **HyFormer-Hierarchical \*** | **0.7988** | **0.7047** | ~27ms | **最优** |
| HyFormer-OfflineLong \* | 0.7940 | 0.7040 | ~13ms | 离线缓存 |

### 探索性实验结果

| 实验 | AUC (Epoch 03) | GAUC (Epoch 03) | 
|------|---------------|----------------|
| TWIN 基线 | 0.7116 | 0.6384 |
| 非线性相似度 | 0.7111 | 0.6301 |
| 可学习门控 | 0.7116 | 0.6363 | 
| 知识蒸馏 (α=0.5) | 0.7052 | 0.6400 | 
| FP16 AMP | 0.7127 | 0.6361 |
| INT8 量量化 | 0.7116 | 0.6385 | 
| Embedding 剪枝 | 0.6990 | 0.6327 |
| **Gumbel-Softmax TopK** | **0.7139** | **0.6507** | 
| **两阶段流水线 (50/50)** | **0.7354** | **0.6784** | 

### DIGINETICA 外部数据验证

统一设置：`--epochs 1 --batch_size 512 --max_seq_len 100 --pos_weight 14.9`

| 模型 | AUC | GAUC |
|------|-----|------|
| LSTM | 0.8777 | 0.7150 |
| Transformer | 0.8740 | 0.7170 |
| TWIN | 0.7140 | 0.6913 |
| HyFormer-Session | 0.8785 | **0.7182** |
| **HyFormer-Static** | **0.8804** | 0.7120 |

---

## 关键发现

1. **HyFormer-Hierarchical** 是离线效果最优模型（AUC 0.7988, GAUC 0.7047）
2. **Gumbel-Softmax TopK** 是唯一有效改进（GAUC +1.9%），训练可微推理零开销
3. **两阶段流水线**（TWIN + HyFormer）用 78% 延迟换回 47% AUC 增益
4. **FP16 / INT8 精度完全无损**，有 GPU 组员可直接加速
5. **item/cate 参数共享可行**，节省 ~0.9% 参数
6. 短序列场景下 **LSTM** 和 **HyFormer-Session** 是简单有效的强基线

---

## 文档清单

| 文档 | 说明 |
|------|------|
| [探索性实验计划.md](探索性实验计划.md) | 全部 18 个实验步骤的详细记录 |
| [探索性实验报告.md](探索性实验报告.md) | 给组员的改动说明和复现指南 |
| [数据集差异分析报告.md](数据集差异分析报告.md) | MarcoCola — 淘宝 vs DIGINETICA 性能对比 |
| [评分优化建议.md](评分优化建议.md) | 答辩 + AI 审查的改进建议 |
| [答辩汇报大纲.md](答辩汇报大纲.md) | 10 分钟答辩脚本 |

---

## 项目结构

```text
.
├── configs/default.yaml
├── requirements.txt
├── train.py                          # 统一训练入口
├── train_distill.py                  # 知识蒸馏
├── scripts/
│   ├── benchmark_latency.py          # 延迟基准
│   ├── build_purchase_sequence.py    # 数据预处理
│   ├── build_diginetica_sequence.py
│   ├── preprocess_taobao_tables.py
│   ├── quantize_twin.py              # INT8 量化
│   ├── prune_embedding.py            # Embedding 剪枝
│   └── two_stage_benchmark.py        # 两阶段流水线
├── tests/smoke_test.py
├── data/                             # 本地数据（不上传）
├── external/Hyformer_Pytorch/        # 外部依赖
└── src/
    ├── data.py / metrics.py / trainer.py / utils.py
    └── models/
        ├── __init__.py               # 模型注册表
        ├── base.py                   # CTRBaseModel + MLP
        ├── attention.py              # 注意力工具
        ├── lstm.py                   # LSTM 基线
        ├── sequence_baselines.py     # LSTM-Attention / Transformer
        ├── sim.py / eta.py           # 检索式模型
        ├── twin.py                   # TWIN 原始
        ├── twin_nonlinear_sim.py     # 实验：非线性相似度
        ├── twin_gate_fusion.py       # 实验：门控融合
        ├── twin_shared_emb.py        # 实验：参数共享
        ├── twin_gumbel_topk.py       # 实验：Gumbel-Softmax 
        ├── hyformer.py ~ hyformer_offline_long.py  # HyFormer 系列
```
