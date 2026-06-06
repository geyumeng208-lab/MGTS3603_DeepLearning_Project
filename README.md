# 基于淘宝用户行为数据的购买预测序列建模

## 项目描述

本项目聚焦电商场景中的用户行为序列建模问题：根据用户当前 session 及历史行为序列，预测用户是否会发生购买行为，并输出购买概率。在真实电商平台中，用户会产生浏览、收藏、加购、购买等多种行为，不同行为类型、行为时间间隔、当前 session 短期兴趣和长期历史兴趣都会影响最终转化。

项目以 LSTM 作为序列建模基线，并尝试 SIM、ETA、TWIN 等长序列检索/注意力模型作为对比。在此基础上，最终选择 HyFormer 系列模型作为主方案，因为 HyFormer 的多序列结构更适合融合电商场景中的异构信息，例如：

- 当前 session 短期行为序列
- 长期历史行为序列
- brand / cate 异构序列
- `pv`、`fav`、`cart`、`buy` 等行为类型
- 时间间隔与近期行为衰减
- 用户画像和商品侧静态特征

最终最优模型为 `HyFormer-Hierarchical`，它在 HyFormer-Session 和 HyFormer-Static 的基础上进一步引入长序列分层压缩：当前 session 保留细粒度行为，最近历史保留最近 100 条行为，更早历史通过 chunk pooling 压缩为长期兴趣序列。在 10 万条购买预测样本上，最终达到 AUC 0.8058、GAUC 0.7126。

本项目实现的主要模型包括：

- `base` / `LSTM`：将用户历史行为序列输入 LSTM，取最后有效隐状态作为长期兴趣表示。
- `SIM`：类别硬筛 GSU + Target Attention ESU。
- `ETA`：SimHash 二进制指纹 + 汉明距离 Top-K 检索 + Target Attention。
- `TWIN`：作为对比实验，参考 TWIN 论文的 CP-GSU 思路，使用共享多头 Target Attention 分数完成 GSU Top-K 检索和 ESU 兴趣聚合，并加入压缩 cross feature bias。
- `HyFormer`：基于 [WestbrookLong/Hyformer_Pytorch](https://github.com/WestbrookLong/Hyformer_Pytorch) 的 HyFormerBackbone，使用序列编码器、query tokens、non-sequence tokens 和 QueryBoostMixer 建模用户行为序列。
- `HyFormer-Opt`：固定 HyFormerBackbone 不变，将历史 brand 序列和 cate 序列拆成两条异构序列，并增强 non-sequence context，用于当前购买预测任务。
- `HyFormer-Time`：在 HyFormer-Opt 基础上加入历史行为到预测时刻的时间间隔、相邻行为间隔、时间桶 embedding 和时间衰减 gate。
- `HyFormer-Event`：在 HyFormer-Time 基础上加入行为类型 embedding，将 `pv`、`fav`、`cart`、`buy` 编码进历史行为 token。
- `HyFormer-MultiGranularity`：将历史行为按 `pv`、`fav`、`cart`、`buy` 拆成多条异构序列，分别建模后由 HyFormer 融合。
- `HyFormer-Session`：根据相邻行为时间间隔切分当前 session 和长期历史，分别建模短期实时兴趣与长期稳定兴趣。
- `HyFormer-Static`：在 HyFormer-Session 基础上加入用户画像和商品/类目侧静态统计特征，作为 non-sequence tokens 融合。
- `HyFormer-Hierarchical`：在长历史序列中保留最近行为细粒度表示，并将更早历史按 chunk pooling 压缩为长期兴趣序列。

项目默认支持两种数据来源：

1. 使用 `--data_path` 指向淘宝广告风格 CSV。
2. 不传数据路径时自动生成合成数据，便于快速跑通训练和测试。

## 数据挂载

由于原始淘宝数据和预处理后的训练 CSV 文件较大，仓库不会直接上传数据文件。`.gitignore` 已排除以下路径：

```text
data/
src/*.csv
src/sampled_10pct/
external/
```

clone 仓库后，需要手动将数据放回本地目录。推荐目录结构如下：

```text
Project_DeepLearning/
├── src/
│   └── sampled_10pct/
│       ├── raw_sample.csv
│       ├── ad_feature.csv
│       ├── user_profile.csv
│       └── behavior_log.csv
└── data/
    ├── purchase_sequence_100k_event.csv
    ├── purchase_sequence_100k_static.csv
    └── purchase_sequence_100k_static_long500.csv
```

如果只有四张原始抽样表，可以重新生成训练 CSV：

```bash
python scripts/build_purchase_sequence.py --behavior_log src/sampled_10pct/behavior_log.csv --user_profile src/sampled_10pct/user_profile.csv --ad_feature src/sampled_10pct/ad_feature.csv --output data/purchase_sequence_100k_static_long500.csv --max_samples 100000 --max_history 500 --min_history 5 --neg_sample_rate 0.02 --with_static_features
```

如果使用 Google Drive、OneDrive 或移动硬盘，可以把数据目录挂载/复制到上述位置；也可以通过 `--data_path` 指向任意位置的已预处理 CSV，例如：

```bash
python train.py --model hyformer_hier --data_path D:/datasets/purchase_sequence_100k_static_long500.csv --epochs 2 --batch_size 256 --max_seq_len 500
```

## 环境

```bash
pip install -r requirements.txt
```

## 快速运行

```bash
python train.py --model hyformer_hier --data_path data/purchase_sequence_100k_static_long500.csv --epochs 2 --batch_size 256 --max_seq_len 500
```

也可以切换模型：

```bash
python train.py --model base
python train.py --model sim
python train.py --model eta
python train.py --model twin
python train.py --model hyformer
python train.py --model hyformer_opt
python train.py --model hyformer_time
python train.py --model hyformer_event
python train.py --model hyformer_multigrain
python train.py --model hyformer_session
python train.py --model hyformer_static
python train.py --model hyformer_hier --max_seq_len 500
```

## CSV 数据格式

如果使用真实或预处理后的淘宝广告数据，请准备如下列：

```text
user_id,ad_id,cate_id,label,hist_ad_ids,hist_cate_ids
```

其中 `hist_ad_ids` 和 `hist_cate_ids` 是用空格分隔的长序列，例如：

```text
101,9988,16,1,"11 29 31 42","3 16 16 8"
```

运行：

```bash
python train.py --model twin --data_path data/taobao_ads.csv --max_seq_len 1000 --top_k 50
```

## 输出指标

训练脚本会输出：

- `loss`：二分类交叉熵损失
- `auc`：全局 AUC
- `gauc`：按用户分组的 GAUC，更贴近广告排序评估

## 结果分析

### 任务定义

本课程项目聚焦电商场景中的用户行为建模问题。在真实平台中，用户会在一个 session 或一段时间窗口内产生点击、搜索、加购、收藏、购买等行为。模型的目标是根据用户当前 session 的实时行为序列，以及候选商品或广告信息，预测用户最终是否会产生购买或点击转化，并输出对应概率。

在本项目实现中，每条样本由用户 ID、候选广告或商品 ID、候选类目、历史行为序列和标签组成。历史行为序列用于刻画用户兴趣，标签 `label=1` 表示用户对候选对象产生正反馈，`label=0` 表示未产生正反馈。若使用 Taobao User Behavior Dataset，可将点击、加购、收藏等行为作为输入序列，将后续是否购买作为标签；若使用淘宝广告点击率数据，则可将广告点击作为正样本标签。

### 序列构建与标签定义

序列构建方式对结果影响较大。本项目采用按时间顺序排列的用户历史行为序列，并保留最近 `max_seq_len` 条行为。对于长序列场景，过短的历史窗口会丢失长期兴趣，过长的窗口又会引入噪声和计算开销。因此项目默认保留最多 1000 条历史行为，并在模型内部通过检索或注意力机制选择与当前候选对象更相关的行为。

标签定义方面，本项目使用二分类设置：

- 正样本：用户在当前 session 或预测窗口内产生购买、点击等目标行为。
- 负样本：用户曝光或浏览后未产生目标行为。

训练目标为二分类交叉熵损失，模型输出经过 sigmoid 后即为购买或点击转化概率。

### 模型对比

`base` 模型采用 LSTM 对行为序列进行编码。LSTM 能够按顺序读取用户行为，捕捉一定的时间依赖关系，是序列建模任务中常用的基线方法。但在电商长序列场景下，LSTM 存在两个问题：第一，序列很长时早期行为信息容易被压缩或遗忘；第二，所有行为被统一编码，模型难以显式区分哪些行为与当前候选商品最相关。

`SIM` 在 LSTM 基线之上引入了“先检索、再精排”的思想。GSU 阶段先根据候选商品类目从长历史中筛选相关行为，减少无关行为干扰；ESU 阶段使用 Target Attention，让模型根据候选商品动态判断不同行为的重要性。相比直接使用 LSTM，SIM 更适合处理超长用户行为序列。

`ETA` 进一步解决 SIM 中 GSU 和 ESU 两阶段目标不完全一致的问题。它使用 SimHash 将候选 item 和历史行为向量转化为二进制指纹，并通过汉明距离检索 Top-K 相似行为。这样检索阶段不再只依赖类目硬规则，而是可以基于向量语义选择更相似的历史行为，使召回结果与后续注意力建模更加一致。

`TWIN` 则进一步统一了 GSU 检索和 ESU 注意力的相似度计算逻辑。它使用同一套相似度函数完成候选行为筛选和兴趣聚合，减少两阶段之间的目标偏差。同时，通过特征压缩降低长序列建模的计算复杂度，更适合大规模电商平台中的实时预测场景。

### 实验设置

为了更贴合课程要求，本项目从 `behavior_log.csv` 中构造购买预测任务：使用用户当前行为之前的历史行为序列作为输入，将当前行为是否为 `buy` 作为标签。正样本为购买行为，负样本为浏览、收藏、加购等非购买行为的抽样结果。主要实验使用 10% 用户抽样数据，构造 100,000 条训练样本。

需要注意的是，下面实验分为三类：主模型对比、严格消融实验和扩展实验。主模型对比和严格消融实验尽量保持同一任务、同一数据构造、同一训练轮数；扩展实验会改变负样本分布或历史长度，因此不与严格消融结果做单变量比较。

### 主模型对比

主模型对比用于比较不同序列建模框架的基础能力。所有模型均在购买预测任务上训练 2 个 epoch，主要使用最大历史长度 100 的数据设置。

| 模型 | 序列建模方式 | AUC | GAUC | 说明 |
| --- | --- | ---: | ---: | --- |
| base / LSTM | LSTM 编码完整历史序列 | 0.6905 | 0.6142 | 基线模型 |
| SIM | 类目硬筛 + Target Attention | 0.6962 | 0.6227 | 注意力召回与聚合 |
| ETA | SimHash 检索 + Target Attention | 0.6884 | 0.6200 | 向量检索式召回 |
| TWIN-lite | 简化统一相似度检索与注意力聚合 | 0.6866 | 0.6202 | 旧版轻量实现 |
| TWIN | CP-GSU + 多头 Target Attention + cross bias | 0.7003 | 0.6136 | TWIN 论文启发实现 |
| HyFormer | HyFormerBackbone + QueryBoostMixer | 0.6938 | 0.6194 | 开源 HyFormer 适配版 |
| HyFormer-Opt | 双序列 HyFormerBackbone + 增强上下文 | 0.7008 | 0.6186 | 针对本任务的 HyFormer 基础优化 |

从主模型对比可以看出，LSTM 已经能够学习基本购买转化模式，但 HyFormer-Opt 和 TWIN 在 AUC 上更好，说明注意力、多序列结构和目标相关建模对购买预测是有效的。由于本项目后续需要融合行为类型、时间、session 和静态特征，HyFormer 的异构多序列结构更适合作为最终主线。

### 严格消融实验

严格消融实验用于分析各个模块对 HyFormer 的增益。该部分尽量保持相同样本规模、相同训练轮数和相同最大历史长度 100，只逐步加入特征或结构模块。不同 CSV 文件主要是为了增加字段，但样本构造 seed 和采样流程保持一致。

| 模型 | 增量模块 | AUC | GAUC | 结论 |
| --- | --- | ---: | ---: | --- |
| HyFormer-Opt | brand / cate 双序列 + 增强上下文 | 0.7008 | 0.6186 | HyFormer 基础版本 |
| HyFormer-Time | 时间间隔 embedding + 时间衰减 | 0.7143 | 0.6401 | 时间特征显著提升用户内排序 |
| HyFormer-Event | 行为类型 embedding | 0.7536 | 0.6001 | 行为类型强烈提升整体购买概率判断，但 GAUC 下降 |
| HyFormer-MultiGranularity | 按 `pv/fav/cart/buy` 拆分多粒度序列 | 0.7620 | 0.6111 | 多序列行为类型建模优于简单 btag embedding |
| HyFormer-Session, 30min | 当前 session + 长期历史双序列 | 0.7869 | 0.6853 | session 实时行为是最关键增益 |
| HyFormer-Session, 60min | 当前 session + 长期历史双序列 | 0.7833 | 0.6679 | 60 分钟阈值有效，但弱于 30 分钟 |
| HyFormer-Static | 用户画像 + 商品/类目静态统计特征 | 0.8042 | 0.6970 | 静态特征补充纯行为序列无法表达的先验 |

严格消融说明，电商购买预测不仅依赖“用户看过什么”，还依赖“什么时候看”“以什么行为方式看”“是否属于当前 session”以及“用户和商品本身有什么属性”。其中 session 级建模带来最大的结构性提升，说明课程要求中强调的当前 session 实时行为非常关键。静态特征进一步提升 AUC 和 GAUC，说明用户画像和商品侧先验可以有效补充序列表示。

### 扩展实验

扩展实验不作为严格单变量消融，而是用于探索更接近真实业务的设置，包括更难负样本和更长历史序列。

| 实验 | 改变点 | AUC | GAUC | 说明 |
| --- | --- | ---: | ---: | --- |
| HyFormer-Session + hard negative | 提高 `fav/cart` 高意图负样本比例，正负约 1:1 | 0.7788 | 0.6545 | 验证集更难，不能与原始负采样直接比较 |
| HyFormer-Hierarchical | `max_seq_len=500`，最近 100 条细粒度 + 更早历史 8 chunk 压缩 | 0.8058 | 0.7126 | 长序列扩展设置，最终最优 GAUC |

Hard negative 实验改变了负样本分布，让模型区分“接近购买但尚未购买”和“真实购买”，更适合检验模型鲁棒性，但不能直接与原始负采样实验比较。长序列分层压缩实验将最大历史长度从 100 扩展到 500，并将更早历史压缩成 chunk 表示，说明长期历史对用户内排序有价值。

### 最终模型选择说明

在严格消融设置下，`HyFormer-Static` 是表现最好的模型，AUC 为 0.8042，GAUC 为 0.6970，说明 session 建模与静态特征融合是最稳定、最有效的核心方案。

在扩展长序列设置下，`HyFormer-Hierarchical` 进一步利用最多 500 条历史行为，并通过分层压缩控制计算成本，最终达到 AUC 0.8058、GAUC 0.7126。因此，本项目最终选择 `HyFormer-Hierarchical` 作为最终模型。

项目也尝试过直接使用 `raw_sample.csv` 中的广告点击标签进行 CTR 预测，但在只使用历史 `brand/cate` 序列时，启发式特征与点击标签的相关性较弱，模型 AUC 仅在 0.51 左右。因此最终实验采用 `behavior_log` 构造购买预测任务，更符合课程中“基于当前 session 实时行为预测最终是否购买”的要求。

综上，电商购买预测任务的有效建模方式不是简单编码完整历史序列，而是同时建模短期 session 兴趣、长期历史兴趣、行为类型、时间间隔和静态画像特征。HyFormer 的多序列结构天然适合这种异构信息融合；结合 session 切分、静态特征和长序列分层压缩后，模型能够在控制计算成本的同时更充分地利用用户行为历史。

## 项目结构

```text
.
├── configs/default.yaml
├── requirements.txt
├── train.py
├── tests/smoke_test.py
└── src
    ├── data.py
    ├── metrics.py
    ├── trainer.py
    ├── utils.py
    └── models
        ├── attention.py
        ├── base.py
        ├── eta.py
        ├── hyformer.py
        ├── hyformer_event.py
        ├── hyformer_hierarchical.py
        ├── hyformer_multigrain.py
        ├── hyformer_optimized.py
        ├── hyformer_session.py
        ├── hyformer_static.py
        ├── hyformer_time.py
        ├── lstm.py
        ├── sim.py
        └── twin.py
```
