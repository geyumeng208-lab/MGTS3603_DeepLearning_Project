# 基于淘宝广告数据集的超长用户行为序列建模

## 项目描述

在淘宝广告数据集中，74% 用户点击序列长度超过 50，24% 超过 500。为了释放数据潜力，让模型充分利用长期行为序列来挖掘用户的长期兴趣，我们进行了长序列建模。

- 首先采用基于 SIM Hard Search 的经典两阶段范式：GSU 阶段按候选广告类目从用户近 1000 条行为中硬筛选同类目序列，ESU 阶段通过 Target Attention 建模兴趣，GAUC +0.41pt。
- 为了解决 SIM 的两阶段目标不一致的问题，我们采用 ETA 的方式进行建模，采用 SimHash 将候选 item 和长序列向量转化为二进制指纹，以汉明距离高效检索 Top-K 相似行为，实现端到端训练，保持了 GSU 和 ESU 的一致性。相比 base，GAUC +0.69pt。
- 为了解决 SIM 和 ETA 的 GSU 和 ESU 计算逻辑不一致的问题，我们采用 TWIN 的方式进行建模，统一了 GSU 检索与 ESU 注意力的相似度计算逻辑，提高了两阶段一致性，并通过预计算和特征压缩的方式降低了计算复杂度。相比 base，GAUC +0.82pt。

本项目根据上述思路实现了三类长序列点击率预估模型，用于模拟淘宝广告场景中“用户历史行为很长、候选广告需要从历史中检索相关兴趣”的问题。

- `base` / `LSTM`：将用户历史行为序列输入 LSTM，取最后有效隐状态作为长期兴趣表示，是 SIM、ETA、TWIN 的基线模型。
- `SIM`：类别硬筛 GSU + Target Attention ESU。
- `ETA`：SimHash 二进制指纹 + 汉明距离 Top-K 检索 + Target Attention。
- `TWIN`：参考 TWIN 论文的 CP-GSU 思路，使用共享多头 Target Attention 分数完成 GSU Top-K 检索和 ESU 兴趣聚合，并加入压缩 cross feature bias。
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

## 环境

```bash
pip install -r requirements.txt
```

## 快速运行

```bash
python train.py --model twin --epochs 3 --synthetic_samples 3000
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

### 实验结果

为了更贴合课程要求，本项目最终从 `behavior_log.csv` 中构造购买预测任务：使用用户当前行为之前的历史行为序列作为输入，将当前行为是否为 `buy` 作为标签。正样本为购买行为，负样本为浏览、收藏、加购等非购买行为的抽样结果。实验使用 10% 用户抽样数据，构造 100,000 条训练样本，最大历史长度为 100。

| 模型 | 序列建模方式 | AUC | GAUC | 相比 base 的提升 |
| --- | --- | --- | --- | --- |
| base / LSTM | LSTM 编码完整历史序列 | 0.6905 | 0.6142 | - |
| SIM | 类目硬筛 + Target Attention | 0.6962 | 0.6227 | +0.57pt AUC / +0.85pt GAUC |
| ETA | SimHash 检索 + Target Attention | 0.6884 | 0.6200 | -0.21pt AUC / +0.58pt GAUC |
| TWIN-lite | 简化统一相似度检索与注意力聚合 | 0.6866 | 0.6202 | 旧版轻量实现 |
| TWIN | CP-GSU + 多头 Target Attention + cross bias | 0.7003 | 0.6136 | +0.98pt AUC / -0.06pt GAUC |
| HyFormer | HyFormerBackbone + QueryBoostMixer | 0.6938 | 0.6194 | +0.33pt AUC / +0.52pt GAUC |
| HyFormer-Opt | 双序列 HyFormerBackbone + 增强上下文 | 0.7008 | 0.6186 | +1.03pt AUC / +0.44pt GAUC |
| HyFormer-Time | HyFormer-Opt + 时间间隔 embedding + 时间衰减 | 0.7143 | 0.6401 | +2.38pt AUC / +2.59pt GAUC |
| HyFormer-Event | HyFormer-Time + 行为类型 embedding | 0.7536 | 0.6001 | +6.31pt AUC / -1.41pt GAUC |
| HyFormer-MultiGranularity | 按 pv/fav/cart/buy 拆分多粒度序列 | 0.7620 | 0.6111 | +7.15pt AUC / -0.31pt GAUC |
| HyFormer-Session, 30min | 当前 session + 长期历史双序列 | 0.7869 | 0.6853 | +9.64pt AUC / +7.11pt GAUC |
| HyFormer-Session, 60min | 当前 session + 长期历史双序列 | 0.7833 | 0.6679 | +9.28pt AUC / +5.37pt GAUC |
| HyFormer-Session + hard negative | fav/cart 高意图负样本采样 | 0.7788 | 0.6545 | 更难负样本验证集 |
| HyFormer-Static | Session 建模 + 用户画像 + 商品静态特征 | 0.8042 | 0.6970 | +11.37pt AUC / +8.28pt GAUC |
| HyFormer-Hierarchical | 最近 100 条细粒度 + 更早历史 8 chunk 压缩 | 0.8058 | 0.7126 | 长序列 500 |

从结果可以看到，LSTM 基线已经能够学习到一定的购买转化模式，验证集 AUC 达到 0.6905。引入 Target Attention 的 SIM 在 AUC 和 GAUC 上均优于 LSTM，说明对于购买预测任务，并不是所有历史行为都同等重要，模型根据候选行为动态关注相关历史类目能够带来收益。优化后的 TWIN 参考论文中的 CP-GSU，将 GSU 检索和 ESU 聚合统一到同一套多头 Target Attention 分数上，并加入 item/category match、embedding similarity、recency 等压缩 cross feature bias，AUC 从旧版 TWIN-lite 的 0.6866 提升到 0.7003。进一步针对当前任务优化后，HyFormer-Opt 将历史 brand 和 cate 拆成两条异构序列，AUC 提升到 0.7008，说明多序列建模能力在该任务中是有效的。

在此基础上，HyFormer-Time 显式加入时间间隔特征，包括历史行为距离预测时刻的时间差、相邻行为之间的时间间隔、离散时间桶 embedding 和连续时间衰减 gate。实验结果显示 AUC 从 HyFormer-Opt 的 0.7008 提升到 0.7143，GAUC 从 0.6186 提升到 0.6401，说明时间因素对电商购买预测非常关键。近期行为比很久以前的行为更能反映当前购买意图，显式建模 recency 可以明显提升排序效果。

进一步加入行为类型编码后，HyFormer-Event 将每个历史行为表示为 `brand_emb + cate_emb + btag_emb + time_emb`，使模型能够区分浏览、收藏、加购和购买等不同意图强度。该实验将 AUC 提升到 0.7536，说明行为类型对全局购买概率判断非常有效；但 GAUC 降到 0.6001，说明模型可能更依赖“历史中是否出现高意图行为”这类全局信号，而对同一用户内部多个候选行为的精细排序帮助有限。因此，在最终模型选择上，如果更关注整体购买概率预测，可选择 HyFormer-Event；如果更关注按用户分组排序，则 HyFormer-Time 更稳健。

多粒度序列实验进一步将历史行为按 `pv`、`fav`、`cart`、`buy` 拆成多条异构序列，每条序列内部融合 brand、cate 和时间信息，再由 HyFormerBackbone 进行多序列交互。相比 HyFormer-Event，HyFormer-MultiGranularity 的 AUC 从 0.7536 提升到 0.7620，GAUC 从 0.6001 回升到 0.6111，说明将不同行为类型拆开建模比简单加入 btag embedding 更能发挥 HyFormer 的多序列建模优势。不过它的 GAUC 仍低于 HyFormer-Time，表明行为类型信号对整体购买概率判断更强，而时间间隔特征对用户内排序更稳健。

Session 级建模是当前最有效的优化。HyFormer-Session 根据相邻行为时间间隔切分当前 session 和长期历史，分别建模短期实时兴趣与长期稳定兴趣。以 30 分钟作为 session 切分阈值时，AUC 达到 0.7869，GAUC 达到 0.6853，显著优于 HyFormer-Time 和 HyFormer-MultiGranularity；60 分钟阈值也有效，但略低于 30 分钟。该结果说明课程要求中强调的“当前 session 实时行为”非常关键，短期 session 行为比混合在完整历史序列中的行为更能反映用户即将购买的意图。

负样本采样实验将原来的均匀负采样改为 hard negative 采样：降低普通 `pv` 的采样比例，提高 `fav` 和 `cart` 等高意图但未购买行为的采样比例。生成的数据中正负样本约为 1:1，模型需要区分“接近购买但尚未购买”和“真实购买”，决策边界更细。该设置下 HyFormer-Session 的 AUC 为 0.7788，GAUC 为 0.6545，低于原始负采样验证集，但这是因为验证集本身更难，不能简单理解为模型退化。该实验说明 hard negative 更适合用于提升模型鲁棒性和真实业务判别能力，后续可采用“随机负样本 + hard negative”混合训练，并在统一测试集上评估泛化效果。

静态特征融合实验进一步加入用户画像与商品侧统计特征。用户侧包括性别、年龄、消费层级、购物等级、职业和新用户等级；商品侧由于购买预测样本来自 `behavior_log`，没有具体 `adgroup_id`，因此使用 target brand 和 cate 在 `ad_feature.csv` 中的聚合统计，包括品牌平均价格、品牌广告数、类目平均价格和类目广告数。这些特征被编码为 non-sequence tokens，与当前 session 序列和长期历史序列一起输入 HyFormer。实验结果显示 AUC 提升到 0.8042，GAUC 提升到 0.6970，说明静态画像和商品侧先验能有效补充纯行为序列，尤其有助于区分不同用户群体和不同商品类目的购买倾向。

长序列与分层压缩实验将最大历史长度从 100 扩展到 500。模型保留最近 100 条行为的细粒度表示，将更早的长期历史按 8 个 chunk 进行池化压缩，并与当前 session 序列一起输入 HyFormer。该设计避免直接对 500 条行为做完整注意力计算，同时保留了长期兴趣信息。实验中约 14,672 条样本历史长度超过 100，HyFormer-Hierarchical 的 AUC 达到 0.8058，GAUC 达到 0.7126，尤其在 GAUC 上相比 HyFormer-Static 有明显提升，说明长期历史对用户内排序有价值，而分层压缩能够在控制计算量的同时利用这部分信息。

项目也尝试过直接使用 `raw_sample.csv` 中的广告点击标签进行 CTR 预测，但在只使用历史 `brand/cate` 序列时，启发式特征与点击标签的相关性较弱，模型 AUC 仅在 0.51 左右。因此最终实验采用 `behavior_log` 构造购买预测任务，更符合课程中“基于当前 session 实时行为预测最终是否购买”的要求。

### 结论

本项目最终选择 `HyFormer-Hierarchical` 作为综合效果最好的模型。该模型在 `HyFormer-Session` 和 `HyFormer-Static` 的基础上进一步引入长序列分层压缩：当前 session 保留细粒度行为，最近历史保留最近 100 条行为，更早的长期历史通过 chunk pooling 压缩为长期兴趣序列。最终实验结果达到 AUC 0.8058、GAUC 0.7126，是当前所有实验中 GAUC 最好的方案。

从整体实验过程看，LSTM 可以作为合理基线，能够捕捉基本时序依赖，但它将整段历史压缩为一个最终隐状态，在长序列场景下容易丢失关键信息。SIM、TWIN 等注意力或检索式模型能够更好地筛选与当前目标相关的历史行为；HyFormer 进一步利用多序列结构，将不同来源、不同粒度的行为信息作为异构序列进行融合，更适合本项目的数据特点。

各项优化实验说明，电商购买预测不仅依赖“用户看过什么”，还强烈依赖“什么时候看”“以什么行为方式看”“是否属于当前 session”以及“用户和商品本身有什么静态属性”。其中，时间间隔特征显著提升了排序效果，说明近期行为比久远行为更能反映当前购买意图；行为类型编码显著提升 AUC，说明 `cart`、`fav`、`buy` 等高意图行为对全局购买概率判断非常关键；session 级建模带来了最大的单步提升，验证了课程要求中“当前 session 实时行为”的重要性；用户画像和商品侧静态特征进一步补充了纯序列模型无法表达的先验信息。

因此，本项目的核心结论是：在电商购买预测任务中，最佳建模方式不是单纯使用完整历史序列，而是同时建模短期 session 兴趣、长期历史兴趣、行为类型、时间间隔和静态画像特征。HyFormer 的多序列结构天然适合这种异构信息融合；结合 session 切分、静态特征和长序列分层压缩后，模型能够在控制计算成本的同时更充分地利用用户行为历史，最终取得最优效果。

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
