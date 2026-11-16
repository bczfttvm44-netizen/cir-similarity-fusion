# CIR Similarity-Level Fusion

Official implementation of *A Similarity-Level Multi-Model Fusion Method for Composed Image Retrieval*
（《面向组合图像检索的相似度层多模型融合方法》）.
The manuscript is currently under review; citation information will be updated upon acceptance.

## Overview (English)

We fuse several **frozen** composed image retrieval (CIR) baselines at the **similarity level**,
so that no baseline has to be retrained end-to-end.

- **Input.** The similarity vectors produced by four baselines (CLIP4CIR, TG-CIR, BLIP4CIR, SPRC)
  are stacked along the model axis into an `[N, 4, D]` tensor.
- **Fusion head.** A lightweight 1D depthwise-separable convolution (`LightDW`) maps the stack into a
  shared representation space, with a residual path that preserves each baseline's original scores.
- **Dynamic weighting.** A learnable per-dimension weight (`nn.Conv1d(4, 1, 1)`) aggregates the fused
  representation into a single relevance score.
- **Training.** InfoNCE / cross-entropy over the fused score. Only the fusion head is optimized;
  all four baselines remain frozen, so the whole procedure costs minutes rather than GPU-days.

Main results: CIRR Recall@10 **93.21%**, CIRR mean Rmean **84.43%**, FashionIQ mean Rmean **67.48%**
(avg. over Dress / Shirt / Toptee).

```bash
pip install -r requirements.txt   # torch, torchvision, numpy, Pillow, tqdm
bash run.sh                       # train + evaluate, once data and features are in place
```

> **Data and baseline features are not included in this repository.** It ships the code and the two
> trained fusion heads (~10 KB each) only. The CIRR / FashionIQ annotations remain under their own
> licenses, and the ~14 GB of pre-computed baseline features are available from the authors on
> request (caiyw18@chinaunicom.cn). See [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md).
>
> Two practical notes: every command must be run from the repository root (the scripts resolve all
> paths relatively), and the FashionIQ evaluation reads original images, so `fashionIQ_dataset/images/`
> must be present locally. The CIRR evaluation needs annotations only.

---

## 方法概述

本方法在**相似度层**对多个冻结的组合图像检索（CIR）基线做后融合：

- **输入**：CLIP4CIR、TG-CIR、BLIP4CIR、SPRC 四个基线各自输出的相似度向量，沿模型维堆叠为 `[N, 4, D]` 张量。
- **融合网络**：轻量级 1D 深度可分离卷积头（`LightDW`），映射到公共表征空间；以**残差连接**保留各基线的原始打分信息。
- **动态加权**：融合后经可学习的逐维加权（`nn.Conv1d(4, 1, 1)`）汇聚为单一打分。
- **训练目标**：InfoNCE / 交叉熵，仅更新融合网络参数，四个基线全程冻结，无需端到端重训。

主要结果（详见论文）：

| 数据集 | 指标 | 数值 |
|---|---|---|
| CIRR | Recall@10 | 93.21% |
| CIRR | 平均 Rmean | 84.43% |
| FashionIQ | 平均 Rmean | 67.48% |

## 目录结构

```
.
├── stacking_cirr/              # CIRR 分支
│   ├── models.py               # LightDW / LightFusion / CIRPlus
│   ├── data_utils.py           # CIRR & FashionIQ 数据集封装
│   ├── utils.py                # collate_fn / RunningAverage / save_model
│   ├── train.py                # 训练融合头
│   ├── cirr_test_submission.py # 生成 CIRR 测试集提交文件
│   └── model/cirr.pt           # 已训练权重（约 9.5 KB）
├── stacking_fiq/               # FashionIQ 分支
│   ├── models.py
│   ├── data_utils.py
│   ├── utils.py
│   ├── train.py
│   ├── validate.py             # FashionIQ 验证集评测
│   └── model/fiq.pt            # 已训练权重（约 11 KB）
├── docs/DATA_PREPARATION.md    # 数据与特征准备说明
├── run.sh                      # 一键复现脚本
├── requirements.txt
└── LICENSE
```

两个分支结构一致，仅数据集与评测协议不同。

## 环境安装

```bash
pip install -r requirements.txt
```

代码对依赖要求很轻：`torch`、`torchvision`、`numpy`、`Pillow`、`tqdm` 即可。

## ⚠️ 复现前置条件（重要）

本仓库**只包含融合网络与训练/评测代码，不包含数据集与基线特征**，因为二者体积与授权均不适合入库：

| 内容 | 体积 | 是否入库 | 说明 |
|---|---|---|---|
| `cirr_dataset/` | 56 MB | ❌ | CIRR 标注（captions / image_splits） |
| `fashionIQ_dataset/` | 24 MB | ❌ | FashionIQ 标注 |
| `feature_cirr/` | 4.3 GB | ❌ | 四个基线在 CIRR 上的预计算特征 |
| `feature_fiq/` | 9.7 GB | ❌ | 四个基线在 FashionIQ 上的预计算特征 |
| `stacking_*/model/*.pt` | ~10 KB | ✅ | 训练好的融合头权重 |

### 基线特征获取

`feature_cirr/` 与 `feature_fiq/` 中的预计算特征（共约 14 GB）**未随仓库分发**，
如有复现需要，请联系作者获取：

> 联系邮箱：caiyw18@chinaunicom.cn

数据集标注可从 CIRR 与 FashionIQ 官方发布渠道下载；
原始图像同样请遵循各自官方许可获取（FashionIQ 评测需要原图，见下文）。

运行前需按以下路径放置数据（路径相对于仓库根目录，脚本用相对路径读取）：

```
.
├── cirr_dataset/                    # CIRR 标注
├── fashionIQ_dataset/               # FashionIQ 标注
├── feature_cirr/{clip,tgcir,blip,blip2}/{train,test}_feature.pt
└── feature_fiq/{clip,tgcir,blip,blip2}/{train_feature.pt, val_feature_{dress,shirt,toptee}.pt}
```

特征文件为 `torch.save` 保存的可索引序列（每个元素为单张图像的相似度向量），按数据集图像索引对齐。
详细的目录约定与特征格式见 [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md)。

> **注意**：本仓库不包含从原始图像生成这些特征文件的脚本。基线特征需由各自官方实现
> （CLIP4CIR、TG-CIR、BLIP4CIR、SPRC）导出为上述格式后放入对应目录，或向作者索取。

## 训练

所有命令须在**仓库根目录**执行（脚本内部使用相对路径）。

CIRR：

```bash
python stacking_cirr/train.py --dataset cirr --batch-size 512 --learning-rate 2e-3 --tau 0.02 --num-epochs 4
```

FashionIQ：

```bash
python stacking_fiq/train.py --dataset fiq --batch-size 512 --learning-rate 3e-3 --tau 0.02 --num-epochs 4
```

训练完成后权重保存至 `stacking_{cirr,fiq}/model/`。

## 评测

CIRR 测试集（生成提交文件到 `submission/`）：

```bash
python stacking_cirr/cirr_test_submission.py --submission-name stacking --data_path cirr_dataset
```

> CIRR 的 `classic` 模式仅读取标注与索引名，不读取原始图片，因此评测不需要 CIRR 原图。

FashionIQ 验证集：

```bash
python stacking_fiq/validate.py --data_path fashionIQ_dataset
```

> FashionIQ 的 `classic` 模式会读取原始图片（`fashionIQ_dataset/images/`），
> 因此 FashionIQ 评测需要本地存在原图，否则需相应调整 `data_utils.py`。

## 一键复现

数据与特征准备就绪后，训练与评测可依次执行：

```bash
bash run.sh
```

> FashionIQ 部分还需要 `fashionIQ_dataset/images/` 原图。

## 引用

若本工作对你有帮助，请引用：

```bibtex
@misc{cirsimfusion2026,
  title        = {面向组合图像检索的相似度层多模型融合方法},
  author       = {作者列表},
  year         = {2026},
  note         = {Manuscript under review}
}
```

> 论文尚在审稿中，录用后将更新为正式 `@article` 条目（补全期刊、卷期与页码）。
> 在此之前引用上述条目即可，不会产生错误指向。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
