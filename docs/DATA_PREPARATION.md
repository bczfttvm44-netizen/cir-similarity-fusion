# 数据与特征准备

本仓库的代码只消费**已经算好的基线特征**，不负责从原始图像生成特征。
本文件说明目录约定、标注文件格式、以及特征文件必须满足的对齐契约。

所有路径均相对于仓库根目录。

---

## 1. 目录总览

```
cir-similarity-fusion/
├── cirr_dataset/
│   ├── cirr/
│   │   ├── captions/
│   │   │   ├── cap.rc2.train.json
│   │   │   ├── cap.rc2.val.json
│   │   │   └── cap.rc2.test1.json
│   │   └── image_splits/
│   │       ├── split.rc2.train.json
│   │       ├── split.rc2.val.json
│   │       └── split.rc2.test1.json
│   └── optimized_images.json
│
├── fashionIQ_dataset/
│   ├── captions/
│   │   ├── cap.dress.train.json / cap.dress.val.json
│   │   ├── cap.shirt.train.json / cap.shirt.val.json
│   │   └── cap.toptee.train.json / cap.toptee.val.json
│   ├── image_splits/
│   │   ├── split.dress.train.json / split.dress.val.json
│   │   ├── split.shirt.train.json / ...
│   │   └── split.toptee.train.json / ...
│   ├── images/                 # 原图（仅 FashionIQ 评测需要）
│   └── optimized_images.json
│
├── feature_cirr/
│   ├── clip/   train_feature.pt, test_feature.pt
│   ├── tgcir/  train_feature.pt, test_feature.pt
│   ├── blip/   train_feature.pt, test_feature.pt
│   └── blip2/  train_feature.pt, test_feature.pt
│
└── feature_fiq/
    ├── clip/   train_feature.pt, val_feature_dress.pt,
    │           val_feature_shirt.pt, val_feature_toptee.pt
    ├── tgcir/  （同上）
    ├── blip/   （同上）
    └── blip2/  （同上）
```

---

## 2. 标注文件

### CIRR

- `cap.rc2.{split}.json`：三元组列表，每条包含 `reference`、`target_hard`、`caption`、`pairid`、`img_set.members` 等字段。
- `split.rc2.{split}.json`：图像名 → 相对路径的映射（`name_to_relpath`）。
- `split` 取值：`train` / `val` / `test1`。

### FashionIQ

- `cap.{dress_type}.{split}.json`：每条包含 `candidate`（参考图名）、`target`（目标图名）、`captions`（两条修改文本）。
- `split.{dress_type}.{split}.json`：该类别下的图像名列表。
- `dress_type` 取值：`dress` / `shirt` / `toptee`；`split` 取值：`train` / `val`。

### optimized_images.json（关键）

两个数据集各有一份，是**固定的图像索引表**，用于保证「数据集图像顺序」与「预计算特征顺序」严格一致。格式为长度 3 的列表：

```json
[
  ["图像名1", "图像名2", ...],        // imagename 列表
  ["路径1", "路径2", ...],            // 与上一项一一对应的路径
  {"图像名1": 0, "图像名2": 1, ...}   // 图像名 → 全局索引
]
```

当前规模：CIRR 16536 张、FashionIQ 24779 张。

训练时若该文件存在，`data_utils.py` 会用它**覆盖**按三元组动态构建的图像顺序。
因此**特征文件的第 i 个元素必须对应 `imagename2id` 中索引为 i 的图像**，否则训练和评测都会静默错位。

---

## 3. 特征文件格式

每个 `.pt` 由 `torch.save` 保存，载入后应是一个**可按下标访问的序列**（list 或 1D 外层 tensor），
每个元素是**单张图像的特征向量**（float32 tensor）。

代码中的消费方式：

```python
# 训练：按图像全局索引取特征
clip_fea = self.clip_feature[refer_indexs]     # -> [B, D]

# 评测：按 pair/样本顺序取特征
clip_fea = itemgetter(*pairs_id_int)(torch.load('feature_cirr/clip/test_feature.pt'))
clip_fea = torch.stack(clip_fea, dim=0)        # -> [N, D]

# 四个基线沿模型维堆叠
query_features = torch.stack([clip_fea, tgcir_fea, blip_fea, blip2_fea], dim=1)  # -> [N, 4, D]
```

因此四个基线的特征长度必须一致，且与对应 split 的样本数对齐。

### 对齐要求汇总

| 文件 | 长度应对齐 |
|---|---|
| `feature_cirr/*/train_feature.pt` | CIRR train 的全局图像数（`optimized_images.json` 长度） |
| `feature_cirr/*/test_feature.pt` | CIRR test1 的查询数（按 `pairid` 顺序） |
| `feature_fiq/*/train_feature.pt` | FashionIQ train 的全局图像数 |
| `feature_fiq/*/val_feature_{type}.pt` | 该类别 val 的样本数 |

---

## 4. 如何获得特征

这些特征文件（约 14 GB）**未随仓库分发**，如有复现需要请联系作者获取。

若需自行生成，可分别使用四个基线的官方实现，在对应数据集上导出相似度/特征向量，并转换为上述格式：

| 目录名 | 对应基线 |
|---|---|
| `clip` | CLIP4CIR |
| `tgcir` | TG-CIR |
| `blip` | BLIP4CIR |
| `blip2` | SPRC |

导出时务必保持与 `optimized_images.json` 相同的图像顺序（训练特征），
以及与 `pairid` / val 样本顺序一致（评测特征）。

---

## 5. 原图依赖（易踩坑）

| 阶段 | 是否读原图 | 说明 |
|---|---|---|
| CIRR 训练 | ❌ | `use_bank=True`，只取特征索引 |
| CIRR 评测 | ❌ | `classic` 模式对 CIRR 直接返回空列表，不 `PIL.Image.open` |
| FashionIQ 训练 | ❌ | 同上，`use_bank=True` |
| FashionIQ 评测 | ✅ | `classic` 模式默认 `fiq_val_type=0`，会 `PIL.Image.open(fashionIQ_dataset/images/xxx.png)` |

所以 FashionIQ 评测需要本地存在 `fashionIQ_dataset/images/` 原图。
若不想准备原图，可改 `stacking_fiq/data_utils.py` 中 `classic` 分支的 FIQ 逻辑，
使 `extract_index_features` 只返回 `image_names`（该函数本身并不使用图像内容）。
