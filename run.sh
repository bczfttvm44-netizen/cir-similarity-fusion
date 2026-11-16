#!/usr/bin/env bash
# 复现脚本：训练与评测
# 用法：在仓库根目录执行  bash run.sh
# 说明：所有路径均为相对路径，必须在仓库根目录下运行。

set -e

# ---------- CIRR ----------
echo "=== [1/4] CIRR 训练 ==="
python stacking_cirr/train.py --dataset cirr --batch-size 512 --learning-rate 2e-3 --tau 0.02 --num-epochs 4

echo "=== [2/4] CIRR 测试集提交 ==="
python stacking_cirr/cirr_test_submission.py --submission-name stacking --data_path cirr_dataset

# ---------- FashionIQ ----------
echo "=== [3/4] FashionIQ 训练 ==="
python stacking_fiq/train.py --dataset fiq --batch-size 512 --learning-rate 3e-3 --tau 0.02 --num-epochs 4

echo "=== [4/4] FashionIQ 验证集评测 ==="
python stacking_fiq/validate.py --data_path fashionIQ_dataset

echo "全部流程完成。"
