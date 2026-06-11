#!/usr/bin/env bash
# ============================================================
# 一键复现脚本 — 探索性实验全部结果
# 用法: bash scripts/run_all.sh [--quick]
#
# 不传参数 = 完整复现（3 epoch），传 --quick = 快速验证（1 epoch）
# ============================================================
set -euo pipefail

DATA="data/purchase_sequence_100k_static_long500.csv"
EPOCHS=${EPOCHS:-3}
SEQ_LEN=100

if [ "${1:-}" = "--quick" ]; then
    EPOCHS=1
    echo "[INFO] 快速模式: epochs=1"
fi

echo "=============================================="
echo "  探索性实验一键复现"
echo "  数据: $DATA"
echo "  epochs: $EPOCHS"
echo "=============================================="

# 0. 环境检查
echo ""
echo "[0/6] 检查依赖..."
python -c "import torch; print(f'  PyTorch {torch.__version__}')"
python -c "import sklearn; print(f'  scikit-learn {sklearn.__version__}')"
python -c "import torchao; print(f'  torchao {torchao.__version__}')" 2>/dev/null || echo "  torchao not installed (skip INT8 quantization)"

# 1. 原始 TWIN 基线（步骤 1）
echo ""
echo "[1/6] 原始 TWIN 基线..."
python train.py --model twin --data_path "$DATA" --epochs "$EPOCHS" --max_seq_len "$SEQ_LEN"

# 2. TWIN 变体 - 非线性相似度（步骤 2）
echo ""
echo "[2/6] TWIN 非线性相似度..."
python train.py --model twin_nonlinear_sim --data_path "$DATA" --epochs "$EPOCHS" --max_seq_len "$SEQ_LEN"

# 3. TWIN 变体 - 可学习门控（步骤 3）
echo ""
echo "[3/6] TWIN 可学习门控..."
python train.py --model twin_gate_fusion --data_path "$DATA" --epochs "$EPOCHS" --max_seq_len "$SEQ_LEN"

# 4. 参数共享（步骤 16）
echo ""
echo "[4/6] TWIN 参数共享..."
python train.py --model twin_shared_emb --data_path "$DATA" --epochs "$EPOCHS" --max_seq_len "$SEQ_LEN"

# 5. Gumbel-Softmax TopK（步骤 17）
echo ""
echo "[5/6] TWIN Gumbel-Softmax TopK..."
python train.py --model twin_gumbel_topk --data_path "$DATA" --epochs "$EPOCHS" --max_seq_len "$SEQ_LEN"

# 6. 知识蒸馏（Phase 2）
echo ""
echo "[6/6] 知识蒸馏 HyFormer → TWIN..."
python train_distill.py --model twin --teacher hyformer_hierarchical \
    --data_path "$DATA" --epochs "$EPOCHS" --max_seq_len "$SEQ_LEN" \
    --alpha 0.5 --T 2.0

echo ""
echo "=============================================="
echo "  ✅ 全部完成！"
echo "  查看 checkpoints/ 目录获取模型权重"
echo "  查看探索性实验计划.md 获取完整结果记录"
echo "=============================================="
