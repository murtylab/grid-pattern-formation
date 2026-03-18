CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --eval-name all \
    --results-root results \
    --checkpoint-path checkpoints/replicate_baseline/epoch_000100.pth \
    --config configs/replicate_baseline.yaml