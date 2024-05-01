#!/bin/bash

# MaskFormer 훈련 스크립트 실행
python train_original.py \
    --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
    --num-gpus 2 \
    OUTPUT_DIR OUT_weights \
    MODEL.WEIGHTS model_LR_5_0019999.pth
