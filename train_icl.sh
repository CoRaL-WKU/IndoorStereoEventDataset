#!/bin/bash

# to finetune MaskFormer
# python train_net.py \
# 	--config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
# 	--num-gpus 2 \
# 	OUTPUT_DIR OUT_weights \
# 	MODEL.WEIGHTS model_LR_5_0019999.pth

# to ICL MaskFormer from evaluation script
python train_net_video.py \
	--config-file configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml \
	--num-gpus 2 \
	--eval-only \
	OUTPUT_DIR output/youtubevis_2019 \
	MODEL.WEIGHTS model_final_c5c739.pkl
