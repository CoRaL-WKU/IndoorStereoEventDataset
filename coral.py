import os
import torch

from pathlib import Path
import numpy as np
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_panoptic
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
import matplotlib.pyplot as plt


def process_images_in_directory(input_directory, output_directory):
    # Check if the output directory exists, if not, create it
    os.makedirs(output_directory, exist_ok=True)

    # List all PNG files in the input directory
    png_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    # Set up detectron2 configuration
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("/home/coraldl/meta/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep_1.yaml")
    cfg.MODEL.WEIGHTS = "/home/coraldl/meta/Mask2Former/model_LR_5_0019999.pth"
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.MODEL.DEVICE = "cuda"
    coco_metadata = MetadataCatalog.get("coco_2017_train_panoptic")

    predictor = DefaultPredictor(cfg)

    for png_file in png_files:
        # Build the full path for each PNG file
        image_path = os.path.join(input_directory, png_file)

        # Process and visualize the image
        im = cv2.imread(str(image_path))
        predictor.device = torch.device("cuda")
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
                                              outputs["panoptic_seg"][1]).get_image()
        #v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
       # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
      #  v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
      #  semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        result = panoptic_result[:, :, ::-1]

        # Save the result image to the output directory
        result_filename = f"{png_file}"
        result_image_path = os.path.join(output_directory, result_filename)
        cv2.imwrite(result_image_path, result)

        # Visualize using Matplotlib
        print(f"Processed and saved: {image_path} -> {result_image_path}")

# Example usage
input_directory = "/home/coraldl/meta/Mask2Former/datasets/Coral_Indoor_Datasets/ALL_Image_webcam"
output_directory = "/home/coraldl/meta/Mask2Former/datasets/outputs"
process_images_in_directory(input_directory, output_directory)

