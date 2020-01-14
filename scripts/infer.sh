#!/bin/bash
#SBATCH --gres=gpu:v100-32:1
#SBATCH -c 8
#SBATCH -o logs/train_%j.out

set -x


-python tools/infer.py --config-file configs/ssigns/simple_retinanet_R_50.yaml \
      --input_file image_list_centered_1.txt \
      --output output/surface_signs/simple_retinanet_0/centered_1 \
      --opts MODEL.WEIGHTS output/surface_signs/simple_retinanet_0/model_final.pth \
