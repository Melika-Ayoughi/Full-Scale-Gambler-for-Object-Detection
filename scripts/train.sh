#!/bin/bash
#SBATCH --gres=gpu:v100-32:1
#SBATCH -c 8
#SBATCH -o logs/train_%j.out

set -x

model_num=$1
python tools/train_net.py \
	--num-gpus 1 --dist-url tcp://0.0.0.0:12345 \
	--config-file "configs/ssigns/simple_retinanet_R_50.yaml" \
	SOLVER.IMS_PER_BATCH 12 SOLVER.BASE_LR 0.0025 \
	OUTPUT_DIR "output/surface_signs/simple_retinanet_${model_num}"

#  --eval_only --resume
# MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS '[[0.5, 0.2, 0.125]]'
#	OUTPUT_DIR "output/surface_signs/retinanet_${model_num}_5"