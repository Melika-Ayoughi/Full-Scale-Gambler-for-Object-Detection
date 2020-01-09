#!/bin/bash
#SBATCH --gres=gpu:v100-32:2
#SBATCH -c 8
#SBATCH -o logs/train_%j.out

set -x

model_num=$1
python tools/train_net.py \
	--num-gpus 2 --dist-url tcp://0.0.0.0:12345 \
	--config-file "configs/ssigns/retinanet_upsample_${model_num}_1x.yaml" \
	SOLVER.IMS_PER_BATCH 24 SOLVER.BASE_LR 0.0025 MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS '[[0.5, 0.2, 0.125]]'\
	OUTPUT_DIR "output/surface_signs/retinanet_${model_num}_5"

#  --eval_only --resume
# MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS '[[0.25, 0.5, 1]]'