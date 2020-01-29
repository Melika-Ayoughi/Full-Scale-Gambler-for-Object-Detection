Example script to do training and inference for surface sign:


## Make inference given a list of folders:

```
python tools/infer.py --config-file configs/ssigns/simple_retinanet_R_50.yaml --plot_output\
      --input_file centered_640.txt \
      --output output/surface_signs/simple_retinanet_0/centered_640 \
      --confidence_threshold 0.001 \
      --opts MODEL.WEIGHTS output/surface_signs/simple_retinanet_0/model_final.pth \
```

## Train a simple Retinanet:
```
python tools/train_net.py \
	--num-gpus 1 --dist-url tcp://0.0.0.0:12345 \
	--config-file "configs/ssigns/simple_retinanet_R_50.yaml" \
	SOLVER.IMS_PER_BATCH 12 SOLVER.BASE_LR 0.0025 \
	OUTPUT_DIR "output/surface_signs/simple_retinanet_0"
```

## Train a normal Retinanet with upsampling the input image two times:
```
python tools/train_net.py \
	--num-gpus 1 \
	--config-file "configs/ssigns/retinanet_upsample_class_agnostic_1x.yaml" \
	SOLVER.IMS_PER_BATCH 12 SOLVER.BASE_LR 0.0025 \
	OUTPUT_DIR "output/surface_signs/retinanet_upsample_0"
```


## Train a Faster-RCNN models:
```
python tools/train_net.py \
	--num-gpus 2 --dist-url tcp://0.0.0.0:15215 \
	--config-file "configs/ssigns/faster_rcnn_R_50_upsample_FPN_${model_num}_1x.yaml" \
	SOLVER.IMS_PER_BATCH 24 SOLVER.BASE_LR 0.0025 MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS '[[0.5, 0.2, 0.125]]'\
	OUTPUT_DIR "output/surface_signs/faster_rcnn_0"
```
