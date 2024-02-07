Official Tensorflow implementation of **TPGU**, from the following paper:<br />Augmenting Interaction Effects in Convolutional Networks with Taylor Polynomial Gated Units. 2024<br />Ligeng Zou, Qi Liu, and Jianhua Dai
:::info
We propose a new activation function dubbed TPGU (Taylor Polynomial Gated Unit), which are inspired by Taylor polynomials of the sigmoid function. In addition to being able to learn the strength of each order of interactions, our proposed TPGU activation function does not require any regularization or normalization of the inputs and outputs, nor any special constraints on the polynomial parameters.
:::
<a name="hpJ9c"></a>
## Model Zoo
ImageNet-1K results on TPU:

| Name | Image Size | FLOPs (G) | Params (M) | Top-1 Acc. (%) | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| ConvNeXt-T * | 2242 | 4.48 | 28.59 | 80.5 |  |
| + TPGU{1,1,1,1} | 2242 | 4.49 | 28.66 | 80.7 |  |
| + TPGU{2,2,2,2} | 2242 | 4.50 | 28.69 | 80.8 |  |
| + TPGU{3,3,3,3} | 2242 | 4.50 | 28.72 | 80.9 |  |
| + TPGU{4,4,4,4} | 2242 | 4.51 | 28.74 | 80.9 |  |
| + TPGU{5,5,5,5} | 2242 | 4.52 | 28.77 | 81.1 |  |
| + TPGU{6,6,6,6} | 2242 | 4.53 | 28.80 | 81.1 |  |
| + TPGU{7,7,7,7} | 2242 | 4.54 | 28.82 | **81.2** |  |
| + TPGU{8,8,8,8} | 2242 | 4.55 | 28.85 | 81.1 |  |
| + TPGU{1,2,3,4} | 2242 | 4.50 | 28.72 | 80.8 |  |
| + TPGU{2,3,4,5} | 2242 | 4.50 | 28.75 | 80.8 |  |
| + TPGU{3,4,5,6} | 2242 | 4.51 | 28.77 | 80.9 |  |
| + TPGU{4,5,6,7} | 2242 | 4.52 | 28.80 | 81.0 |  |
| + TPGU{5,6,7,8} | 2242 | 4.53 | 28.83 | 81.2 |  |
| + TPGU{6,7,8,9} | 2242 | 4.54 | 28.86 | 81.2 |  |
| + TPGU{7,8,9,10} | 2242 | 4.55 | 28.89 | 81.1 |  |
| ConvNeXt-S * | 2242 | 8.73 | 50.22 | 82.2 |  |
| + TPGU{7,7,7,7} | 2242 | 8.82 | 50.71 | 82.5 |  |

| Name | Image Size | FLOPs (G) | Params (M) | Top-1 Acc. (%) | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| HorNet-T * | 2242 | 4.01 | 22.41 | 81.4 |  |
| + TPGU{1,1,1,1} | 2242 | 4.01 | 22.48 | 81.2 |  |
| + TPGU{2,2,2,2} | 2242 | 4.01 | 22.50 | 81.3 |  |
| + TPGU{3,3,3,3} | 2242 | 4.02 | 22.53 | **81.7** |  |
| + TPGU{4,4,4,4} | 2242 | 4.03 | 22.55 | 81.4 |  |
| + TPGU{1,2,3,4} | 2242 | 4.02 | 22.53 | 81.5 |  |
| + TPGU{2,3,4,5} | 2242 | 4.02 | 22.55 | 81.5 |  |
| + TPGU{3,4,5,6} | 2242 | 4.03 | 22.58 | 81.5 |  |
| HorNet-S * | 2242 | 8.87 | 49.52 | 82.2 |  |
| + TPGU{3,3,3,3} | 2242 | 8.89 | 49.71 | 82.4 |  |

| Name | Image Size | FLOPs (G) | Params (M) | Top-1 Acc. (%) | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| Swin-T * | 2242 | 4.51 | 28.53 | 80.3 |  |
| + TPGU{1,1,1,1} | 2242 | 4.52 | 28.59 | 79.5 |  |
| + TPGU{2,2,2,2} | 2242 | 4.53 | 28.60 | 80.0 |  |
| + TPGU{3,3,3,3} | 2242 | 4.54 | 28.62 | **80.6** |  |
| + TPGU{4,4,4,4} | 2242 | 4.54 | 28.64 | 80.5 |  |
| + TPGU{1,2,3,4} | 2242 | 4.53 | 28.62 | 80.5 |  |
| + TPGU{2,3,4,5} | 2242 | 4.54 | 28.64 | 80.4 |  |
| Swin-S * | 2242 | 8.78 | 49.94 | 82.1 |  |
| + TPGU{3,3,3,3} | 2242 | 8.82 | 50.12 | 82.2 |  |

<a name="UKHLY"></a>
## Requirements

- tensorflow==2.12.0
- tf-models-official==2.12.0
<a name="Lw5ab"></a>
## **Data preparation**
If you are new to ImageNet, you can follow the tutorial here: [https://cloud.google.com/tpu/docs/imagenet-setup](https://cloud.google.com/tpu/docs/imagenet-setup)<br />If you already have downloaded the ImageNet dataset and organize the directory structure like this:
```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
then you can run the following command using the imagenet_to_gcs.py (modified from the [original script](https://raw.githubusercontent.com/tensorflow/tpu/master/tools/datasets/imagenet_to_gcs.py)) to convet imagenet to TFRecords:
```bash
python imagenet_to_gcs.py --raw_data_dir="path_to_image_net" \
--local_scratch_dir="path_to_save_processed_dataset" \
--gcs_upload=False
```
<a name="Or8Js"></a>
## Training
Before you train any model, remember to check two places:

1. "input_path" in the config file (yaml format).
2. "orders" or "polynomial_orders" for the polynomial in TPGU in network_model.py.

To train a model on ImageNet from scratch on a TPU VM, run:
```bash
python3 run.py \
--experiment=tf_vision_example_experiment \
--mode=train_and_eval \
--seed=0 \
--tpu=local \
--model_dir=path_to_save_ckpt \
--log_dir=path_to_save_logs \
--config_file=path_to_yaml_config_file \
```
If you want to train on GPUs, you could only modify the "distribution_strategy" value in the config file (we don't have GPUs to test it though).
<a name="JTltm"></a>
## Acknowledgements
Our code is based on [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [Hornet](https://github.com/raoyongming/HorNet), [Swin Transformer](https://github.com/rishigami/Swin-Transformer-TF) and [TF Vision Example Project](https://github.com/tensorflow/models/tree/master/official/vision/examples/starter).  We would like to thank [Google's TPU Research Cloud (TRC)](https://sites.research.google/trc) for their generous support of Cloud TPUs used in this project.
