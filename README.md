Official Tensorflow implementation of **TPGU**, from the following paper:<br />Augmenting Interaction Effects in Convolutional Networks with Taylor Polynomial Gated Units. 2024<br />Ligeng Zou, Qi Liu, and Jianhua Dai

We propose a new activation function dubbed TPGU (Taylor Polynomial Gated Unit), which are inspired by Taylor polynomials of the sigmoid function. In addition to being able to learn the strength of each order of interactions, our proposed TPGU activation function does not require any regularization or normalization of the inputs and outputs, nor any special constraints on the polynomial parameters.

<a name="hpJ9c"></a>
## Model Zoo
ImageNet-1K results on TPU:

| Name | Image Size | FLOPs (G) | Params (M) | Top-1 Acc. (%) | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| ConvNeXt-T * | 224*224 | 4.48 | 28.59 | 80.5 | [Baidu Cloud](https://pan.baidu.com/s/1zQLaqMFKE_we4OTportnzg?pwd=9z6a) |
| + TPGU{1,1,1,1} | 224*224 | 4.49 | 28.66 | 80.7 | [Baidu Cloud](https://pan.baidu.com/s/11SretQpP0daSLqyI0bjxuQ?pwd=ol89) |
| + TPGU{2,2,2,2} | 224*224 | 4.50 | 28.69 | 80.8 | [Baidu Cloud](https://pan.baidu.com/s/1zO74X7aRLvxu9x9YcPMA8g?pwd=jw98) |
| + TPGU{3,3,3,3} | 224*224 | 4.50 | 28.72 | 80.9 | [Baidu Cloud](https://pan.baidu.com/s/1NTDZSNIfeyX1PjhtbEk0QQ?pwd=0xpm) |
| + TPGU{4,4,4,4} | 224*224 | 4.51 | 28.74 | 80.9 | [Baidu Cloud](https://pan.baidu.com/s/1SOiHAh2BZod-eoKz8wAKeg?pwd=zxwd) |
| + TPGU{5,5,5,5} | 224*224 | 4.52 | 28.77 | 81.1 | [Baidu Cloud](https://pan.baidu.com/s/1xzDAA6w3Vl9U7JNZVoQcGA?pwd=u5p9) |
| + TPGU{6,6,6,6} | 224*224 | 4.53 | 28.80 | 81.1 | [Baidu Cloud](https://pan.baidu.com/s/1APF5_NZglk4atStooBYrNQ?pwd=m7qv) |
| + TPGU{7,7,7,7} | 224*224 | 4.54 | 28.82 | **81.2** | [Baidu Cloud](https://pan.baidu.com/s/1RBBdoYpY6CtGi-maGmKKSQ?pwd=0y94) |
| + TPGU{8,8,8,8} | 224*224 | 4.55 | 28.85 | 81.1 | [Baidu Cloud](https://pan.baidu.com/s/13RwroA4yiNRDTEZJfhYwvQ?pwd=3hoq) |
| + TPGU{1,2,3,4} | 224*224 | 4.50 | 28.72 | 80.8 | [Baidu Cloud](https://pan.baidu.com/s/1qYKeQTFQGDb5KL2MnIUcjA?pwd=ht0c) |
| + TPGU{2,3,4,5} | 224*224 | 4.50 | 28.75 | 80.8 | [Baidu Cloud](https://pan.baidu.com/s/1rYILx06XBVAAkSmNgRFEpA?pwd=f4yu) |
| + TPGU{3,4,5,6} | 224*224 | 4.51 | 28.77 | 80.9 | [Baidu Cloud](https://pan.baidu.com/s/10iG6unMzPcwMnfZnuYVeoA?pwd=ixny) |
| + TPGU{4,5,6,7} | 224*224 | 4.52 | 28.80 | 81.0 | [Baidu Cloud](https://pan.baidu.com/s/1QeYPs3N8YWBp-3x1gjZSNg?pwd=qvrk) |
| + TPGU{5,6,7,8} | 224*224 | 4.53 | 28.83 | 81.2 | [Baidu Cloud](https://pan.baidu.com/s/1hpm10DYTd_IXd3-legw77g?pwd=0gmv) |
| + TPGU{6,7,8,9} | 224*224 | 4.54 | 28.86 | 81.2 | [Baidu Cloud](https://pan.baidu.com/s/1veSqryJ57AJidd40DqPbiA?pwd=7vdb) |
| + TPGU{7,8,9,10} | 224*224 | 4.55 | 28.89 | 81.1 | [Baidu Cloud](https://pan.baidu.com/s/1YspMDZASP4TRZoFzT8QF1g?pwd=5pd6) |
| ConvNeXt-S * | 224*224 | 8.73 | 50.22 | 82.2 | [Baidu Cloud](https://pan.baidu.com/s/1ZLUe3SXzJpsjWADmOeNWyw?pwd=q4v7) |
| + TPGU{7,7,7,7} | 224*224 | 8.82 | 50.71 | 82.5 | [Baidu Cloud](https://pan.baidu.com/s/1Kjtpe14t-DeIBYJAwSl2oQ?pwd=k8gw) |

| Name | Image Size | FLOPs (G) | Params (M) | Top-1 Acc. (%) | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| HorNet-T * | 224*224 | 4.01 | 22.41 | 81.4 | [Baidu Cloud](https://pan.baidu.com/s/1qjuTWKUsN6iPm8FvYj9-YQ?pwd=zn22) |
| + TPGU{1,1,1,1} | 224*224 | 4.01 | 22.48 | 81.2 | [Baidu Cloud](https://pan.baidu.com/s/1OaxQhrdpuwxucShpkRvt3w?pwd=9m1q) |
| + TPGU{2,2,2,2} | 224*224 | 4.01 | 22.50 | 81.3 | [Baidu Cloud](https://pan.baidu.com/s/1tPUvugaZHQxEFJ5YEOCAHQ?pwd=go2d) |
| + TPGU{3,3,3,3} | 224*224 | 4.02 | 22.53 | **81.7** | [Baidu Cloud](https://pan.baidu.com/s/17M1php5aEK9ueG6QPEEqpA?pwd=a9vo) |
| + TPGU{4,4,4,4} | 224*224 | 4.03 | 22.55 | 81.4 | [Baidu Cloud](https://pan.baidu.com/s/1iE6RHFQ9lzIvMUAj3fauSw?pwd=vxy2) |
| + TPGU{1,2,3,4} | 224*224 | 4.02 | 22.53 | 81.5 | [Baidu Cloud](https://pan.baidu.com/s/1kV_DhJ763dTbP_uM6d3Ydg?pwd=aoam) |
| + TPGU{2,3,4,5} | 224*224 | 4.02 | 22.55 | 81.5 | [Baidu Cloud](https://pan.baidu.com/s/1R0GTijgusHuO2Xfn_E3X2A?pwd=l5an) |
| + TPGU{3,4,5,6} | 224*224 | 4.03 | 22.58 | 81.5 | [Baidu Cloud](https://pan.baidu.com/s/16-d64por5dQHtJnfc8ieiQ?pwd=9nq9) |
| HorNet-S * | 224*224 | 8.87 | 49.52 | 82.2 | [Baidu Cloud](https://pan.baidu.com/s/1T6a02IucQt2Hz_dU20Rk9Q?pwd=4an1) |
| + TPGU{3,3,3,3} | 224*224 | 8.89 | 49.71 | 82.4 | [Baidu Cloud](https://pan.baidu.com/s/1F1zwzax4_YJZhuFhnWj2Gg?pwd=9gjw) |

| Name | Image Size | FLOPs (G) | Params (M) | Top-1 Acc. (%) | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| Swin-T * | 224*224 | 4.51 | 28.53 | 80.3 | [Baidu Cloud](https://pan.baidu.com/s/1qMNBCVehoEdbICwz4F-vYA?pwd=ucrz) |
| + TPGU{1,1,1,1} | 224*224 | 4.52 | 28.59 | 79.5 | [Baidu Cloud](https://pan.baidu.com/s/1qQnKCwko01MH6SXGmcrw0Q?pwd=vsid) |
| + TPGU{2,2,2,2} | 224*224 | 4.53 | 28.60 | 80.0 | [Baidu Cloud](https://pan.baidu.com/s/1P4gnqpk3PSIT0xIKx4-Yfw?pwd=nzdt) |
| + TPGU{3,3,3,3} | 224*224 | 4.54 | 28.62 | **80.6** | [Baidu Cloud](https://pan.baidu.com/s/1-RiEb3Y2mPCgilDolPNLNA?pwd=06zj) |
| + TPGU{4,4,4,4} | 224*224 | 4.54 | 28.64 | 80.5 | [Baidu Cloud](https://pan.baidu.com/s/1vxVpqYSq_FfwyFsYrlcBjA?pwd=loyz) |
| + TPGU{1,2,3,4} | 224*224 | 4.53 | 28.62 | 80.5 | [Baidu Cloud](https://pan.baidu.com/s/1z2TsF6XBnkXJA8_DBYQSxw?pwd=9qlj) |
| + TPGU{2,3,4,5} | 224*224 | 4.54 | 28.64 | 80.4 | [Baidu Cloud](https://pan.baidu.com/s/1QcTKhjuHzKOEKfTR-WFD7Q?pwd=cfpv) |
| Swin-S * | 224*224 | 8.78 | 49.94 | 82.1 | [Baidu Cloud](https://pan.baidu.com/s/1yqS5mAQtwKwQrazEURX8Gw?pwd=tspo) |
| + TPGU{3,3,3,3} | 224*224 | 8.82 | 50.12 | 82.2 | [Baidu Cloud](https://pan.baidu.com/s/1MCQQn8vh8sUqvlMJxeQSBw?pwd=d2e4) |

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
