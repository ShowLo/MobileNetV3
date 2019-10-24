# MobileNetV3
An implementation of MobileNetV3 with pyTorch

# Theory
&emsp;You can find the paper of MobileNetV3 at [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).

# Prepare data

* CIFAR-10
* CIFAR-100
* ImageNet: Please move validation images to labeled subfolders, you can use the script [here](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

# Train

* Train from scratch:

```
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 256
```

* Train from one checkpoint(for example, train from `epoch_33.pth`, the `--start-epoch` parameter is corresponding to the epoch of the checkpoint):

```
CUDA_VISIBLE_DEVICES=3 python train.py --batch-size 256 --resume /MobileNetV3/output/epoch_33.pth --start-epoch 33 --num-epochs 100
```

# Pretrained models

&emsp;TO DO.

# Experiments

## Training setting

### on ImageNet

&emsp;TO DO.

### on CIFAR-10

1. number of epochs: 150 for MobileNetV3-Small and 200 for MobileNetV3-Large
2. learning rate schedule: cosine, minium lr of 1e-5, initial lr=0.7
3. weight decay: 4e-5
4. batch size: 256
5. optimizer: SGD with momentum=0.9

### on CIFAR-100

&emsp;TO DO.

## MobileNetV3-Large

### on ImageNet

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Offical 1.0  | 219 M     | 5.4  M     | 72.0%     |     -     |
| Ours    1.0  | 216.6 M   | 5.47 M     | -         |     -     |

### on CIFAR-10

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  | 8.36 M    | 5.47 M     | -         |     -     |

### on CIFAR-100

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  | 8.36 M    | 5.47 M     | -         |     -     |

## MobileNetV3-Small

### on ImageNet

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Offical 1.0  | 66 M      | 2.9  M     | 67.4%     |     -     |
| Ours    1.0  | 56.51 M   | 2.53 M     | -         |     -     |

### on CIFAR-10

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  | 3.18 M    | 2.53 M     | -         |     -     |

### on CIFAR-100

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  | 3.18 M    | 2.53 M     | -         |     -     |

## Dependency

&emsp;This project uses Python 3.7 and PyTorch 1.1.0. The FLOPs and Parameters and measured using [torchsummaryX](https://github.com/nmhkahn/torchsummaryX).
