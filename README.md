# weight-transfer

This repository contains utilities for **converting Keras models to PyTorch, converting TF models to Keras, and converting TF models to PyTorch**.

This repository works well with CUDA 10.2, PyTorch 1.9.0, and Tensorflow 2.1.0.

**Be careful to verify the results or use an alternative method to convert, as this repository can not guaranteed to produce exactly the same predictions or outputs for other versions of Tensorflow, Keras and PyTorch.**

## Prerequisites
- Linux or macOS
- Python 3
- Anaconda
- PyTorch 1.9.0
- Tensorflow 2.1.0

You need to have Anaconda, PyTorch, and Tensorflow installed beforehand, see the [Anaconda website](https://www.anaconda.com), [PyTorch website](https://www.pytorch.org), and the [Tensorflow website](https://tensorflow.google.cn) for how to easily install that.

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/Kira-Z-China/weight-transfer
cd weight-transfer
```
- Please type the command：
```
pip install -r requirements.txt
```



## Run

To load weights from Keras model to Pytorch model, run :

```
python load_from_keras_to_torch.py --sourcefile='weight_keras/weights_300.hdf5'
```

To convert weights from Tensorflow model to Keras model, run :

```
python transfer_from_tf_to_keras.py --infile='weight_tf/wgts_epochs_10000.ckpt' --outfile='weight_out_keras/wgts_epochs_10000.hdf5'
```
To load weights from Tensorflow model to Pytorch model, run :

```
python transfer_from_tf_to_keras.py --infile='weight_tf/wgts_epochs_10000.ckpt' --outfile='weight_out_keras/wgts_epochs_10000.hdf5'

python load_from_keras_to_torch.py --sourcefile='weight_out_keras/wgts_epochs_10000.hdf5'
```


