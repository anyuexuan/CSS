# Conditional Self-Supervised Learning for Few-Shot Classification

Code for "Conditional Self-Supervised Learning for Few-Shot Classification" in IJCAI 2021.

## Enviroment

Python3

Pytorch

## Getting started

### CIFAR-FS

- Change directory to `./filelists/cifar`
- Download [CIFAR-FS](https://drive.google.com/file/d/1i4atwczSI9NormW5SynaHa1iVN1IaOcs/view)
- run `bash ./cifar.sh`

### CUB

- Change directory to `./filelists/CUB`

- run `bash ./download_CUB.sh`

### mini-ImageNet

- Change directory to `./filelists/miniImagenet`
- Download files for [mini-ImageNet](https://drive.google.com/drive/folders/1x2yFyHyBblfPrR0jQgWlF3c83UHYOEzF?usp=sharing)
- run `bash ./miniImagenet.sh`

## Running

```
python run_css.py
```

