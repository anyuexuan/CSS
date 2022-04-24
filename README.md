# Conditional Self-Supervised Learning for Few-Shot Classification

Code for "Conditional Self-Supervised Learning for Few-Shot Classification" in IJCAI 2021.

If you use the code in this repo for your work, please cite the following bib entries:

```
@inproceedings{An2021CSS,
  author    = {Yuexuan An and
               Hui Xue and
               Xingyu Zhao and
               Lu Zhang},
  title     = {Conditional Self-Supervised Learning for Few-Shot Classification},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI} 2021, Virtual Event / Montreal, Canada, 19-27 August 2021},
  pages     = {2140--2146},
  year      = {2021},
}
```

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
- Download [mini-ImageNet](https://drive.google.com/file/d/1hQqDL16HTWv9Jz15SwYh3qq1E4F72UDC/view)
- run `bash ./miniImagenet.sh`

## Running

```
python run_css.py
```

## Acknowledgment

Our project references the codes and datasets in the following repo and papers.

[CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltechucsd birds-200-2011 dataset. 2011.

Luca Bertinetto, João F. Henriques, Philip H. S. Torr, Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. ICLR 2019.

Oriol Vinyals, Charles Blundell, Tim Lillicrap, Koray Kavukcuoglu, Daan Wierstra. Matching Networks for One Shot Learning. NIPS 2016: 3630-3638.

