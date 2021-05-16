import os

data_dir = dict(
    CUB=os.path.dirname(__file__) + '/../filelists/CUB/',
    miniImagenet=os.path.dirname(__file__) + '/../filelists/miniImagenet/',
    omniglot=os.path.dirname(__file__) + '/../filelists/omniglot/',
    emnist=os.path.dirname(__file__) + '/../filelists/emnist/',
    cifar=os.path.dirname(__file__) + '/../filelists/cifar/',
    fc100=os.path.dirname(__file__) + '/../filelists/fc100/',
)


num_workers = 4
test_iter_num = 600


def get_stop_epoch(algorithm, dataset, n_shot=5):
    if algorithm in ['baseline', 'baseline++']:
        if dataset in ['omniglot', 'cross_char']:
            stop_epoch = 5
        elif dataset in ['CUB']:
            stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
        elif dataset in ['miniImagenet', 'cross']:
            stop_epoch = 400
        else:
            stop_epoch = 400  # default
    else:  # meta-learning methods
        stop_epoch = 400
    return stop_epoch