import torch
import numpy as np
from . import backbone
from .config import *
import os
import random
import glob
from data.datamgr import SimpleDataManager, SetDataManager
import h5py

# region common parameters

base_path = os.path.dirname(__file__).replace('\\', '/') + '/..'

model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101
)

start_epoch = 0  # Starting epoch
save_freq = 50  # Save frequency
train_n_way = 5  # class num to classify for training
test_n_way = 5  # class num to classify for testing (validation)
adaptation = False
noise_rate = 0.

if torch.cuda.is_available():
    use_cuda = True
    print('GPU detected, running with GPU!')
else:
    print('GPU not detected, running with CPU!')
    use_cuda = False


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 0
set_seed(seed)


# endregion

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1).long(), 1)


def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num = len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append(np.mean(cl_data_file[cl], axis=0))
        stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

    mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
    mu_j = np.transpose(mu_i, (1, 0, 2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

    for i in range(cl_num):
        DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
    return np.mean(DBs)


def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x != 0) for x in cl_data_file[cl]]))
    return np.mean(cl_sparsity)


def get_image_size(model_name, dataset):
    if 'Conv' in model_name:
        if dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    return image_size


def get_train_files(dataset):
    if dataset == 'cross':
        base_file = data_dir['miniImagenet'] + 'all.json'
        val_file = data_dir['CUB'] + 'val.json'
    elif dataset == 'cross_char':
        base_file = data_dir['omniglot'] + 'noLatin.json'
        val_file = data_dir['emnist'] + 'val.json'
    else:
        base_file = data_dir[dataset] + 'base.json'
        val_file = data_dir[dataset] + 'val.json'
    return base_file, val_file


def get_train_loader(algorithm, image_size, base_file, val_file, train_n_way, test_n_way, n_shot, noise_rate=0.,
                     val_noise=True, num_workers=4):
    if algorithm in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=16)
        base_loader = base_datamgr.get_data_loader(base_file, aug=True)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    else:
        n_query = max(1, int(
            16 * test_n_way / train_n_way))  # if test_n_way <train_n_way, reduce n_query to keep batch size small
        base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=train_n_way, n_support=n_shot,
                                      noise_rate=noise_rate, num_workers=num_workers)  # n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=True)
        if val_noise:
            val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=test_n_way, n_support=n_shot,
                                         noise_rate=noise_rate, num_workers=num_workers)
        else:
            val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=test_n_way, n_support=n_shot,
                                         noise_rate=0., num_workers=num_workers)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor
    return base_loader, val_loader


def get_novel_file(dataset, split='novel'):
    if dataset == 'cross':
        if split == 'base':
            loadfile = data_dir['miniImagenet'] + 'all.json'
        else:
            loadfile = data_dir['CUB'] + split + '.json'
    elif dataset == 'cross_char':
        if split == 'base':
            loadfile = data_dir['omniglot'] + 'noLatin.json'
        else:
            loadfile = data_dir['emnist'] + split + '.json'
    else:
        loadfile = data_dir[dataset] + split + '.json'
    return loadfile


def get_model_name(model_name, dataset):
    if dataset in ['omniglot', 'cross_char']:
        assert model_name == 'Conv4', 'omniglot only support Conv4 without augmentation'
        model_name = 'Conv4S'
    return model_name


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir, epoch=None):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None
    if epoch is not None:
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
        return resume_file
    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

def get_checkpoint_dir(algorithm, model_name, dataset, train_n_way, n_shot, addition=None):
    if addition is None:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s' % (dataset, model_name, algorithm)
    else:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s_%s' % (dataset, model_name, algorithm, str(addition))
    if not algorithm in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (train_n_way, n_shot)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir