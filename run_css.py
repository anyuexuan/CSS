from utils.utils import *
from methods.CSS import CSS
import copy

datasets = ['cifar', 'CUB', 'miniImagenet']

classification_head = 'cosine'

for dataset in datasets:
    print(dataset)
    # region parameters
    algorithm = 'css'  # protonet/matchingnet/relationnet
    model_name = 'Conv4'  # Conv4/Conv6/ResNet10/ResNet18/ResNet34/ResNet50/ResNet101
    n_shot = 5  # number of labeled data in each class, same as n_support
    stop_epoch = -1
    pre_train_epoch = -1
    ssl_train_epoch = -1
    if_pre_train = True
    if_ssl_train = True
    if_meta_train = True
    if_test = True
    if 'Conv' in model_name:
        image_resize = (84, 84)
    else:
        image_resize = (224, 224)
    # endregion

    image_size = get_image_size(model_name=model_name, dataset=dataset)

    if stop_epoch == -1:
        stop_epoch = get_stop_epoch(algorithm=algorithm, dataset=dataset, n_shot=n_shot)
    if pre_train_epoch == -1:
        pre_train_epoch = stop_epoch
    if ssl_train_epoch == -1:
        ssl_train_epoch = stop_epoch
    checkpoint_dir = get_checkpoint_dir(algorithm=algorithm, model_name=model_name, dataset=dataset,
                                        train_n_way=train_n_way, n_shot=n_shot, addition='%f' % noise_rate)
    base_file, val_file = get_train_files(dataset=dataset)
    base_loader, val_loader = get_train_loader(algorithm=algorithm, image_size=image_size, base_file=base_file,
                                               val_file=val_file, train_n_way=train_n_way, test_n_way=test_n_way,
                                               n_shot=n_shot, noise_rate=noise_rate, val_noise=True,
                                               num_workers=num_workers)


    def pre_train():
        print('Start pre-training!')
        model = CSS(model_dict[model_name], n_way=train_n_way, n_support=n_shot, use_cuda=use_cuda,
                    adaptation=adaptation, image_size=image_resize, classification_head=classification_head)
        if use_cuda:
            model = model.cuda()
        max_acc = 0
        optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 1e-3},
                                      {'params': model.projection_mlp_1.parameters(), 'lr': 1e-6}])
        for pre_epoch in range(0, pre_train_epoch):
            model.train()
            model.pre_train_loop(pre_epoch, base_loader, optimizer)  # model are called by reference, no need to return
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model.eval()
            acc = model.pre_train_test_loop(val_loader)
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                print('epoch:', pre_epoch, 'pre_train val acc:', acc, 'best!')
                max_acc = acc
                outfile = os.path.join(checkpoint_dir, 'pre_train_best.tar')
                torch.save({'epoch': pre_epoch, 'state': model.state_dict()}, outfile)
            if (pre_epoch % save_freq == 0) or (pre_epoch == stop_epoch - 1):
                outfile = os.path.join(checkpoint_dir, 'pre_train_{:d}.tar'.format(pre_epoch))
                torch.save({'epoch': pre_epoch, 'state': model.state_dict()}, outfile)
        return model


    def ssl_train():
        print('Start ssl-training!')
        model = CSS(model_dict[model_name], n_way=train_n_way, n_support=n_shot, use_cuda=use_cuda,
                    adaptation=adaptation, image_size=image_resize, classification_head=classification_head)
        if use_cuda:
            model = model.cuda()
        outfile = os.path.join(checkpoint_dir, 'pre_train_best.tar')
        tmp = torch.load(outfile)
        model.load_state_dict(tmp['state'])
        max_acc = 0
        optimizer = torch.optim.Adam([{'params': model.ssl_feature_extractor.parameters(), 'lr': 1e-3},
                                      {'params': model.projection_mlp_1.parameters(), 'lr': 1e-3},
                                      {'params': model.projection_mlp_2.parameters(), 'lr': 1e-3},
                                      {'params': model.prediction_mlp.parameters(), 'lr': 1e-3}, ])
        for ssl_epoch in range(0, ssl_train_epoch):
            model.train()
            model.ssl_train_loop(ssl_epoch, base_loader, optimizer)
            model.eval()
            acc = model.ssl_test_loop(val_loader)
            if acc > max_acc:
                print('epoch:', ssl_epoch, 'ssl_train val acc:', acc, 'best!')
                max_acc = acc
                outfile = os.path.join(checkpoint_dir, 'ssl_train_best.tar')
                torch.save({'epoch': ssl_epoch, 'state': model.state_dict()}, outfile)
            if (ssl_epoch % save_freq == 0) or (ssl_epoch == ssl_train_epoch - 1):
                outfile = os.path.join(checkpoint_dir, 'ssl_train_{:d}.tar'.format(ssl_epoch))
                torch.save({'epoch': ssl_epoch, 'state': model.state_dict()}, outfile)
        return model


    def meta_train():
        print('Start meta-training!')
        model = CSS(model_dict[model_name], n_way=train_n_way, n_support=n_shot, use_cuda=use_cuda,
                    adaptation=adaptation, image_size=image_resize, classification_head=classification_head)
        if use_cuda:
            model = model.cuda()
        model.pre_feature_extractor = copy.deepcopy(model.feature_extractor)
        outfile = os.path.join(checkpoint_dir, 'ssl_train_best.tar')
        tmp = torch.load(outfile)
        model.load_state_dict(tmp['state'])
        max_acc = 0
        optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 1e-3},
                                      {'params': model.projection_mlp_1.parameters(), 'lr': 1e-3},
                                      {'params': model.alpha, 'lr': 1e-3}])
        for epoch in range(start_epoch, stop_epoch):
            model.train()
            model.meta_train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
            model.eval()
            acc = model.test_loop(val_loader)
            if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                print("--> Best model! save...", acc)
                max_acc = acc
                outfile = os.path.join(checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if (epoch % save_freq == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        return model


    def test(phase='test'):
        print('Start testing!')
        model = CSS(model_dict[model_name], n_way=train_n_way, n_support=n_shot, use_cuda=use_cuda,
                    adaptation=adaptation, image_size=image_resize, classification_head=classification_head)
        if use_cuda:
            model = model.cuda()
        if phase == 'pre':
            modelfile = os.path.join(checkpoint_dir, 'pre_train_best.tar')
            assert modelfile is not None
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])
        elif phase == 'ssl':
            modelfile = os.path.join(checkpoint_dir, 'ssl_train_best.tar')
            assert modelfile is not None
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])
        elif phase == 'test':
            modelfile = get_best_file(checkpoint_dir)
            assert modelfile is not None
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])

        loadfile = get_novel_file(dataset=dataset, split='novel')
        datamgr = SetDataManager(image_size, n_eposide=test_iter_num, n_query=15, n_way=test_n_way, n_support=n_shot,
                                 noise_rate=0., num_workers=num_workers)
        novel_loader = datamgr.get_data_loader(loadfile, aug=False)
        model.eval()
        if phase == 'pre':
            acc_mean, acc_std = model.pre_train_test_loop(novel_loader, return_std=True)
        elif phase == 'ssl':
            acc_mean, acc_std = model.ssl_test_loop(novel_loader, return_std=True)
        else:
            acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))
        return model


    if if_pre_train:
        pre_train()
    if if_test:
        test('pre')
    if if_ssl_train:
        ssl_train()
    if if_test:
        test('ssl')
    if if_meta_train:
        meta_train()
    test()
