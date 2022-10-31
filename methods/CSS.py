import torch
from torch import nn
import numpy as np
import copy
from methods.meta_template import MetaTemplate
from torchvision import transforms
from PIL import Image


class CSS(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, use_cuda=True, adaptation=False, image_size=(84, 84),
                 classification_head='cosine'):
        super(CSS, self).__init__(model_func, n_way, n_support, use_cuda=use_cuda, adaptation=adaptation)
        self.loss_fn = nn.CrossEntropyLoss()
        self.pre_feature_extractor = copy.deepcopy(self.feature_extractor)
        self.ssl_feature_extractor = copy.deepcopy(self.feature_extractor)
        self.projection_mlp_1 = nn.Sequential(
            nn.Linear(self.feature_extractor.final_feat_dim, 2048),
        )
        self.projection_mlp_2 = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048)
        )
        self.prediction_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
        )
        self.alpha = nn.Parameter(torch.ones([1]))
        self.gamma = nn.Parameter(torch.ones([1]) * 2, requires_grad=False)
        self.image_size = image_size
        self.classification_head = classification_head

    def cosine_similarity(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return (x * y).sum(2)

    def euclidean_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return torch.pow(x - y, 2).sum(2)

    def set_pre_train_forward(self, x):
        z_support, z_query = self.parse_feature(x)
        z_support = self.projection_mlp_1(z_support)
        z_query = self.projection_mlp_1(z_query)
        z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
        if self.classification_head == 'consine':
            return self.cosine_similarity(z_query, z_proto) * 10
        else:
            return -self.euclidean_dist(z_query, z_proto)

    def set_pre_train_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
        if self.use_cuda:
            y_query = y_query.cuda()
        scores = self.set_pre_train_forward(x)
        return self.loss_fn(scores, y_query)

    def pre_train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_pre_train_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss

    def pre_train_test_loop(self, test_loader, record=None, return_std=False):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            with torch.no_grad():
                x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
                z_all = self.feature_extractor.forward(x)
                z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
                z_support = z_all[:, :self.n_support]  # [N, S, d]
                z_query = z_all[:, self.n_support:]  # [N, Q, d]
                z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
                z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
                scores = self.cosine_similarity(z_query, z_proto)
                y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
                topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
                topk_ind = topk_labels.cpu().numpy()  # index of topk
                acc_all.append(np.sum(topk_ind[:, 0] == y_query) / len(y_query) * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def f(self, x):
        # x:[N*(S+Q),n_channel,h,w]
        x = self.ssl_feature_extractor(x)
        x = self.projection_mlp_1(x)
        x = self.projection_mlp_2(x)
        return x

    def h(self, x):
        # x:[N*(S+Q),2048]
        x = self.prediction_mlp(x)
        return x

    def D(self, p, z):
        z = z.detach()
        p = torch.nn.functional.normalize(p, dim=1)
        z = torch.nn.functional.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def data_augmentation(self, img):
        # x:[n_channel,h,w], torch.Tensor
        x = transforms.RandomResizedCrop(self.image_size, interpolation=Image.BICUBIC)(img)
        x = transforms.RandomHorizontalFlip()(x)
        if np.random.random() < 0.8:
            x = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(x)
        else:
            x = transforms.RandomGrayscale(p=1.0)(x)
        x = transforms.GaussianBlur((5, 5))(x)
        return x

    def contrastive_loss(self, x):
        # x:[N*(S+Q),n_channel,h,w]
        x1 = x.clone()
        x2 = x.clone()
        for index in range(x.shape[0]):
            x1[index] = self.data_augmentation(x[index])
            x2[index] = self.data_augmentation(x[index])
        z1, z2 = self.f(x1), self.f(x2)
        p1, p2 = self.h(z1), self.h(z2)
        loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        return loss

    def ssl_train_loop(self, epoch, train_loader, optimizer):
        self.train()
        print_freq = 10
        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):  # x:[N, S+Q, n_channel, h, w]
            if self.use_cuda:
                x = x.cuda()
            x = x.reshape([x.shape[0] * x.shape[1], *x.shape[2:]])  # x:[N*(S+Q),n_channel,h,w]
            x_ssl = torch.nn.functional.normalize(self.ssl_feature_extractor(x), dim=1)
            x_pre = torch.nn.functional.normalize(self.feature_extractor(x).detach(), dim=1)
            loss = self.contrastive_loss(x) - torch.mean(torch.sum((x_ssl * x_pre), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))

    def ssl_test_loop(self, test_loader, record=None, return_std=False):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            with torch.no_grad():
                x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
                z_all = self.ssl_feature_extractor.forward(x)
                z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
                z_support = z_all[:, :self.n_support]  # [N, S, d]
                z_query = z_all[:, self.n_support:]  # [N, Q, d]
                z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
                z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
                scores = self.cosine_similarity(z_query, z_proto)
                y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
                topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
                topk_ind = topk_labels.cpu().numpy()  # index of topk
                acc_all.append(np.sum(topk_ind[:, 0] == y_query) / len(y_query) * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def set_forward(self, x):
        z_support, z_query = self.parse_feature(x)
        z_support = self.projection_mlp_1(z_support)
        z_query = self.projection_mlp_1(z_query)
        z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
        if self.classification_head == 'consine':
            return self.cosine_similarity(z_query, z_proto) * 10
        else:
            return -self.euclidean_dist(z_query, z_proto)

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
        if self.use_cuda:
            y_query = y_query.cuda()
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)

    def meta_train_loop(self, epoch, train_loader, optimizer):
        self.train()
        print_freq = 10
        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):  # x:[N, S+Q, n_channel, h, w]
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            if self.use_cuda:
                x = x.cuda()
            xx = x.reshape([x.shape[0] * x.shape[1], *x.shape[2:]])  # x:[N*(S+Q),n_channel,h,w]
            with torch.no_grad():
                x_pre = self.pre_feature_extractor(xx)
                x_ssl = self.ssl_feature_extractor(xx)
                x_aggregation = nn.functional.normalize(torch.cat([x_pre, x_ssl], dim=1), dim=1)
                similarity = torch.sum(torch.unsqueeze(x_aggregation, dim=0) * torch.unsqueeze(x_aggregation, dim=1),
                                       dim=2)
                for index in range(similarity.shape[0]):
                    similarity[index, index] = 0
                D = torch.diag(torch.sum(similarity, dim=1) ** -0.5)
                A = D @ similarity @ D
            if self.use_cuda:
                augment = (self.alpha * torch.eye(A.shape[0], A.shape[0]).cuda() + A) ** self.gamma
            else:
                augment = (self.alpha * torch.eye(A.shape[0], A.shape[0]) + A) ** self.gamma
            z_all = augment @ self.feature_extractor(xx)
            z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, -1)
            z_support = z_all[:, :self.n_support]
            z_query = z_all[:, self.n_support:]
            z_support = self.projection_mlp_1(z_support)
            z_query = self.projection_mlp_1(z_query)
            z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
            z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
            if self.use_cuda:
                y_query = y_query.cuda()
            scores = self.cosine_similarity(z_query, z_proto) * 10
            loss = self.loss_fn(scores, y_query) + self.set_forward_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss
