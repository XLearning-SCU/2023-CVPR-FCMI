import os
import warnings
import numpy as np
import torch
from torch.utils import data
import torchvision
from PIL import Image

from MainLauncher import path_operator

# from Utils import PathPresettingOperator
#
# path_operator = PathPresettingOperator.PathOperator(model_name='FairClustering')

NumWorkers = 4


def get_dataset(dataset, group=-1, path=path_operator.get_dataset_path, args=None, res_selector=None, transforms=None,
                flatten=False):
    """

    :param flatten:
    :param transforms:
    :param dataset:
    :param group:
    :param path:
    :param args:
    :param res_selector: [0,1,2,3]: img, g, target, index; [0,2]: img, g, target, index
    :return:
    """
    if args is None:
        mnist_train = False
    else:
        mnist_train = args.MnistTrain

    class MNIST(torchvision.datasets.MNIST):
        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            if flatten:
                img = img.view((-1))
            if res_selector is not None:
                return np.asarray([img, 0, target, index], dtype=object)[res_selector]
            return img, 0, target, index

    class Rev_MNIST(torchvision.datasets.MNIST):
        def __init__(self, *arg, **kwargs):
            warnings.warn('Rev_MNIST index')
            super(Rev_MNIST, self).__init__(*arg, **kwargs)

        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            img, g, target, index = img, 1, target, index + (60000 if mnist_train else 10000)
            if flatten:
                img = img.view((-1))
            if res_selector is not None:
                return np.asarray([img, 1, target, index], dtype=object)[res_selector]
            return img, g, target, index

    class USPS(torchvision.datasets.USPS):
        def __init__(self, *arg, **kwargs):
            warnings.warn('USPS index')
            super(USPS, self).__init__(*arg, **kwargs)

        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img, mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            if flatten:
                img = img.view((-1))
            img, g, target, index = img, 1, target, index + (60000 if mnist_train else 10000)
            if res_selector is not None:
                return np.asarray([img, 1, target, index], dtype=object)[res_selector]
            return img, g, target, index

    class MyImageFolder_0(torchvision.datasets.ImageFolder):
        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            img, g, target, index = sample, 0, target, index
            return img, g, target, index

    class MyImageFolder_1(torchvision.datasets.ImageFolder):
        def __init__(self, root, transform, ind_bias=2817, *args, **kwargs):
            super(MyImageFolder_1, self).__init__(root, transform, *args, **kwargs)
            self.ind_bias = ind_bias

        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            img, g, target, index = sample, 1, target, index + self.ind_bias
            return img, g, target, index

    def featurelize(dataset):
        test_loader = data.DataLoader(dataset, batch_size=512, num_workers=NumWorkers)
        net = torchvision.models.resnet50(pretrained=True).cuda()
        # newnorm
        if args.dataset =='MTFL':
            mean = 0.34949973225593567
            std = 0.3805956244468689
        elif args.dataset =='Office':
            mean = 0.3970963954925537
            std = 0.43060600757598877
        else:
            raise NotImplementedError('args.dataset')

        # old norm
        # mean = 0.39618438482284546
        # std = 0.4320564270019531
        def forward(net, x):
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)
            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3(x)
            x = net.layer4(x)
            x = net.avgpool(x)
            x = torch.flatten(x, 1)
            if args is None or args.FeatureType == 'Tanh':
                x = torch.nn.Tanh()(x)
            elif args.FeatureType == 'Gaussainlize':
                x = (x - mean) / std
            elif args.FeatureType == 'Normalize':
                x /= torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
            elif args.FeatureType == 'Default':
                pass
            elif args.FeatureType == 'GlS_GaussainlizeAndSigmoid':
                x = torch.nn.Sigmoid()((x - mean) / std)
            elif args.FeatureType == 'GlT_GaussainlizeAndTanh':
                x = torch.nn.Tanh()((x - mean) / std)
            elif args.FeatureType == 'Sigmoid':
                x = torch.nn.Sigmoid()(x)
            else:
                raise NotImplementedError('FeatureType')
            return x

        net.eval()
        feature_vec, type_vec, group_vec, idx_vec = [], [], [], []
        with torch.no_grad():
            for (x, g, y, idx) in test_loader:
                x = x.cuda()

                g = g.cuda()
                c = forward(net, x)

                feature_vec.extend(c.cpu().numpy())
                type_vec.extend(y.cpu().numpy())
                group_vec.extend(g.cpu().numpy())
                idx_vec.extend(idx.cpu().numpy())
        # feature_vec, type_vec, group_vec = np.array(feature_vec), np.array(
        #     type_vec), np.array(group_vec)
        x = torch.from_numpy(np.array(feature_vec))

        # x = (x - x.min()) / (x.max() - x.min())
        print('torch.mean(x)=={}, torch.std(x)=={}'.format(torch.mean(x), torch.std(x)))
        print('torch.min(x)=={}, torch.max(x)=={}'.format(torch.min(x), torch.max(x)))
        print('torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))=={}'.format(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))))
        # assert False
        img, g, target, index = x, torch.as_tensor(group_vec), torch.as_tensor(type_vec), torch.as_tensor(idx_vec)
        if flatten:
            img = img.view((len(img), -1))
        if res_selector is not None:
            item = np.asarray([img, g, target, index], dtype=object)[res_selector]
        else:
            item = [img, g, target, index]

        return data.TensorDataset(*item)

    if dataset == 'MNISTUSPS':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(28),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ])
        mnist = MNIST(train=mnist_train,
                      download=True,
                      root=path(''),
                      transform=transforms)
        usps = USPS(train=True,
                    download=True,
                    root=path('USPS'),
                    transform=transforms)
        if group == -1:
            data_set = data.ConcatDataset([mnist, usps])
        elif group == 0:
            data_set = mnist
        elif group == 1:
            data_set = usps
        else:
            raise NotImplementedError('group')

        class_num = 10
    elif dataset == 'ReverseMNIST':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5])
            ])

        class Reverse:
            def __call__(self, img):
                return 1 - img

        rev_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            Reverse(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        mnist = MNIST(train=mnist_train,
                      download=False,
                      root=path(''),
                      transform=transforms)
        rev_mnist = Rev_MNIST(train=mnist_train,
                              download=True,
                              root=path(''),
                              transform=rev_transforms)
        if group == -1:
            data_set = data.ConcatDataset([mnist, rev_mnist])
        elif group == 0:
            data_set = mnist
        elif group == 1:
            data_set = rev_mnist
        else:
            raise NotImplementedError('group')

        class_num = 10
    elif dataset == 'Office':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(224),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                # newnorm
                torchvision.transforms.Normalize(mean=[0.7076, 0.7034, 0.7021],
                                                 std=[0.3249, 0.3261, 0.3275]),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                                  std=[0.229, 0.224, 0.225]),
            ])
        amazon = MyImageFolder_0(path('Office/amazon/images'),
                                 transform=transforms)
        webcam = MyImageFolder_1(path('Office/webcam/images'),
                                 transform=transforms)
        if group == -1:
            office = data.ConcatDataset([amazon, webcam])
        elif group == 0:
            office = amazon
        elif group == 1:
            office = MyImageFolder_1(path('Office/webcam/images'),
                                     transform=transforms, ind_bias=0)
        else:
            raise NotImplementedError('group')
        # print(office)
        # office
        # imgs = torch.stack([t[0] for t in office], dim=0)
        # print(imgs.shape)
        # t = torch.mean(imgs, dim=[0,2,3])
        # print(t)
        # t = torch.std(imgs, dim=[0,2,3])
        # print(t)
        data_set = featurelize(dataset=office)
        # assert False
        class_num = 31
    elif dataset == 'HAR':
        train_X = np.loadtxt(os.path.join(path('HAR'), 'train/X_train.txt'))
        train_G = np.loadtxt(os.path.join(path('HAR'), 'train/subject_train.txt'))
        train_Y = np.loadtxt(os.path.join(path('HAR'), 'train/y_train.txt'))
        test_X = np.loadtxt(os.path.join(path('HAR'), 'test/X_test.txt'))
        test_G = np.loadtxt(os.path.join(path('HAR'), 'test/subject_test.txt'))
        test_Y = np.loadtxt(os.path.join(path('HAR'), 'test/y_test.txt'))
        X = torch.cat((torch.from_numpy(train_X), torch.from_numpy(test_X)),
                      dim=0)
        G = torch.cat((torch.from_numpy(train_G), torch.from_numpy(test_G)),
                      dim=0) - 1
        Y = torch.cat((torch.from_numpy(train_Y), torch.from_numpy(test_Y)),
                      dim=0) - 1
        re_ind = np.argsort(G.numpy(), kind='stable') if group == -1 else G == group
        X = X[re_ind]
        G = G[re_ind]
        Y = Y[re_ind]

        #####
        # v = torch.sort(torch.sqrt(torch.sum(X ** 2, dim=0)))
        # print(v[0])
        # print(torch.amax(X , dim=0))
        # print(torch.amin(X , dim=0))
        # print(v[0][0])
        # print(v[0][0][0])
        # print(X.shape)
        # print(v[:50])
        # print(v[-50:])
        # assert False
        idx = torch.arange(0, X.shape[0])
        img, g, target, index = X.float(), G.long(), Y.long(), idx.long()
        if res_selector is not None:
            item = np.asarray([img, g, target, index], dtype=object)[res_selector]
        else:
            item = [img, g, target, index]

        data_set = data.TensorDataset(*item)
        class_num = 6

    elif dataset == 'MTFL':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(224),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                                  std=[0.229, 0.224, 0.225]),
                #newnorm
                torchvision.transforms.Normalize(mean=[0.4951, 0.4064, 0.3579],
                                                 std=[0.2868, 0.2594, 0.2545]),
            ])
        use_1000 = True
        g0 = MyImageFolder_0(root=path('MTFLfold{}/g0'.format('-1000' if use_1000 else '')),
                             transform=transforms)
        g1 = MyImageFolder_1(ind_bias=1000 if use_1000 else 11346,
                             root=path('MTFLfold{}/g1'.format('-1000' if use_1000 else '')),
                             transform=transforms)
        if group == -1:
            mtfl = data.ConcatDataset([g0, g1])
        elif group == 0:
            mtfl = g0
        elif group == 1:
            mtfl = MyImageFolder_1(ind_bias=0,
                                   root=path('MTFLfold{}/g1'.format('-1000' if use_1000 else '')),
                                   transform=transforms)
        else:
            raise NotImplementedError('group')
        # imgs = torch.stack([t[0] for t in mtfl], dim=0)
        # print(imgs.shape)
        # t = torch.mean(imgs, dim=[0,2,3])
        # print(t)
        # t = torch.std(imgs, dim=[0,2,3])
        # print(t)
        # assert  False
        # office = amazon
        data_set = featurelize(dataset=mtfl)
        class_num = 2
    else:
        raise NotImplementedError('split')
    return data_set, class_num


def get_dataloader(dataset, batch_size=512, **kwargs):
    data_set, class_num = get_dataset(dataset, **kwargs)
    train_loader = data.DataLoader(data_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True, num_workers=NumWorkers)
    test_loader = data.DataLoader(data_set, batch_size=batch_size * 100, num_workers=NumWorkers)

    return train_loader, test_loader, class_num


def get_clusters(args):
    item_path = os.path.join(path_operator.get_checkpoint_path(level=1), 'Items0321')
    file_mnist_test = os.path.join(item_path, 'mnist_test_clusters89.67.txt')
    file_mnist_train = os.path.join(item_path, 'MnistTrain94.31B256.txt')
    file_amazon = os.path.join(item_path, 'amazon72.81B032ReValue.txt')
    file_webcam = os.path.join(item_path, 'webcamOurLoaderRevalveBatchWiseB032_84.03.txt')
    file_usps = os.path.join(item_path, 'usps_train_clusters85.10.txt')
    root_har = os.path.join(item_path, 'HAR')
    root_mtfl = os.path.join(item_path, 'MTFL')

    if args.dataset == 'MNISTUSPS':  # 87.75 93.31
        if args.MnistTrain:
            file_mnist = file_mnist_train
        else:
            file_mnist = file_mnist_test
        file_list = [
            file_mnist,
            file_usps,
        ]
    elif args.dataset == 'ReverseMNIST':  # 89.67 94.31
        if args.MnistTrain:
            file_mnist = file_mnist_train
        else:
            file_mnist = file_mnist_test
        file_list = [
            file_mnist,
            file_mnist,
        ]
    elif args.dataset == 'Office':  # 75.28
        file_list = [
            file_amazon,
            file_webcam,
        ]
    elif args.dataset == 'MTFL':
        file_list = np.sort([os.path.join(root_mtfl, f) for f in os.listdir(root_mtfl) if f.endswith('txt')])
    elif args.dataset == 'HAR':  # 81.70
        file_list = np.sort([os.path.join(root_har, f) for f in os.listdir(root_har) if f.endswith('txt')])
    else:
        raise NotImplementedError("")

    def debug(x):
        print(x.shape)
        return x

    clusters = torch.cat(
        [debug(torch.from_numpy(np.loadtxt(c).astype(np.float32)).long()) for c in file_list],
        dim=0,
    ).cuda()
    return clusters
