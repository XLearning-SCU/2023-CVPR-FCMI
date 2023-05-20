import itertools
# import time
# import warnings

# import numpy
import math
import os
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Experiment.Visualize import tsne, visualize_scatter, visual_matrix_console
from Utils import TimeOperator, DirectoryOperator
from DataSetMaster.dataset import get_clusters
from evaluate import UMAP, evaluate2
import evaluate
from loss import instance_contrastive_loss
import loss as loss_model
import faiss
from MainLauncher import LimitKmeans, CodeTest, DrawMax


def visualize(feature_vec, type_vec, group_vec, pred_vec, epoch):
    vis_fea = tsne(feature_vec)
    visualize_scatter(vis_fea,
                      fig_path='../Visualization/E{:03d}Type.jpg'.format(epoch),
                      label_color=type_vec,
                      label_shape=type_vec,
                      )
    visualize_scatter(vis_fea,
                      fig_path='../Visualization/E{:03d}Cluster.jpg'.format(epoch),
                      label_color=pred_vec,
                      label_shape=type_vec,
                      )
    visualize_scatter(vis_fea,
                      fig_path='../Visualization/E{:03d}Group.jpg'.format(epoch),
                      label_color=group_vec,
                      label_shape=type_vec,
                      )


def show_distribution_ct(type_vec, group_vec, pred_vec, class_num, group_num):
    v = np.zeros((class_num, class_num, group_num), dtype=int)
    for t, c, g in zip(type_vec, pred_vec, group_vec):
        v[t, c, g] += 1
    visual_matrix_console(x=v)


# if __name__ == '__main__':
#     gn = 3
#     t = np.arange(1000)
#     typ = t % 10
#     pred = t * t % 10
#     gr = t % gn
#     show_distribution_ct(type_vec=typ, group_vec=gr, pred_vec=pred, class_num=10, group_num=gn)


def inference(net, test_dataloader):
    net.eval()
    feature_vec, type_vec, group_vec, pred_vec = [], [], [], []
    with torch.no_grad():
        for (x, g, y, idx) in test_dataloader:
            x = x.cuda()
            g = g.cuda()
            h = net.encode(x, g)
            c = net.encode_class(h).detach()
            pred = torch.argmax(c, dim=1)
            feature_vec.extend(h[0].detach().cpu().numpy())
            type_vec.extend(y.cpu().numpy())
            group_vec.extend(g.cpu().numpy())
            pred_vec.extend(pred.cpu().numpy())
    feature_vec, type_vec, group_vec, pred_vec = (
        np.array(feature_vec),
        np.array(type_vec),
        np.array(group_vec),
        np.array(pred_vec),
    )
    # print(pred_vec[:50])
    d = net.representation_dim
    kmeans = faiss.Clustering(d, net.class_num)
    kmeans.verbose = False
    kmeans.niter = 300
    kmeans.nredo = 10
    if LimitKmeans:
        kmeans.max_points_per_centroid = 1000
        kmeans.min_points_per_centroid = 10
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)
    # print(feature_vec.shape)
    kmeans.train(feature_vec, index)
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(net.class_num, d)
    net.train()
    # print(centroids.shape)
    return feature_vec, type_vec, group_vec, pred_vec, centroids


def show_distribution(cluster_vec, group_vec, class_num, group_num):
    for it in np.arange(group_num):
        print('{:4d}, '.format(it), end='')
    print('')
    cluster_group = torch.zeros((class_num, group_num), dtype=torch.int)
    for i, j in zip(cluster_vec, group_vec):
        cluster_group[i, j] += 1
    # cluster_group = cluster_group[torch.argsort(torch.sum(cluster_group, dim=1))]
    for line in cluster_group:
        print('{:4d}: '.format(torch.sum(line)), end='')
        for it in line:
            print('{:4d}, '.format(it), end='')
        print('')


def save_checkpoint(state, epoch):
    """
    it has been trained for *epoch* epochs
    """
    filename = 'Epoch{:03d}.checkpoint'.format(epoch)
    checkpoint_dir = os.path.join(
        os.path.dirname(os.getcwd()),
        'Checkpoints',
        filename
    )
    DirectoryOperator.FoldOperator(directory=checkpoint_dir).make_fold()
    if os.path.exists(checkpoint_dir):
        warnings.warn('Checkpoint exist and been replaced.({})'.format(checkpoint_dir))
    print('Save check point into {}'.format(checkpoint_dir))
    torch.save(state, checkpoint_dir)


class Net(nn.Module):
    def __init__(self, class_num, group_num, args):
        super(Net, self).__init__()
        self.class_num = class_num
        if args.representation_dim > 0:
            self.representation_dim = args.representation_dim
        else:
            self.representation_dim = class_num
        self.group_num = group_num
        self.encoder_out_dim = 784
        self.args = args
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                for _ in range(self.group_num if args.GroupWiseEncoder else 1)
            ]
        )
        self.encoder_linear = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.encoder_out_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.representation_dim),
                )
                for _ in range(2 if args.WithFeatureB else 1)
            ]
        )
        self.cluster_centers = F.normalize(
            torch.rand(self.class_num, self.representation_dim), dim=1
        ).cuda()
        self.decoder_linear = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.representation_dim, 256), nn.ReLU(), nn.Linear(256, self.encoder_out_dim), nn.ReLU(),
                )
                for _ in range(2 if args.WithFeatureB else 1)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        32 if args.WithFeatureB else 16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        32, 32, kernel_size=3, stride=1, padding=1, output_padding=0
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
                    nn.Tanh(),
                )
                for _ in range(self.group_num if args.GroupWiseDecoder else 1)
            ]
        )
        self.discriminator_foreground = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(), nn.Linear(self.representation_dim, 512),
                nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
                nn.Linear(512, 256), nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2), nn.Linear(256, 1))
            for _ in range(group_num)
        ])
        self.initialize_weights(self.discriminator_foreground)

    def initialize_weights(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def update_cluster_center(self, center):
        center = torch.from_numpy(center).cuda()
        if self.args.RepresentationType == 'Normalize':
            center = F.normalize(center, dim=1)
        elif self.args.RepresentationType == 'None':
            pass
        elif self.args.RepresentationType == 'Relu':
            pass
        else:
            raise NotImplementedError('RepresentationType')
        self.cluster_centers = center

    def encode(self, x, group_indices: torch.tensor):
        if self.args.GroupWiseEncoder:
            hh = torch.zeros((len(x), self.encoder_out_dim), device='cuda')
            for g in torch.unique(group_indices):
                ind = g == group_indices
                # hh[ind] = self.encoder[g](x[ind])
                hh[ind] = self.encoder[g](x[ind])
        else:
            hh = self.encoder[0](x)
        a = self.encoder_linear[0](hh)
        if self.args.RepresentationType == 'Normalize':
            a = F.normalize(a, dim=1)
        elif self.args.RepresentationType == 'None':
            pass
        elif self.args.RepresentationType == 'Relu':
            pass
        else:
            raise NotImplementedError('RepresentationType')
        if self.args.WithFeatureB:
            b = self.encoder_linear[1](hh)
            if self.args.RepresentationType == 'Normalize':
                b = F.normalize(b, dim=1)
            elif self.args.RepresentationType == 'None':
                pass
            elif self.args.RepresentationType == 'Relu':
                b = nn.ReLU()(b)
            else:
                raise NotImplementedError('RepresentationType')
        else:
            b = None

        return a, b
        # return [F.normalize(el(hh), dim=1) for el in self.encoder_linear]

    def encode_class(self, h):
        # print(h[0].shape)
        # print(self.cluster_centers.shape)
        if self.args.RepresentationType == 'Normalize':
            c = h[0] @ self.cluster_centers.T
        elif self.args.RepresentationType == 'None':
            c = -torch.cdist(h[0], self.cluster_centers) ** 2 / 700
        elif self.args.RepresentationType == 'Relu':
            c = -torch.cdist(h[0], self.cluster_centers) ** 2 / 700
        else:
            raise NotImplementedError('RepresentationType')
        # print('c.shape == {}'.format(c.shape))
        return c

    def decode(self, z, group_indices):
        if self.args.WithFeatureB:
            z = torch.cat([dec(zi).view(-1, 16, 7, 7) for zi, dec in zip(z, self.decoder_linear)], dim=1)
        else:
            z = self.decoder_linear[0](z[0]).view(-1, 16, 7, 7)
        if self.args.GroupWiseDecoder:
            x_ = torch.zeros((len(z), 1, 28, 28), device='cuda')
            for g in torch.unique(group_indices):
                ind = g == group_indices
                if torch.sum(ind) == 0:
                    pass
                elif torch.sum(ind) == 1:
                    x_[ind] = self.decoder[g](torch.concat([z[ind], z[ind]]))[[0]]
                else:
                    x_[ind] = self.decoder[g](z[ind])
        else:
            x_ = self.decoder[0](z)
        return x_

    # def forward(self, x, **kwargs):
    #     t = self.encode(x, None)[0]
    #     g = np.arange(len(t)) % 2
    #     return np.sum([d(t).sum().detach().cpu() for d in self.discriminator_foreground])

    def run(self, epochs, train_dataloader, test_dataloader, args):
        if args.loss_self_cons:
            clusters = get_clusters(args=args)

        optimizer_g = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.encoder_linear.parameters(),
                self.decoder_linear.parameters(),
                self.decoder.parameters(),
            ),
            lr=args.LearnRate,
            betas=(args.betas_a, args.betas_v),
            weight_decay=args.WeightDecay
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator_foreground.parameters(),
            lr=args.LearnRate,
            betas=(args.betas_a, args.betas_v),
            weight_decay=args.WeightDecay
        )
        mse_loss = nn.MSELoss().cuda()
        ce_loss = nn.CrossEntropyLoss().cuda()
        timer_all = TimeOperator.TimeOperator()
        timer_train = TimeOperator.TimeOperator()
        timer_infer = TimeOperator.TimeOperator()
        type_detail_shown = False
        start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                # if args.gpu is None:
                #     checkpoint = torch.load(args.resume)
                # else:
                #     # Map model to be loaded to specified single gpu.
                #     loc = 'cuda:{}'.format(args.gpu)
                #     checkpoint = torch.load(args.resume, map_location=loc)
                start_epoch = checkpoint['epoch']
                self.load_state_dict(checkpoint['state_dict'])
                optimizer_g.load_state_dict(checkpoint['optimizer']['optimizer_g'])
                optimizer_d.load_state_dict(checkpoint['optimizer']['optimizer_d'])
                self.__dict__ = checkpoint['self_dic']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                # warnings.warn('This is not equal to start from the beginning due to different rands states.')
                epochs = start_epoch + 1
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        for epoch in range(start_epoch, epochs):
            if (epoch + 1) == epochs or (epoch + 1) == 2 or (epoch + 1) == args.WarmAll:
                dct = {
                    'epoch'     : epoch,
                    'state_dict': self.state_dict(),
                    'optimizer' : {'optimizer_g': optimizer_g.state_dict(), 'optimizer_d': optimizer_d.state_dict()},
                }
                dct = {**dct, 'self_dic': self.__dict__}
                save_checkpoint(dct, epoch=epoch)
            if (epoch + 1) <= args.LearnRateWarm:
                lr = args.LearnRate * (epoch + 1) / args.LearnRateWarm
            else:
                if args.LearnRateDecayType == 'None':
                    lr = args.LearnRate
                elif args.LearnRateDecayType == 'Exp':
                    lr = args.LearnRate * ((1 + 10 * (epoch + 1 - args.LearnRateWarm) / (
                            args.train_epoch - args.LearnRateWarm)) ** -0.75)
                elif args.LearnRateDecayType == 'Cosine':
                    lr = args.LearnRate * 0.5 * (1. + math.cos(
                        math.pi * (epoch + 1 - args.LearnRateWarm) / (args.train_epoch - args.LearnRateWarm)))
                else:
                    raise NotImplementedError('args.LearnRateDecayType')
            if lr != args.LearnRate:
                def adjust_learning_rate(optimizer):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                adjust_learning_rate(optimizer_g)
                adjust_learning_rate(optimizer_d)

            timer_all.time()
            timer_infer.time()
            if (epoch + 1) == 101:
                args.VisualFreq = max(50, args.VisualFreq)
            # t_tmp = time.time()
            feature_vec, type_vec, group_vec, pred_vec, centers = inference(
                self, test_dataloader
            )
            # print('inf-time', time.time() - t_tmp)
            # t_tmp = time.time()
            if epoch % 1 == 0:
                self.update_cluster_center(centers)
                # warnings.warn('self.update_cluster_center(centers)')
            # print('center-time', time.time() - t_tmp)
            # t_tmp = time.time()
            pred_adjusted = evaluate2(feature_vec, pred_vec, type_vec, group_vec)
            # print('eva-time', time.time() - t_tmp)
            if (epoch + 1) == epochs or (epoch + 1) == 2 or (epoch + 1) % max(1, args.WarmAll) == 0:
                np_dir = '../NpPoints/Np{:03d}'.format(epoch)
                DirectoryOperator.FoldOperator(directory=np_dir).make_fold()
                np.savez(np_dir, feature_vec=feature_vec, type_vec=type_vec,
                         group_vec=group_vec, pred_vec=pred_adjusted, epoch=epoch)
            if epoch == 3:
                evaluate.BestBalance = 0.0
                evaluate.BestEntropy = 0.0
                evaluate.BestFairness = 0.0
                evaluate.BestNmiFair = 0.0
            print('Clu\\g ', end='')
            show_distribution(cluster_vec=pred_adjusted, group_vec=group_vec, class_num=self.class_num,
                              group_num=self.group_num)
            if not type_detail_shown:
                type_detail_shown = True
                print('Typ\\g ', end='')
                show_distribution(cluster_vec=type_vec, group_vec=group_vec, class_num=self.class_num,
                                  group_num=self.group_num)
            show_distribution_ct(type_vec=type_vec, group_vec=group_vec, pred_vec=pred_adjusted,
                                 class_num=self.class_num,
                                 group_num=self.group_num)

            if (epoch + 1) % args.VisualFreq == 0:
                # if (epoch + 1) % args.VisualFreq == 0 or (epoch + 1) == args.VisualFreq // 2 + 1 or \
                #         args.VisualFreq - 5 <= (epoch + 1) <= args.VisualFreq + 10:
                if len(feature_vec) > DrawMax:
                    it = np.arange(len(feature_vec))
                    np.random.shuffle(it)
                    ind = it[:DrawMax]
                    feature_vec_visual = feature_vec[ind]
                    type_vec_visual = type_vec[ind]
                    group_vec_visual = group_vec[ind]
                    pred_vec_visual = pred_adjusted[ind]
                else:
                    feature_vec_visual = feature_vec
                    type_vec_visual = type_vec
                    group_vec_visual = group_vec
                    pred_vec_visual = pred_adjusted
                if args.DrawTSNE:
                    visualize(feature_vec=feature_vec_visual, type_vec=type_vec_visual, group_vec=group_vec_visual,
                              pred_vec=pred_vec_visual, epoch=epoch)
                if args.DrawUmap:
                    UMAP(
                        feature_vec_visual,
                        type_vec_visual,
                        group_vec_visual,
                        pred_vec_visual,
                        self.class_num,
                        self.group_num,
                        args,
                        epoch=epoch,
                    )
                loss_model.PrintedGlobalDiversityLoss = False
            if args.resume and epoch == epochs - 1:
                return
            type_vec = torch.from_numpy(type_vec)
            group_vec = torch.from_numpy(group_vec)
            pred_vec = torch.from_numpy(pred_vec).cuda()
            print()
            timer_infer.time()
            timer_train.time()
            self.train()
            confidence_sum = 0.0
            loss_reconstruction_epoch = 0.0
            loss_reconstruct_all_epoch = 0.0
            discriminative_test_loss = 0.0
            loss_discriminative_epoch = 0.0
            loss_decenc_epoch = 0.0
            loss_global_balance_epoch = 0.0
            loss_info_global_epoch = 0.0
            loss_info_balance_epoch = 0.0
            loss_info_fair_epoch = 0.0

            # independent_global_balance_epoch = 0.0
            loss_group_wise_balance_epoch = 0.0
            loss_cluster_wise_balance_epoch = 0.0
            loss_onehot_epoch = 0.0
            loss_consistency_epoch = 0.0
            loss_proto_epoch = 0.0

            for iter, (x, g, y, idx) in enumerate(train_dataloader):
                # print(x[200])
                x = x.cuda()
                g = g.cuda()
                h0, h1 = self.encode(x, g)
                x_ = self.decode([h0, h1], g)
                c = self.encode_class([h0, h1])
                # print('c == {}'.format(c))

                # print('args.BalanceTemperature == {}'.format(args.BalanceTemperature))
                # print('c_ == {}'.format(c_))
                confidence_sum += F.softmax(c / 0.2, dim=1).detach().max(dim=1).values.mean()
                loss = 0
                if args.Reconstruction and epoch < args.ReconstructionEpoch:
                    loss_rec = mse_loss(x, x_)
                    if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                        fe = x.view((len(x), -1))
                        fe_ = x_.view((len(x_), -1))
                        print('torch.mean(x)=={}, torch.std(x)=={}'.format(torch.mean(fe), torch.std(fe)))
                        print('torch.min(x)=={}, torch.max(x)=={}'.format(torch.min(fe), torch.max(fe)))
                        # print('torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))=={}'.format(
                        #     torch.sqrt(torch.sum(fe ** 2, dim=1, keepdim=False))))
                        # print('torch.sqrt(torch.sum(x ** 2, dim=0, keepdim=True))=={}'.format(
                        #     torch.sqrt(torch.sum(fe ** 2, dim=0, keepdim=False))))
                        print('x[:3]=={}'.format(fe[:3]))
                        print('x_[:3]=={}'.format(fe_[:3]))

                        # print()

                    # loss_rec = 0
                    # print('loss_rec.item() == {}'.format(loss_rec.item()))
                    # print('mse_loss(x[g==0], x_[g==0]).item() == {}'.format(mse_loss(x[g==0], x_[g==0]).item()))
                    # print('mse_loss(x[g==1], x_[g==1]).item() == {}'.format(mse_loss(x[g==1], x_[g==1]).item()))

                    loss = loss_rec * args.Reconstruction
                    loss_reconstruction_epoch += loss_rec.item()
                if epoch >= args.WarmAll:
                    if args.reconstruct_all:
                        hs = [h0, h1.detach()[np.random.permutation(len(h1))]]
                        reconstruct_all = mse_loss(h0, self.encode(self.decode(hs, g), g)[0])
                        loss += reconstruct_all * args.reconstruct_all
                        loss_reconstruct_all_epoch += reconstruct_all.item()
                    if self.group_num <= 2:
                        g_shuffle = 1 - g
                    else:
                        g_shuffle = g[torch.randperm(c.shape[0])]
                    for _ in range(args.DiscriminativeTest):
                        loss_dis = 0
                        optimizer_d.zero_grad()
                        # if isinstance(loss_dis, int):
                        #     loss_bac = loss_dis
                        # else:
                        #     loss_bac = loss_dis.item()
                        for i in range(self.group_num):
                            r = self.discriminator_foreground[i](h0.detach())
                            loss_dis += ((r[g == i] - 1) ** 2).sum() + ((r[g_shuffle == i] - 0) ** 2).sum()
                        loss_dis /= x.shape[0]
                        loss_dis.backward()
                        discriminative_test_loss += loss_dis.item()

                        # print('loss_dis.item() - loss_bac == {}'.format(loss_dis.item() - loss_bac))
                        optimizer_d.step()

                    if args.Discriminative:
                        for _ in range(3):
                            loss_dis = 0
                            optimizer_d.zero_grad()
                            # if isinstance(loss_dis, int):
                            #     loss_bac = loss_dis
                            # else:
                            #     loss_bac = loss_dis.item()
                            for i in range(self.group_num):
                                r = self.discriminator_foreground[i](h0.detach())
                                loss_dis += ((r[g == i] - 1) ** 2).sum() + ((r[g_shuffle == i] - 0) ** 2).sum()
                            loss_dis /= x.shape[0]
                            loss_dis.backward()
                            # print('loss_dis.item() == {}'.format(loss_dis.item()))
                            # print('loss_dis.item() - loss_bac == {}'.format(loss_dis.item() - loss_bac))
                            optimizer_d.step()
                        loss_dis = 0
                        for i in range(self.group_num):
                            r = self.discriminator_foreground[i](h0)
                            loss_dis += ((r - 0.5) ** 2).sum()
                        loss_dis /= x.shape[0]
                        loss += loss_dis * args.Discriminative
                        loss_discriminative_epoch += loss_dis.item()
                    if args.Decenc:
                        if len(np.unique(group_vec)) <= 2:
                            lab_g = 1 - g
                        else:
                            lab_g = g_shuffle
                        x_rev = self.decode([h0.detach(), h1.detach() if h1 is not None else None], lab_g)
                        h_rev, h1_rev = self.encode(x_rev, lab_g)
                        loss_decenc = mse_loss(h0, h_rev)
                        loss += loss_decenc * args.Decenc
                        loss_decenc_epoch += loss_decenc.item()
                    if epoch >= args.WarmBalance:
                        c_balance = F.softmax(c / args.SoftAssignmentTemperatureBalance, dim=1)

                        def get_loss(matrix_a, matrix_b):
                            if not args.BalanceLossNoDetach:
                                matrix_b = matrix_b.detach()
                            if args.BalanceLossType == 'KL':
                                # print(O)
                                # print(E)
                                # print('O / E', matrix_a / matrix_b)
                                # print('torch.log(O / E)', torch.log(matrix_a / matrix_b))
                                # print('O * torch.log(O / E)', matrix_a * torch.log(matrix_a / matrix_b))
                                # print('torch.sum(O * torch.log(O / E))', torch.sum(matrix_a * torch.log(matrix_a / matrix_b)))
                                return torch.sum(matrix_a * torch.log(matrix_a / matrix_b)) / len(matrix_a)
                                # return F.kl_div(torch.log(matrix_a), matrix_b, reduction="batchmean")
                            elif args.BalanceLossType == 'MSE':
                                return mse_loss(matrix_a, matrix_b)
                            elif args.BalanceLossType == 'MAE':
                                return torch.mean(torch.abs(matrix_a - matrix_b))
                            else:
                                raise NotImplementedError('BalanceLossType')

                        O = torch.zeros((self.class_num, self.group_num)).cuda()
                        E = torch.zeros((self.class_num, self.group_num)).cuda()
                        for b in range(self.group_num):
                            O[:, b] = torch.sum(c_balance[g == b], dim=0)
                            E[:, b] = (g == b).sum()
                        E[E <= 0] = torch.min(E[E > 0]) / 10
                        O[O <= 0] = torch.min(O[O > 0]) / 10
                        pcg = O / torch.sum(O)
                        if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                            print('c[:3] == \n{}'.format(c[:3]))
                            print('c_balance[:3] == \n{}'.format(c_balance[:3]))
                            O_Hard = torch.zeros((self.class_num, self.group_num)).cuda()
                            for ci, gi in zip(c_balance, g):
                                O_Hard[torch.argmax(ci), gi] += 1
                            print('O_Hard == \n{}'.format(O_Hard))
                            print('pcg == \n{}'.format(pcg))

                        if args.GlobalBalanceLoss:
                            global_balance_loss = get_loss(
                                (O / torch.sum(O)).view((1, -1)),
                                # (E / torch.sum(E, dim=1, keepdim=True)).view((1, -1))
                                (E / torch.sum(E)).view((1, -1))
                            )
                            # matrix_a = (O / torch.sum(O)).view((1, -1))
                            # matrix_b = (E / torch.sum(E)).view((1, -1)).detach()
                            # global_balance_loss = torch.sum(matrix_a * torch.log(matrix_a / matrix_b))/len(matrix_a)
                            loss_global_balance_epoch += global_balance_loss.item()
                            loss += global_balance_loss * args.GlobalBalanceLoss
                        if args.InfoGlobalLoss:
                            pc = torch.sum(pcg, dim=1, keepdim=False)
                            info_global_loss = torch.sum(pc * torch.log(pc))
                            loss_info_global_epoch += info_global_loss.item()
                            loss += info_global_loss * args.InfoGlobalLoss
                            assert False
                        if args.InfoBalanceLoss:
                            pc = torch.sum(pcg, dim=1, keepdim=False)
                            info_balance_loss = torch.sum(pc * torch.log(pc))
                            loss_info_balance_epoch += info_balance_loss.item()
                            loss += info_balance_loss * args.InfoBalanceLoss
                            if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                                print('pc == {}'.format(pc))
                                print('pc * torch.log(pc) == {}'.format(pc * torch.log(pc)))

                        if args.InfoFairLoss:
                            pc = torch.sum(pcg, dim=1, keepdim=True)
                            pg = torch.sum(pcg, dim=0, keepdim=True)
                            info_fair_loss = torch.sum(pcg * torch.log(pcg / (pc * pg)))
                            loss_info_fair_epoch += info_fair_loss.item()
                            loss += info_fair_loss * args.InfoFairLoss
                        # def get_loss(matrix_a, matrix_b):
                        #     if (epoch + 1) % args.VisualFreq == 0 and iter==0:
                        #         print('c_[:3] == \n{}'.format(c_[:3]))
                        #         O_Hard = torch.zeros((self.class_num, self.group_num)).cuda()
                        #         for c, g_ in zip(c_, g):
                        #             O_Hard[torch.argmax(c), g_] += 1
                        #         print('O_Hard == \n{}'.format(O_Hard))
                        #         print('O == \n{}'.format(torch.exp(O)))
                        #         print('E == \n{}'.format(E))
                        #     if args.BalanceLossType == 'KL':
                        #         return F.kl_div(torch.log(matrix_a), matrix_b.detach(), reduction="batchmean")
                        #     elif args.BalanceLossType == 'MSE':
                        #         return mse_loss(matrix_a, matrix_b)
                        #     elif args.BalanceLossType == 'MAE':
                        #         raise NotImplementedError('BalanceLossType')
                        #     else:
                        #         raise NotImplementedError('BalanceLossType')
                        # O = torch.zeros((self.class_num, self.group_num)).cuda()
                        # E = torch.zeros((self.class_num, self.group_num)).cuda()
                        # for b in range(self.group_num):
                        #     O[:, b] = torch.sum(c_[g == b], dim=0)
                        #     E[:, b] = (g == b).sum()
                        # if args.GlobalBalanceLoss:
                        #     global_balance_loss = get_loss(
                        #         (O / torch.sum(O)).view((1, -1)),
                        #         (E / torch.sum(E)).view((1, -1))
                        #     )
                        #     loss_global_balance_epoch += global_balance_loss.item()
                        #     loss += global_balance_loss * args.GlobalBalanceLoss

                        # if args.IndependentBalanceLoss:
                        #     independent_balance_loss = get_loss(
                        #         (O / torch.sum(O)).view((1, -1)),
                        #         (E / torch.sum(E)).view((1, -1))
                        #     )
                        #     # p_matrix = O / torch.sum(O)
                        #     # p_c_g = torch.sum(p_matrix, dim=1, keepdim=True)
                        #     # p_g = torch.sum(p_matrix, dim=0, keepdim=True)
                        #     # independent_balance_loss = get_loss(
                        #     #     p_matrix.view((1, -1)),
                        #     #     (p_c * p_g).view((1, -1))
                        #     # )
                        #     independent_global_balance_epoch += independent_balance_loss.item()
                        #     loss += independent_balance_loss * args.IndependentBalanceLoss
                        if args.GroupWiseBalanceLoss:
                            group_wise_balance_loss = get_loss(
                                (O / torch.sum(O, dim=1, keepdim=True)),
                                (E / torch.sum(E, dim=1, keepdim=True))
                            )
                            loss_group_wise_balance_epoch += group_wise_balance_loss.item()
                            loss += group_wise_balance_loss * args.GroupWiseBalanceLoss
                        if args.ClusterWiseBalanceLoss:
                            cluster_wise_balance_loss = get_loss(
                                (O / torch.sum(O, dim=0, keepdim=True)).transpose(1, 0),
                                (E / torch.sum(E, dim=0, keepdim=True)).transpose(1, 0)
                            )
                            loss_cluster_wise_balance_epoch += cluster_wise_balance_loss.item()
                            loss += cluster_wise_balance_loss * args.ClusterWiseBalanceLoss
                    if epoch >= args.WarmOneHot and args.OneHot:
                        c_onehot = F.softmax(c / args.SoftAssignmentTemperatureHot, dim=1)
                        loss_onehot = -torch.sum(c_onehot * torch.log(c_onehot + 1e-8)) / float(len(c_onehot))
                        loss += loss_onehot * args.OneHot
                        loss_onehot_epoch += loss_onehot.item()
                    if epoch >= args.WarmConsistency and args.loss_self_cons:
                        if args.SelfConsLossType == 'Feature':
                            loss_self_cons = instance_contrastive_loss(h0, clusters[idx], g)
                        elif args.SelfConsLossType == 'Assignment' or args.SelfConsLossType == 'LogAssignment':
                            def get_loss(ob, ev):
                                if args.SelfConsLossType == 'Assignment':
                                    return mse_loss(ob, ev)
                                elif args.SelfConsLossType == 'LogAssignment':
                                    log_sf = -torch.log_softmax(ob / 0.5, dim=1)
                                    if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                                        print('ob[:5, :5] == \n{}'.format(ob[:5, :5]))
                                        print('torch.exp(-log_sf)[:5, :5] == \n{}'.format(torch.exp(-log_sf)[:5, :5]))
                                        print('log_sf[:5, :5] == \n{}'.format(log_sf[:5, :5]))
                                    losse = torch.sum(ev * log_sf) / torch.sum(ev)
                                    return losse
                                else:
                                    raise NotImplementedError('args.SelfConsLossType')

                            hot = torch.eye(self.class_num).cuda()
                            loss_self_cons = 0.
                            assignment = clusters[idx]
                            for g0 in torch.unique(g):
                                # print('g0 == {}'.format(g0))
                                c_cons = F.softmax(c / args.SoftAssignmentTemperatureSelfCons, dim=1)
                                pm = c_cons[g0 == g]
                                # print('pm.shape == {}'.format(pm.shape))
                                p1m = hot[assignment[g0 == g]]
                                # print('p1m.shape == {}'.format(p1m.shape))
                                # lc = mse_loss(pm @ pm.T, p1m@p1m.T)
                                if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                                    print('(pm @ pm.T)[:5, :5] == \n{}'.format((pm @ pm.T)[:5, :5]))
                                    print('(p1m@p1m.T)[:5, :5] == \n{}'.format((p1m @ p1m.T)[:5, :5]))

                                loss_self_cons += get_loss(pm @ pm.T, p1m @ p1m.T)
                                # print('loss_self_cons == {}'.format(lc))
                        else:
                            raise NotImplementedError('args.SelfConsLossType')
                        loss += loss_self_cons * args.loss_self_cons
                        loss_consistency_epoch += loss_self_cons.item()
                    if epoch >= args.WarmUpProto and args.loss_cons:
                        loss_cons = ce_loss(c, pred_vec[idx])
                        loss += loss_cons * args.loss_cons
                        loss_proto_epoch += loss_cons.item()
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()

            len_train_dataloader = len(train_dataloader)
            confidence_sum /= len_train_dataloader
            loss_reconstruction_epoch /= len_train_dataloader
            loss_reconstruct_all_epoch /= len_train_dataloader
            discriminative_test_loss /= len_train_dataloader * max(1, args.DiscriminativeTest)
            loss_discriminative_epoch /= len_train_dataloader
            loss_decenc_epoch /= len_train_dataloader
            loss_global_balance_epoch /= len_train_dataloader
            loss_info_global_epoch /= len_train_dataloader
            loss_info_balance_epoch /= len_train_dataloader
            loss_info_fair_epoch /= len_train_dataloader
            loss_group_wise_balance_epoch /= len_train_dataloader
            loss_cluster_wise_balance_epoch /= len_train_dataloader
            loss_onehot_epoch /= len_train_dataloader
            loss_consistency_epoch /= len_train_dataloader
            loss_proto_epoch /= len_train_dataloader

            print('Epoch [{: 3d}/{: 3d}]'.format(epoch + 1, epochs), end='')
            if loss_reconstruction_epoch != 0:
                print(', Reconstruction:{:04f}'.format(loss_reconstruction_epoch), end='')
            if loss_reconstruct_all_epoch != 0:
                print(', ReconstructAll:{:04f}'.format(loss_reconstruct_all_epoch), end='')
            if discriminative_test_loss != 0:
                print(', DiscriminativeTest:{:04f}'.format(discriminative_test_loss), end='')
            if loss_discriminative_epoch != 0:
                print(', Discriminative:{:04f}'.format(loss_discriminative_epoch), end='')
            if loss_decenc_epoch != 0:
                print(', Decenc:{:04f}'.format(loss_decenc_epoch), end='')
            if loss_global_balance_epoch != 0:
                print(', GlobalBalance:{:04f}'.format(loss_global_balance_epoch), end='')
            if loss_info_global_epoch != 0:
                print(', InfoGlobal:{:04f}'.format(loss_info_global_epoch), end='')
            if loss_info_balance_epoch != 0:
                print(', InfoBalance:{:04f}'.format(loss_info_balance_epoch), end='')
            if loss_info_fair_epoch != 0:
                print(', InfoFair:{:04f}'.format(loss_info_fair_epoch), end='')
            # if independent_global_balance_epoch != 0:
            # print(', IndependentBalance:{:04f}'.format(independent_global_balance_epoch), end='')
            if loss_group_wise_balance_epoch != 0:
                print(', GroupWiseBalance:{:04f}'.format(loss_group_wise_balance_epoch), end='')
            if loss_cluster_wise_balance_epoch != 0:
                print(', ClusterWiseBalance:{:04f}'.format(loss_cluster_wise_balance_epoch), end='')
            if loss_consistency_epoch != 0:
                print(', Consistency:{:04f}'.format(loss_consistency_epoch), end='')
            if loss_onehot_epoch != 0:
                print(', OneHot:{:04f}'.format(loss_onehot_epoch), end='')
            if loss_proto_epoch != 0:
                print(', Proto:{:04f}'.format(loss_proto_epoch), end='')
            if confidence_sum != 0:
                print(', Confidence:{:04f}'.format(confidence_sum), end='')
            print()
            timer_train.time()
            timer_all.time()
            if (epoch + 1) % args.VisualFreq == 0:
                timer_train.show_process(process_now=epoch + 1, process_total=args.train_epoch, name='Train')
                timer_infer.show_process(process_now=epoch + 1, process_total=args.train_epoch, name='Infer')
                timer_all.show_process(process_now=epoch + 1, process_total=args.train_epoch, name='All')
            if CodeTest:
                assert False


class MyNormalize(nn.Module):
    def __call__(self, x):
        return x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))


class Gassilize(nn.Module):
    def __init__(self, args):
        if args.dataset == 'MTFL':
            warnings.warn('bad mean and std')
        super(Gassilize, self).__init__()

    def __call__(self, x):
        return (x - 0.39618438482284546) / 0.4320564270019531


class NetFCN(Net):
    def __init__(self, input_dim, class_num, group_num, args):
        super().__init__(class_num, group_num, args=args)
        self.input_dim = input_dim
        self.encoder_out_dim = 512

        def get_encoder_list():
            if args.BatchNormType[0] == '1':
                return nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.encoder_out_dim),
                    nn.BatchNorm1d(self.encoder_out_dim),
                    nn.ReLU(),
                )
            elif args.BatchNormType[0] == '0':
                return nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.encoder_out_dim),
                    nn.ReLU(),
                )
            else:
                raise NotImplementedError('')

        def get_encoder_line_list():
            if args.BatchNormType[1] == '1':
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, self.representation_dim),
                )
            elif args.BatchNormType[1] == '0':
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.representation_dim),
                )
            else:
                raise NotImplementedError('')

        def get_decoder_list():
            m_list = [
                nn.Linear(1024, input_dim),
            ]
            if args.dataset == 'MTFL' or args.dataset == 'Office':
                if args.ActivationType == 'None':
                    pass
                elif args.ActivationType == 'Sigmoid':
                    m_list.append(nn.Sigmoid())
                elif args.ActivationType == 'Tanh':
                    m_list.append(nn.Tanh())
                elif args.ActivationType == 'Normalize':
                    m_list.append(MyNormalize())
                elif args.ActivationType == 'Gaussainlize':
                    m_list.append(Gassilize(args))
                elif args.ActivationType == 'GlS_GaussainlizeAndSigmoid':
                    m_list.append(Gassilize(args))
                    m_list.append(nn.Sigmoid())
                elif args.ActivationType == 'GlT_GaussainlizeAndTanh':
                    m_list.append(Gassilize(args))
                    m_list.append(nn.Tanh())
                else:
                    raise NotImplementedError('')
            elif args.dataset == 'HAR':
                m_list.append(nn.Tanh())
            else:
                raise NotImplementedError('')

            if args.BatchNormType[3] == '1':
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim * 2 if args.WithFeatureB else self.encoder_out_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    *m_list
                )
            elif args.BatchNormType[3] == '0':
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim * 2 if args.WithFeatureB else self.encoder_out_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    *m_list
                )

            else:
                raise NotImplementedError('')

        def get_decoder_line_list():
            if args.BatchNormType[2] == '1':
                return nn.Sequential(
                    nn.Linear(self.representation_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, self.encoder_out_dim),
                    nn.BatchNorm1d(self.encoder_out_dim),
                    nn.ReLU(),
                )
            elif args.BatchNormType[2] == '0':
                return nn.Sequential(
                    nn.Linear(self.representation_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.encoder_out_dim),
                    nn.ReLU(),
                )
            else:
                raise NotImplementedError('')

        self.encoder = nn.ModuleList([
            get_encoder_list() for _ in range(self.group_num if args.GroupWiseEncoder else 1)
        ])
        self.encoder_linear = nn.ModuleList([
            get_encoder_line_list() for _ in range(2 if args.WithFeatureB else 1)
        ])
        self.decoder_linear = nn.ModuleList([
            get_decoder_line_list() for _ in range(2 if args.WithFeatureB else 1)
        ])
        self.decoder = nn.ModuleList([
            get_decoder_list() for _ in range(self.group_num if args.GroupWiseDecoder else 1)
        ])

    def decode(self, z, group_indices):
        if self.args.WithFeatureB:
            z = torch.cat([dec(zi) for zi, dec in zip(z, self.decoder_linear)], dim=1)
        else:
            z = self.decoder_linear[0](z[0])
        if self.args.GroupWiseDecoder:
            x_ = torch.zeros((len(z), self.input_dim), device='cuda')
            for g in torch.unique(group_indices):
                ind = g == group_indices
                if torch.sum(ind) == 0:
                    pass
                elif torch.sum(ind) == 1:
                    x_[ind] = self.decoder[g](torch.concat([z[ind], z[ind]]))[[0]]
                else:
                    x_[ind] = self.decoder[g](z[ind])
        else:
            x_ = self.decoder[0](z)
        return x_


def test():
    def get_kl_loss(type_num, group_num, n, batch_num=1000):
        mse_loss = nn.MSELoss().cuda()

        def get_loss(matrix_a, matrix_b):
            # if args.BalanceLossType == 'KL':
            # return F.kl_div(torch.log(matrix_a), matrix_b.detach(), reduction="batchmean") * (n / 20000) / (group_num / 2)
            # elif args.BalanceLossType == 'MSE':
            return mse_loss(matrix_a, matrix_b) * (n / 20000) * (group_num / 2) * (type_num / 10) ** 2
            # elif args.BalanceLossType == 'MAE':
            #     pass
            # else:
            #     raise NotImplementedError('BalanceLossType')

        loss = 0

        for _ in range(batch_num):
            # c_ = torch.rand((n, type_num))*2-1
            # c_ /= torch.sum(c_, dim=1, keepdim=True)
            c_ = torch.softmax(torch.cos(torch.rand((n, type_num)) * torch.pi) / 0.1, dim=1)
            # print(c_[:2])
            g = torch.as_tensor(torch.floor(torch.rand(n) * group_num), dtype=torch.int)
            # print(g)
            O = torch.zeros((type_num, group_num)).cuda()
            E = torch.zeros((type_num, group_num)).cuda()
            for b in range(group_num):
                O[:, b] = torch.sum(c_[g == b], dim=0)
                E[:, b] = (g == b).sum()

            # if args.GlobalBalance:
            loss += get_loss(
                (O / torch.sum(O)).view((1, -1)),
                (E / torch.sum(E)).view((1, -1))
            )
        # if args.GroupWiseBalanceLoss:
        #     loss +=  get_loss(
        #         (O / torch.sum(O, dim=1, keepdim=True)),
        #         (E / torch.sum(E, dim=1, keepdim=True))
        #     ) * n / group_num
        #     loss_group_wise_balance_epoch += group_wise_balance_loss.item()
        #     loss += group_wise_balance_loss * args.GroupWiseBalanceLoss
        # if args.ClusterWiseBalanceLoss:
        #     loss += get_loss(
        #         (O / torch.sum(O, dim=0, keepdim=True)).transpose(1, 0),
        #         (E / torch.sum(E, dim=0, keepdim=True)).transpose(1, 0)
        #     ) * n / group_num
        #     loss_cluster_wise_balance_epoch += cluster_wise_balance_loss.item()
        #     loss += cluster_wise_balance_loss * args.ClusterWiseBalanceLoss
        #
        # matrix_a = torch.rand((4, 3))
        # matrix_b = torch.ones((4, 3))
        # matrix_a /= torch.sum(matrix_a, dim=1, keepdim=True)
        # matrix_b /= torch.sum(matrix_b, dim=1, keepdim=True)
        # kl = F.kl_div(torch.log(matrix_a), matrix_b.detach(), reduction="batchmean")
        print('{:.08f}'.format(loss / batch_num))

    get_kl_loss(type_num=10, group_num=2, n=512)
    get_kl_loss(type_num=10, group_num=2, n=512)
    # get_kl_loss(type_num=10, group_num=2, n=5120)
    # get_kl_loss(type_num=10, group_num=30, n=5120)
    get_kl_loss(type_num=100, group_num=2, n=512)


# 0.16537261 0.08160374
# 0.16192900 0.08480503
# 0.16470969 0.08299696
# 0.17563063 0.20080499
# 0.16742396 0.08369189
def test2():
    type_num = 10
    group_num = 2
    n = 512
    O = torch.zeros((type_num, group_num))
    E = torch.zeros((type_num, group_num))
    c_ = torch.softmax(torch.cos(torch.rand((n, type_num)) * torch.pi) / 0.1, dim=1)
    # print(c_[:2])
    g = torch.as_tensor(torch.floor(torch.rand(n) * group_num), dtype=torch.int)
    print(c_)
    for b in range(group_num):
        O[:, b] = torch.sum(c_[g == b], dim=0)
        E[:, b] = (g == b).sum()
    print(O)


if __name__ == '__main__':
    test()
    # matrix_a = torch.linspace(1, 11, 12).view((4, -1))
    # matrix_b = torch.linspace(1, 111, 12).view((4, -1))

    # my_kl = torch.mean(torch.sum(matrix_a * torch.log(matrix_a / matrix_b), dim=1))
    # print(my_kl)

    # my_kl2 = torch.mean(torch.sum(matrix_b * torch.log(matrix_b / matrix_a), dim=1))
    # print(my_kl2)
