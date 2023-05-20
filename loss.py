
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, class_num, temperature):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature = temperature

        self.mask_ins = self.mask_correlated_samples(batch_size)
        self.mask_clu = self.mask_correlated_samples(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, size):
        N = 2 * size
        mask = torch.ones((N, N)).cuda()
        mask = mask.fill_diagonal_(0)
        for i in range(size):
            mask[i, size + i] = 0
            mask[size + i, i] = 0
        mask = mask.bool()
        return mask

    # def forward_(self, c, predicted_label):
    #     # Entropy Loss
    #     # p = c.sum(0).view(-1)
    #     # p /= p.sum()
    #     # ne_loss = math.log(p.size(0)) + (p * torch.log(p)).sum()
    #
    #     # CE Loss
    #     # previous_predicted_indices = (predicted_label != -1)
    #     # idx, counts = torch.unique(predicted_label[previous_predicted_indices],
    #     #                            return_counts=True)
    #     # n = torch.sum(counts)
    #     # freq = n / counts.float()
    #     # weight = torch.ones(self.class_num).cuda()
    #     # # weight[idx] = freq / freq.sum() * self.class_num
    #     # weight[idx] = freq
    #     # criterion = nn.CrossEntropyLoss(weight=weight)
    #     # logits = c.t()[previous_predicted_indices]
    #     # if n > 0:
    #     #     loss_ce = criterion(logits,
    #     #                         predicted_label[previous_predicted_indices])
    #     # else:
    #     #     loss_ce = 0
    #     pass

    def forward(self, z_i, z_j, g):
        # z_i, z_j = F.normalize(z_i, dim=1), F.normalize(z_j, dim=1)
        mask = torch.eq(g.view(-1, 1), g.view(1, -1)).cuda()
        # mask = torch.eye(self.batch_size).bool().cuda()
        mask = mask.float()

        contrast_count = 2
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(self.batch_size * anchor_count).view(-1, 1).cuda(),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, self.batch_size).mean()

        return loss


# Printed = True

# def cluster_diversity_loss(R, batch_indices, type_num, batch_num, args):
#     O = torch.zeros((type_num, batch_num)).cuda()
#     E = torch.zeros((type_num, batch_num)).cuda()
#     batch_size = R.shape[0]
#     # phi = torch.zeros((batch_size, batch_num)).cuda()
#     for i in range(batch_size):
#         O[:, batch_indices[i]] += R[i]
#         # phi[i, batch_indices[i]] = 1.0
#     for b in range(batch_num):
#         E[:, b] = (batch_indices == b).sum()  # / batch_size  # * R.sum(dim=0)
#     # loss_kl = 0s
#     # for i in range(batch_size):
#     #     for k in range(type_num):
#     #         for b in range(batch_num):
#     #             if batch_indices[i] == b:
#     #                 loss_kl += R[i, k] * torch.log(O[k, b] / E[k, b])
#     # loss_kl = torch.mul(torch.mm(R, torch.log(O.detach() / E.detach())), phi).sum() / batch_size
#     if args.DivBalance:
#         O = torch.log(O / torch.sum(O, dim=1, keepdim=True))
#         E = E / torch.sum(E, dim=1, keepdim=True)
#     else:
#         if args.DivKl:
#             div = E * (1 - E / torch.sum(E, dim=1, keepdim=True))
#             O = O / torch.sqrt(div * torch.sum(O, dim=1, keepdim=True) / torch.sum(E, dim=1, keepdim=True))
#             E = E / torch.sqrt(div)
#
#         O = O.log_softmax(dim=1)
#         E = E.softmax(dim=1)
#         # O /= len(batch_indices) / type_num / batch_num
#         # E /= len(batch_indices) / batch_num
#     # global Printed
#     # if not Printed:
#     #     Printed = True
#     #     print('R[:3] == \n{}'.format(R[:3]))
#     #     O_Hard = torch.zeros((type_num, batch_num)).cuda()
#     #     for c, g in zip(R, batch_indices):
#     #         O_Hard[torch.argmax(c), g] += 1
#     #     print('O_Hard == \n{}'.format(O_Hard))
#     #     print('O == \n{}'.format(torch.exp(O)))
#     #         # print('torch.exp(O.log_softmax(dim=1)) == {}'.format(torch.exp(O.log_softmax(dim=1))))
#     #
#     #     print('E == \n{}'.format(E))
#     #         # print('E.softmax(dim=1).detach() == {}'.format(E.softmax(dim=1).detach()))
#     global PrintedGlobalDiversityLoss
#     if not PrintedGlobalDiversityLoss:
#         PrintedGlobalDiversityLoss = True
#         print('R[:3] == \n{}'.format(R[:3]))
#         O_Hard = torch.zeros((type_num, batch_num)).cuda()
#         for c, g in zip(R, batch_indices):
#             O_Hard[torch.argmax(c), g] += 1
#         print('O_Hard == \n{}'.format(O_Hard))
#         print('O == \n{}'.format(torch.exp(O)))
#         print('E == \n{}'.format(E))
#
#     loss_kl = F.kl_div(
#         O, E.detach(), reduction="batchmean"
#     )  # + F.kl_div(E.log_softmax(dim=1).detach(),
#     # print('loss_kl == {}'.format(loss_kl))
#     #   O.softmax(dim=1),
#     #   reduction='batchmean')
#     # print(loss_kl)
#
#     # print('ProDistribution:')
#     # for line in O:
#     #     print(line.cpu())
#     return loss_kl
#
#
# PrintedGlobalDiversityLoss = False


# def global_diversity_loss(R, batch_indices, type_num, batch_num=2):
#     O = torch.zeros((type_num, batch_num)).cuda()
#     E = torch.zeros((type_num, batch_num)).cuda()
#     batch_size = R.shape[0]
#     # phi = torch.zeros((batch_size, batch_num)).cuda()
#     for i in range(batch_size):
#         O[:, batch_indices[i]] += R[i]
#         # phi[i, batch_indices[i]] = 1.0
#     for b in range(batch_num):
#         E[:, b] = (batch_indices == b).sum()  # / batch_size  # * R.sum(dim=0)
#     O = O.view((1, -1))
#     E = E.view((1, -1))
#     # print('O / torch.sum(O) == \n{}'.format(O / torch.sum(O)))
#     # print('E / torch.sum(E) == \n{}'.format(E / torch.sum(E)))
#
#     O = torch.log(O / torch.sum(O))
#     E = E / torch.sum(E)
#     # global PrintedGlobalDiversityLoss
#     # if not PrintedGlobalDiversityLoss:
#     #     PrintedGlobalDiversityLoss = True
#     #     print('R[:3] == \n{}'.format(R[:3]))
#     #     O_Hard = torch.zeros((type_num, batch_num)).cuda()
#     #     for c, g in zip(R, batch_indices):
#     #         O_Hard[torch.argmax(c), g] += 1
#     #     print('O_Hard == \n{}'.format(O_Hard))
#     #     print('O == \n{}'.format(torch.exp(O)))
#     #     print('E == \n{}'.format(E))
#
#     loss_kl = F.kl_div(
#         O, E.detach(), reduction="batchmean"
#     )  # + F.kl_div(E.log_softmax(dim=1).detach(),
#     return 4 * loss_kl - one_hot_loss_weight*torch.sum(R * torch.log(R + 1e-8)) / float(len(R))


# def cluster_diversity_clusterwiseloss(R, batch_indices, type_num, batch_num, args):
#     if args.EntropyCW:
#         entro = -torch.sum(R * torch.log(R + 1e-8)) / float(len(R))
#         r = torch.mean(R, dim=0)
#         entro_h = -torch.sum(r * torch.log(r + 1e-8))
#         entro_loss = entro - 4 * entro_h
#         return entro_loss
#     else:
#         # O = torch.zeros((type_num, batch_num))
#         # E = torch.zeros((type_num, batch_num))
#         O = torch.zeros((type_num, batch_num)).cuda()
#         E = torch.zeros((type_num, batch_num)).cuda()
#         batch_size = R.shape[0]
#         # phi = torch.zeros((batch_size, batch_num)).cuda()
#         for i in range(batch_size):
#             O[:, batch_indices[i]] += R[i]
#             # phi[i, batch_indices[i]] = 1.0
#         for b in range(batch_num):
#             E[:, b] = (batch_indices == b).sum()  # / batch_size  # * R.sum(dim=0)
#         # loss_kl = 0s
#         # for i in range(batch_size):
#         #     for k in range(type_num):
#         #         for b in range(batch_num):
#         #             if batch_indices[i] == b:
#         #                 loss_kl += R[i, k] * torch.log(O[k, b] / E[k, b])
#         # loss_kl = torch.mul(torch.mm(R, torch.log(O.detach() / E.detach())), phi).sum() / batch_size
#         O = torch.transpose(O, 1, 0)
#         E = torch.transpose(E, 1, 0)
#         if DivBalance:
#             O = torch.log(O / torch.sum(O, dim=1, keepdim=True))
#             E = E / torch.sum(E, dim=1, keepdim=True)
#         else:
#             if DivKl:
#                 div = E * (1 - E / torch.sum(E, dim=1, keepdim=True))
#                 O = O / torch.sqrt(div * torch.sum(O, dim=1, keepdim=True) / torch.sum(E, dim=1, keepdim=True))
#                 E = E / torch.sqrt(div)
#
#             O = O.log_softmax(dim=1)
#             E = E.softmax(dim=1)
#         # print(O)
#         # print(E)
#         loss_kl = F.kl_div(
#             O, E.detach(), reduction="batchmean"
#         )  # + F.kl_div(E.log_softmax(dim=1).detach(),
#         print('R == {}'.format(R))
#         O_Hard = torch.zeros((type_num, batch_num)).cuda()
#         for c, g in zip(R, batch_indices):
#             O_Hard[torch.argmax(c), g] += 1
#         print('O_Hard == {}'.format(O_Hard))
#         print('O == {}'.format(O))
#         print('torch.exp(O.log_softmax(dim=1)) == {}'.format(torch.exp(O.log_softmax(dim=1))))
#
#         # print('E == {}'.format(E))
#         print('E.softmax(dim=1).detach() == {}'.format(E.softmax(dim=1).detach()))
#
#         print('loss_kl.item() == {}'.format(loss_kl.item()))
#         print('loss_kla.item() == {}'.format(F.kl_div(
#             O[[0]].log_softmax(dim=1), E[[0]].softmax(dim=1).detach(), reduction="batchmean"
#         ).item()))
#         print('loss_klb.item() == {}'.format(F.kl_div(
#             O[[1]].log_softmax(dim=1), E[[1]].softmax(dim=1).detach(), reduction="batchmean"
#         ).item()))
#         # print(loss_kl)
#         #
#         # v = O.softmax(dim=1)
#         # print(v)
#         # v = v * torch.log((v) / E.softmax(dim=1))
#         # print(v)
#         # lo = torch.sum(torch.sum(v, dim=1))
#         # print(lo)
#         #   O.softmax(dim=1),
#         #   reduction='batchmean')
#         # print(loss_kl)
#
#         # print('ProDistribution:')
#         # for line in O:
#         #     print(line.cpu())
#         return loss_kl


# if __name__ == '__main__':
#     cluster_diversity_clusterwiseloss(np.arange(15).reshape(5, -1) % 3, np.arange(5) % 2, 3, batch_num=2)


# def cluster_consistency_loss(c, g, preds):
#     loss = 0
#     for i in range(2):
#         group_idx = (g == i)
#         group_pred = preds[group_idx]
#         group_c = c[group_idx]

#         loss += ContrastiveLoss(group_idx.sum(), 10, 0.5)(group_c, group_c,
#                                                           group_pred)
#     return loss


def mask_correlated(size):
    N = 2 * size
    mask = torch.ones((N, N)).cuda()
    mask = mask.fill_diagonal_(0)
    for i in range(size):
        mask[i, size + i] = 0
        mask[size + i, i] = 0
    mask = mask.bool()
    return mask


def cluster_consistency_loss(c_i, c_j, class_num):
    c_i = c_i.t()
    c_j = c_j.t()
    N = 2 * class_num
    c = torch.cat((c_i, c_j), dim=0)
    sim = nn.CosineSimilarity(dim=2)(c.unsqueeze(1), c.unsqueeze(0))
    # sim = c @ c.T
    sim_i_j = torch.diag(sim, class_num)
    sim_j_i = torch.diag(sim, -class_num)
    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_clusters = sim[mask_correlated(class_num)].reshape(N, -1)
    labels = torch.zeros(N).to(positive_clusters.device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    cluster_loss = nn.CrossEntropyLoss(reduction="sum")(logits, labels) / N

    # # BT
    # def off_diagonal(x):
    #     # return a flattened view of the off-diagonal elements of a square matrix
    #     n, m = x.shape
    #     assert n == m
    #     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    # z1 = c_i
    # z2 = c_j
    # # bn = nn.BatchNorm1d(class_num, affine=False).cuda()
    # # c = bn(z1).T @ bn(z2)
    # c = z1.T @ z2

    # # sum the cross-correlation matrix between all gpus
    # c.div_(z1.shape[0])

    # # use --scale-loss to multiply the loss by a constant factor
    # # see the Issues section of the readme
    # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    # off_diag = off_diagonal(c).pow_(2).sum()
    # cluster_loss = on_diag + 5e-3 * off_diag  #3.9e-3

    return cluster_loss


def instance_loss(c, h, g, K=1):
    batch_size = c.shape[0]
    similarity = h[g == 0] @ h[g == 1].T
    mask_a = torch.zeros_like(similarity).cuda().bool()
    mask_b = torch.zeros_like(similarity).cuda().bool()
    sort_a = torch.argsort(-similarity, dim=1)
    sort_b = torch.argsort(-similarity, dim=0)
    for i in range(similarity.shape[0]):
        mask_a[i, sort_a[i, :K]] = True
    for i in range(similarity.shape[1]):
        mask_b[sort_b[:K, i], i] = True
    mask = mask_a & mask_b
    same_y = torch.eq(c[g == 0].view(-1, 1), c[g == 1].view(1, -1)).cuda()
    # print(same_y[mask].sum() / mask.sum())
    pred_similarity = h[g == 0] @ h[g == 1].T
    instance_loss = (
        pred_similarity[mask].add_(-1).pow_(2).mean()
    )  # + 5e-3 * pred_similarity[~mask].pow_(2).sum()

    # mask = torch.eye(batch_size).bool().to(z_i.device)
    # mask = mask.float()

    # contrast_count = 2
    # contrast_feature = torch.cat((z_i, z_j), dim=0)

    # anchor_feature = contrast_feature
    # anchor_count = contrast_count

    # # compute logits
    # anchor_dot_contrast = torch.div(
    #     torch.matmul(anchor_feature, contrast_feature.T), 0.5)
    # # for numerical stability
    # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    # logits = anchor_dot_contrast - logits_max.detach()

    # # tile mask
    # mask = mask.repeat(anchor_count, contrast_count)
    # # mask-out self-contrast cases
    # logits_mask = torch.scatter(
    #     torch.ones_like(mask), 1,
    #     torch.arange(batch_size * anchor_count).view(-1, 1).to(z_i.device), 0)
    # mask = mask * logits_mask

    # # compute log_prob
    # exp_logits = torch.exp(logits) * logits_mask
    # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # # compute mean of log-likelihood over positive
    # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # # loss
    # instance_loss = -mean_log_prob_pos
    # instance_loss = instance_loss.view(anchor_count, batch_size).mean()

    return instance_loss


def consistency_loss(c, g, pre_cluster):
    loss = 0.0
    mse_loss = nn.MSELoss()
    for i in range(g.max() + 1):
        group_idx = g == i
        c_group = c[group_idx]
        pre_cluster_group = pre_cluster[group_idx]
        target_structure = (
            torch.eq(pre_cluster_group.view(-1, 1), pre_cluster_group.view(1, -1))
                .cuda()
                .float()
        )
        current_structure = c_group @ c_group.T
        loss += mse_loss(current_structure, target_structure)
    return loss


def instance_contrastive_loss(c, pseudo_label, g):
    loss = 0.0
    # g= g.cuda()
    loss_rec = np.zeros(len(torch.unique(g)))
    for i in torch.unique(g):
        batch_size = (g == i).sum()
        if batch_size <= 0:
            continue
        # print('g == i.shape == {}'.format((g == i).shape))

        # # print(c.cpu())
        # # print((g == i).cpu().numpy())
        # c_group = c.cpu()[g == i].cuda()
        # # print('c_group.shape == {}'.format(c_group.shape))
        # # print('c_group == {}'.format(c_group))
        # # print(pseudo_label.shape)
        #
        # pred_group = pseudo_label.cpu()[g == i].cuda()
        # # print(pred_group.shape)
        # # print(pred_group.cpu())
        # # pred_group = pseudo_label.cpu()[g == i].cuda()
        # s = torch.cuda.Stream()
        # with torch.cuda.stream(s):
        #     (g == i).sum()
        #     # print('(g == i).sum() == {}'.format((g == i).sum()))
        #     c_group = c[g == i]
        #     # print('c.shape == {}'.format(c.shape))
        #     # print('c_group.shape == {}'.format(c_group.shape))
        # s = torch.cuda.Stream()
        # with torch.cuda.stream(s):
        #     (g == i).sum()
        #     # print('(g == i).sum() == {}'.format((g == i).sum()))
        #     pred_group = pseudo_label[g == i]
        # print('c_group.shape == {}'.format(c_group.shape))
        # print('pred_group.shape == {}'.format(pred_group.shape))

        # torch.cuda.current_stream().wait_stream(s)
        c_group = c[(g == i)]
        pred_group = pseudo_label[(g == i)]
        # c_group = c.cpu()[(g == i)].cuda()
        # pred_group = pseudo_label.cpu()[(g == i)].cuda()

        mask = (
            torch.eq(pred_group.view(-1, 1), pred_group.view(1, -1)).bool()
        )
        mask = mask.float()

        contrast_count = 1
        contrast_feature = c_group

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), 0.5
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask), 1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(c.device),
        #     0)
        # mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits)  # * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, batch_size).mean()
        loss += instance_loss
        loss_rec[i] += instance_loss.item()
        # visualize_image(x=np.exp([[log_prob.detach().cpu().numpy(), mask.detach().cpu().numpy()]]), verbose=1, show=True, fig_path='./{}.jpg'.format(np.random.random()))
        # print('SelfConsistencyLoss == {}'.format(instance_loss.item()))
    # print('loss_self_cons:', loss_rec)
    return loss
