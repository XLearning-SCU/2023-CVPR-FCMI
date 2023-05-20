import os
import time

import torch
# from tqdm import tqdm
from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc

from Experiment.Visualize import PrintTimer
from Utils.DirectoryOperator import FileOperator

import torch.nn.functional as F


def inference(net, test_dataloader):
    net.eval()
    feature_vec, type_vec, group_vec, pred_vec = [], [], [], []
    with torch.no_grad():
        for (x, g, y, idx) in test_dataloader:
            x = x.cuda()
            # x = x.view(x.shape[0], -1)
            g = g.cuda()
            c = net.encode(x, g).detach()
            pred = torch.argmax(c, dim=1)
            feature_vec.extend(c.cpu().numpy())
            type_vec.extend(y.cpu().numpy())
            group_vec.extend(g.cpu().numpy())
            pred_vec.extend(pred.cpu().numpy())
    feature_vec, type_vec, group_vec, pred_vec = np.array(
        feature_vec), np.array(type_vec), np.array(group_vec), np.array(
        pred_vec)
    # tqdm.write("Finished with features shape {}".format(feature_vec.shape))
    # idx, counts = torch.unique(torch.from_numpy(pred_vec), return_counts=True)
    # tqdm.write('%d clusters assigned with cluster distribution:' %
    #            (counts.shape[0]),
    #            end=' ')
    # tqdm.write(' '.join(str(ct) for ct in counts.numpy()))
    kmeans = KMeans(n_clusters=len(set(type_vec))).fit(feature_vec)
    pred_vec = kmeans.labels_
    net.train()

    return feature_vec, type_vec, group_vec, pred_vec


def evaluate(feature_vec, pred_vec, type_vec, group_vec, best_acc, fair_metric=False):
    # tqdm.write("Evaluating the clustering results...")
    # print('Evaluating the clustering results... ')
    nmi, ari, acc, pred_adjusted = cluster_metrics(type_vec, pred_vec)
    if best_acc < acc:
        best_acc = acc
    print('nmi={:5.02f}, ari={:5.02f}, acc={:5.02f}, BestAcc={:5.02f}'.format(
        nmi * 100, ari * 100, acc * 100, best_acc * 100))
    # tqdm.write('NMI=%.4f, ACC=%.4f, ARI=%.4f' % (nmi, acc, ari), end='')
    if fair_metric:
        kl, ari_b = fair_metrics(feature_vec, group_vec, pred_vec, type_vec)
        print(', KL=%.4f, ARI_b=%.4f' % (kl, ari_b), end='')
    # tqdm.write('')
    return best_acc


BestAcc = 0.
BestAri = 0.
BestNmi = 0.
BestBalance = 0.
BestFairness = 0.
BestNmiFair_avg = 0.
BestNmiFair = 0.
BestFmeasure = 0.
BestEntropy = 0.


def calculate_entropy(input):
    """
    calculates the entropy score
    :param input: tensor for which entropy needs to be calculated
    """
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=0)
    return entropy


def calculate_balance(predicted, size_0, k=10):
    """
    calculates the balance score of a model output_0
    :param predicted: tensor with the model predictions (n_samples_g0+n_samples_g1, )V(0:cluster)
    :param size_0: size of sample
    :param k: amount of clusters
    """
    count = torch.zeros((k, 2))
    # cluster, grounp
    for i in range(size_0):
        count[predicted[i], 0] += 1
    for i in range(size_0, predicted.shape[0]):
        count[predicted[i], 1] += 1

    count[count == 0] = 1e-5
    print(count)
    balance_0 = torch.min(count[:, 0] / count[:, 1])
    balance_1 = torch.min(count[:, 1] / count[:, 0])
    # print(balance_1)
    en_0 = calculate_entropy(count[:, 0] / torch.sum(count[:, 0]))
    en_1 = calculate_entropy(count[:, 1] / torch.sum(count[:, 1]))

    return min(balance_0, balance_1).numpy(), en_0.numpy(), en_1.numpy()


def my_balance(predicted, g, cluster_num, group_num):
    """

    :param predicted:
    :param g:
    :param cluster_num:
    :param group_num:
    :return: balance, entro: entropy of the proportion of the size of the clusters of i-th group.
    """
    count = torch.zeros((cluster_num, group_num))
    # cluster, grounp
    # count_time = 0
    for ci, gi in zip(predicted, g):
        # count_time += 1
        # print(ci, gi)
        # print(ci)
        # print(gi)
        count[ci, gi] += 1
        # print(count_time, count)
    # print(count_time)
    # print(predicted.shape)
    # print(g.shape)
    # print(count)

    count[count == 0] = 1e-5
    balance_v = torch.amin(
        torch.amin(count, dim=1) / torch.amax(count, dim=1)
    )
    # print(balance_v)

    epsilon = 1e-5
    prob = count / torch.sum(count, dim=0, keepdim=True)
    entro = torch.sum(-prob * torch.log(prob + epsilon), dim=0)

    # print(entro)

    return balance_v.numpy(), entro.numpy()


# if __name__ == '__main__':
#     print('[{}]'.format(1))
# predicted = torch.arange(100) % 3
# g =torch.as_tensor( torch.arange(100) > 50, dtype=torch.int)
# print(predicted.shape)
# my_balance(predicted.cuda(), g.cuda(), cluster_num=3, group_num=2)
# print(predicted.shape)
# b, e0, e1 = balance(predicted=predicted, size_0=(g==0).sum(), k=3 )
# print(b, e0, e1)
def normalized_mutual_information_without_mean(o):
    """

    :param o: o_cg = p_cg
    :return: 1-NMI(c,g)
    """

    def entroph(v):
        """

        :param v: element-wise entroph
        :return:
        """
        # print(v)
        # print(torch.log(v))
        # print(torch.log(v) * v)
        # print(-torch.sum(torch.log(v) * v))
        return -torch.sum(torch.log(v) * v)

    hc = entroph(torch.sum(o, dim=1))
    hg = entroph(torch.sum(o, dim=0))
    hclg = -torch.sum(torch.log(o / torch.sum(o, dim=0, keepdim=True)) * o)
    icg = hc - hclg
    # nmi = icg / ((hc + hg) / 2)
    nmi = icg / hc
    # print('o == {}'.format(o))
    # print('icg == {}'.format(icg))
    # print('hclg == {}'.format(hclg))
    # print('hc == {}'.format(hc))
    # print('hg == {}'.format(hg))
    # print('nmi == {}'.format(nmi))
    return 1 - nmi


def normalized_mutual_information(o):
    """

    :param o: o_cg = p_cg
    :return: 1-NMI(c,g)
    """

    def entroph(v):
        """

        :param v: vector or matrix(trait to be vector)
        :return:element-wise entroph
        """
        # print(v)
        # print(torch.log(v))
        # print(torch.log(v) * v)
        # print(-torch.sum(torch.log(v) * v))
        # vv = v[v!=0]
        return -torch.sum(torch.log((v + 1e-10 * (v == 0))) * v)

    # hc = entroph(torch.sum(o, dim=1))
    hg = entroph(torch.sum(o, dim=0))
    # pg = torch.sum(o, dim=0, keepdim=True)
    # pclg = o / (pg + 1e-10 * (pg == 0))
    # hclg = -torch.sum(torch.log(pclg + 1e-10 * (pclg == 0)) * o)

    pc = torch.sum(o, dim=1, keepdim=True)
    pglc = o / (pc + 1e-10 * (pc == 0))
    pglc[torch.sum(pglc, dim=1, keepdim=False) == 0] = 1 / len(pglc[0])
    hglc_c = -torch.sum(torch.log(pglc + 1e-10 * (pglc == 0)) * pglc, dim=1)
    # hglc = torch.sum(o,  dim=1) @ hglc_c
    # icg = hc - hclg
    # icg = hg - hglc
    # nmi = icg / ((hc + hg) / 2)
    # hclx = 0.088141
    # icx = hc - hclx
    # nmi = icg / hg
    # print('o == {}'.format(o))
    # print('icg == {}'.format(icg))
    # print('hclg == {}'.format(hclg))
    # print('hglc == {}'.format(hglc))
    # print('hc == {}'.format(hc))
    # print('hg == {}'.format(hg))
    # print('nmi == {}'.format(nmi))
    # print()
    # return 1-nmi
    return torch.min(hglc_c) / hg


class FMeasure:
    def __init__(self, beta=1):
        """

        :param beta: r/p = beta, s.t. dr = dp
        """
        self.beta = beta

    def __call__(self, p, r):
        return (self.beta ** 2 + 1) * (p * r) / (self.beta ** 2 * p + r)


def relative_fairness(O, E):
    # print('O / E', O / E)
    # print('torch.log(O / E)', torch.log(O / E))
    # print('O * torch.log(O / E)', O * torch.log(O / E))
    # print('torch.sum(O * torch.log(O / E))', torch.sum(O * torch.log(O / E)))
    return 1 - torch.sum(O * torch.log(O / E))


def evaluate2(feature_vec, pred_vec, type_vec, group_vec):
    # tqdm.write("Evaluating the clustering results...")
    # print('Evaluating the clustering results... ')
    nmi, ari, acc, pred_adjusted = cluster_metrics(type_vec, pred_vec)
    if group_vec is not None:
        balance, entro = my_balance(pred_vec, group_vec, cluster_num=np.unique(type_vec).shape[0],
                                    group_num=np.unique(group_vec).shape[0])
    else:
        balance, entro = 0, 0
    entro_v = np.mean(entro)

    gs = np.unique(group_vec)
    ts = np.unique(type_vec)
    class_num = len(ts)
    group_num = len(gs)
    O = torch.zeros((class_num, group_num)).cuda()
    E = torch.zeros((class_num, group_num)).cuda()

    for b in gs:
        ind_g = b == group_vec
        pred_vec_g = pred_vec[ind_g]
        type_vec_g = type_vec[ind_g]
        for t in ts:
            O[t, b] = np.sum(pred_vec_g == t)
            E[t, b] = np.sum(type_vec_g == t)
    E += 1e-3
    O += 1e-6
    # fairness = F.kl_div(
    #     torch.log((O / torch.sum(O)).view((1, -1))),
    #     (E / torch.sum(E)).view((1, -1)),
    #     reduction="batchmean")
    # print(fairness)

    O = (O / torch.sum(O))
    E = (E / torch.sum(E))
    # print(O)
    # print(E)

    fairness = relative_fairness(O, E)
    NmiFair = normalized_mutual_information(O)
    NmiFair_avg = normalized_mutual_information_without_mean(O)
    Fmeasure = FMeasure(beta=1)(nmi, NmiFair)
    # Fmeasure = FMeasure(beta=1)(acc, NmiFair)
    global BestAcc, BestAri, BestNmi, BestBalance, BestEntropy, BestFairness, BestNmiFair, BestNmiFair_avg, BestFmeasure
    if BestAcc < acc:
        BestAcc = acc
    if BestAri < ari:
        BestAri = ari
    if BestNmi < nmi:
        BestNmi = nmi
    if BestBalance < balance:
        BestBalance = balance
    if BestFairness < fairness:
        BestFairness = fairness
    if BestNmiFair < NmiFair:
        BestNmiFair = NmiFair
    if BestNmiFair_avg < NmiFair_avg:
        BestNmiFair_avg = NmiFair_avg
    if BestFmeasure < Fmeasure:
        BestFmeasure = Fmeasure
    if BestEntropy < entro_v:
        BestEntropy = entro_v

    # print(', '.join([
    #     '{}={:5.02f}|{:5.02f}'.format(metric_name, val * 100, best_val * 100) for metric_name, val, best_val in [
    #         ['ACC', acc, BestAcc, ],
    #         ['NMI', nmi, BestNmi, ],
    #         ['Bal', balance, BestBalance, ],
    #         ['MNCE', NmiFair, BestNmiFair, ],
    #         ['Fmeasure', Fmeasure, BestFmeasure, ],
    #     ]
    # ]))
    print(', '.join([
        '{}={:5.01f}|{:5.03f}'.format(metric_name, val * 100, best_val * 100) for metric_name, val, best_val in [
            ['ACC', acc, BestAcc, ],
            ['NMI', nmi, BestNmi, ],
            ['Bal', balance, BestBalance, ],
            ['MNCE', NmiFair, BestNmiFair, ],
            ['Fmeasure', Fmeasure, BestFmeasure, ],
        ]
    ]))
    # print(
    #     'NMI={:5.02f}|{:5.02f}, ARI={:5.02f}|{:5.02f}, ACC={:5.02f}|{:5.02f}, Balance={:5.02f}|{:5.02f}, MNCE_avg={:5.02f}|{:5.02f},MNCE_min={:5.02f}|{:5.02f}, Fmeasure={:5.02f}|{:5.02f}, Entropy={:5.02f}|{:5.02f}[{}],'.format(
    #         nmi * 100, BestNmi * 100,
    #         ari * 100, BestAri * 100,
    #         acc * 100, BestAcc * 100,
    #         balance * 100, BestBalance * 100,
    #         # fairness * 100, BestFairness * 100,
    #         NmiFair_avg * 100, BestNmiFair_avg * 100,
    #         NmiFair * 100, BestNmiFair * 100,
    #         Fmeasure * 100, BestFmeasure * 100,
    #         entro_v, BestEntropy, entro
    #     )
    # )
    return pred_adjusted
    # tqdm.write('NMI=%.4f, ACC=%.4f, ARI=%.4f' % (nmi, acc, ari), end='')
    # if fair_metric:
    #     kl, ari_b = fair_metrics(feature_vec, group_vec, pred_vec, type_vec)
    #     print(', KL=%.4f, ARI_b=%.4f' % (kl, ari_b), end='')
    # tqdm.write('')


def cluster_metrics(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(label, pred_adjusted)
    # acc = 0
    return nmi, ari, acc, pred_adjusted


def fair_metrics(feature_vec, batch_vec, pred_vec, type_vec):
    pass


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(y_true,
                                                cluster_assignments,
                                                labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    pred_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = pred_to_true_cluster_labels[cluster_assignments]
    return np.asarray(y_pred, dtype=int)


def UMAP(feature_vec, type_vec, group_vec, pred_vec, n_type, n_batch, args, epoch):
    t = time.time()
    # print("Performing UMAP Visualization...")
    # print('feature_vec.shape == {}'.format(feature_vec.shape))
    sc.set_figure_params(figsize=(4, 4), dpi=300)

    # type_vec = pd.DataFrame(type_vec)
    # for key in cell_type_dict.keys():
    #     type_vec.replace(key, cell_type_dict[key], inplace=True)
    # group_vec = pd.DataFrame(group_vec)
    # for key in batch_dict.keys():
    #     batch_vec.replace(key, batch_dict[key], inplace=True)

    adata = sc.AnnData(feature_vec)
    # print('adata.shape == {}'.format(adata.shape))
    sc.pp.neighbors(adata)
    adata.obs['cluster'] = pd.DataFrame(pred_vec).values.astype(np.str_)
    adata.obs['type'] = pd.DataFrame(type_vec).values.astype(np.str_)
    adata.obs['group'] = pd.DataFrame(group_vec).values.astype(np.str_)

    sc.tl.umap(adata)
    sc.pl.umap(adata,
               color=['cluster'],
               palette=sns.color_palette("husl", n_type),
               save='E{:03d}UmapCluster{}.png'.format(epoch, str(args.dataset)),
               show=False)
    sc.pl.umap(adata,
               color=['type'],
               palette=sns.color_palette("husl", n_type),
               save='E{:03d}UmapType{}.png'.format(epoch, str(args.dataset)),
               show=False)
    sc.pl.umap(adata,
               color=['group'],
               palette=sns.color_palette("hls", n_batch),
               save='E{:03d}UmapGroup{}.png'.format(epoch, str(args.dataset)),
               show=False)
    roott = './figures/'
    for root, dirs, files in os.walk(roott):
        # print(root)
        # print(dirs)
        # print(files)
        for f in files:
            # print(os.path.join('../Visualization', f))
            FileOperator(
                os.path.join(root, f)
            ).rename(
                os.path.join('../Visualization', f.replace('umapE', 'E')),
                auto_rename=True
            )
    if PrintTimer:
        print('VisualizeScatter finished with in {:.03f} seconds (x.shape == {}).'.format(
            time.time() - t,
            feature_vec.shape,
        ))


def main():
    # pred_vec = (np.arange(300) + 10) % 30
    # pred_vec = (np.arange(10)+10)%3
    # type_vec = np.arange(300) % 30
    # group_vec = np.arange(300) % 2
    # relative_fairness()

    # data = np.load('/mnt/18t/pengxin/Codes/0412/RunSet-0419_Load/Office_NetShuffleLikGDecFeaTanActSigWarmAll20InfoBalanceLoss0.05InfoFairLoss0.20Tb0.10OneHot0.05Th0.10_G0B0512torch1110/NpPoints/Np001.npz')
    data = np.load('/mnt/18t/pengxin/Checkpoints/FairClustering/FCMI/SotaNp/MouseAtlas/Np079.npz')
    feature_vec = data['feature_vec']
    type_vec = data['type_vec']
    group_vec = data['group_vec']
    pred_vec = data['pred_vec']
    epoch = data['epoch']
    evaluate2(None, pred_vec, type_vec, group_vec)
    # print(nmi)
    # roott = './figures/umap/'
    # for root, dirs, files in os.walk(roott):
    #     for f in files:
    #         print(os.path.join('../Visualization', f))
    #         FileOperator(
    #             os.path.join(root, f)
    #         ).rename(
    #             os.path.join('../Visualization', f)
    #         )


if __name__ == '__main__':
    main()
