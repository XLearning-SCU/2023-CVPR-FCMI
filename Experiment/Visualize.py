import os
import sys
import time

import numpy as np
import torch.functional
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tsnecuda import TSNE as TsneCuda
from Utils import DirectoryOperator

# sys.path.extend(['/xlearning/pengxin/Software/Anaconda/envs/torch171P37/lib'])
# os.execl()
PrintTimer = True


def visualize_plot(x, y, labels, show=False, fig_path=None):
    plt.figure(figsize=(10, 10))
    for xi, yi, label in zip(x, y, labels):
        plt.plot(xi, yi, label=label)
    plt.legend()
    plt.grid()

    if fig_path is not None:
        DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
        plt.savefig(fig_path)
    if show:
        plt.show()
    plt.close()


def visualize_image(x, verbose=0, show=False, fig_path=None):
    """

    :param show:
    :param fig_path:
    :param x:
    (row, line, pic_h, pic_w) or (row, line, pic_h, pic_w, pic_c), pic_c = 1,3,4
    :return:
    """
    x = np.asarray(x)
    if verbose:
        print('img.min() == {}'.format(np.min(x)))
        print('img.max() == {}'.format(np.max(x)))
    x -= np.min(x)
    x /= np.max(x)
    row, line = x.shape[:2]
    plt.figure(figsize=(x.shape[1] * x.shape[3] / 90, x.shape[0] * x.shape[2] / 90))  # w, h

    count = 0
    for rx in x:
        for image in rx:
            count += 1
            plt.subplot(row, line, count)
            plt.imshow(image, cmap='gray', )
            plt.xticks([])
            plt.yticks([])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    if fig_path is not None:
        DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
        plt.savefig(fig_path)
    if show:
        plt.show()
    plt.close()


# Load = False
# Save = not Load
# CountSave = 0


def tsne(x, **kwargs):
    # global CountSave
    t = time.time()
    x = np.asarray(x, dtype=np.float64)
    # if Save:
    #     np.savez('../../np{}'.format(CountSave), x=x)
    #     CountSave += 1
    #     print('save')
    # if Load:
    #     x = np.load('./np.npz')['x']
    # print('len(x) == {}'.format(len(x)))
    # print('start tsne')
    # print(x)
    # print(type(x))
    # print(x.dtype)
    # print(kwargs)
    # y = TSNE(**kwargs, n_jobs=12).fit_transform((x.reshape((len(x), -1))))
    if len(x) < 5000:
        # print(1)
        y = TSNE(**kwargs, metric='cosine', init='pca', learning_rate='auto').fit_transform((x.reshape((len(x), -1))))
        # y = TSNE(**kwargs).fit_transform((x.reshape((len(x), -1))))
        # y = TSNE(**kwargs).fit_transform((x.reshape((len(x), -1))))
    else:
        # print(2)
        y = TsneCuda(**kwargs, metric='innerproduct').fit_transform((x.reshape((len(x), -1))))
        # y = TsneCuda(**kwargs).fit_transform((x.reshape((len(x), -1))))
        # y = TsneCuda(**kwargs).fit_transform((x.reshape((len(x), -1))))
    if PrintTimer:
        # print(3)
        print('TSNE finished with in {:.03f} seconds (x.shape == {}, y.shape == {}).'.format(
            time.time() - t,
            x.shape,
            y.shape
        ))
    return y


DiscriminativeNumber = True


def visualize_scatter(x, label_color=None, label_shape=None, fig_path=None, show=False, **kwargs):
    """

    :param show:
    :param fig_path:
    :param x: (n_samples, 2)
    :param label_color:
    :param label_shape: list of int
    :return:
    """
    t = time.time()

    if label_color is None:
        label_color = [0] * len(x)
    # print('label_color.shape == {}'.format(label_color.shape))
    # print('label_color == {}'.format(label_color))
    vmin = np.min(label_color)
    vmax = np.max(label_color)
    color_num = len(np.unique(label_color))
    if color_num <= 10:
        cmap = 'tab10'
    elif color_num <= 20:
        cmap = 'tab20'
    else:
        vmean = (vmax + vmin) / 2
        vd = (vmax - vmin) / 2
        vmax = vmean + vd * 1.1
        vmin = vmean - vd * 1.2
        cmap = 'gist_ncar'
    if DiscriminativeNumber:
        shape_base = 1
    else:
        shape_base = int(np.log10(color_num - 1 + 0.5)) + 1
    if label_shape is None:
        label_shape = [None] * len(x)
    else:
        label_shape = [
            ('${:0' + '{:d}'.format(shape_base) + 'd}$').format(
                (shape % 10) if DiscriminativeNumber else shape) for shape in label_shape
        ]

    fig_size = np.sqrt(len(x)) * 0.10 * shape_base
    w = np.max(x[:, 0]) - np.min(x[:, 0])
    h = np.max(x[:, 1]) - np.min(x[:, 1])
    k = np.sqrt(w / h)
    left = 0.35
    right = 0.25
    w = fig_size * k + left
    h = fig_size / k + right
    plt.figure(figsize=(w, h))
    # plt.figure()
    number_size = 25 * shape_base

    # if InGroup:
    label_shape = np.asarray(label_shape)
    for it in np.unique(label_shape):
        ind = label_shape == it
        plt.scatter(x[ind, 0], x[ind, 1],
                    c=label_color[ind], vmin=vmin, vmax=vmax, cmap=cmap,
                    marker=it, s=number_size,
                    )
    # else:
    #     for i in range(len(x)):
    #         plt.scatter(x[i, 0], x[i, 1],
    #                     c=label_color[i], vmin=vmin, vmax=vmax, cmap='tab20',
    #                     marker=label_shape[i], s=number_size,
    #                     )
    plt.subplots_adjust(left=left / w, right=1, top=1, bottom=right / h)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.grid()
    if fig_path is not None:
        DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
        plt.savefig(fig_path)
    if show:
        plt.show()
    plt.close()
    if PrintTimer:
        print('VisualizeScatter finished with in {:.03f} seconds (x.shape == {}).'.format(
            time.time() - t,
            x.shape,
        ))


def visual_matrix_console(x):
    if len(x.shape) <= 2:
        x = x.reshape((*x.shape, 1))
    base_wid = int(np.log10(np.max(x) + 0.5)) + 1
    head_wid = x.shape[2] * (1 + base_wid)
    head_sep = int(head_wid // 2) + 1
    print('t\\c ', end='')
    for i in range(x.shape[1]):
        print(('{:' + '{}'.format(head_sep) + 'd}').format(i), end=' ' * (head_wid - head_sep))
    print()
    for i, line in enumerate(x):
        print('{:2d}: '.format(i), end='')
        for cl in line:
            sg = True
            for g in cl:
                if sg:
                    sg = False
                else:
                    print(' ', end='')
                if g != 0:
                    # print('base_wid == {}'.format(base_wid))
                    # print('g == {}'.format(g))
                    print(('{:' + str(base_wid) + 'd}').format(g), end='')
                else:
                    print(' ' * base_wid, end='')
            print('|', end='')
        print()


def main():
    UseTsne = True
    # n = 2
    # n = 100
    # 2.089 0.325

    # n = 1000
    # 3.538 0.372

    # n = 4000
    # n = 5000
    # 10.064 0.829

    n = 10000
    # 19.424 1.425
    # 10.347

    # n = 30000
    # 69.801 4.415
    # 10.341

    # n = 100000
    # 10.876 11.9

    # n = 150000
    # 12.365 14.980
    clu = 10
    dim = 10
    x = np.random.random((n, dim if UseTsne else 2))
    # x += t.reshape((-1, 1)) * 0.41

    t = np.arange(n) % clu
    for i, j in enumerate(t):
        x[i, int(j // dim)] += 1
        x[i, int(j % dim)] += 1
    # print(x)
    x = x / np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    # print(x)
    label_color = t
    label_shape = t
    if UseTsne:
        x = tsne(x)
    ind = len(x) - np.arange(len(x)) - 1
    visualize_scatter(
        x[ind],
        label_color=label_color[ind],
        label_shape=label_shape[ind],
        # fig_path='../c.png',
        show=True,
    )


def main2():
    visualize_image(x=np.random.random((3, 50, 28, 28, 1)), show=True)


def main3():
    def kl(p, q):
        return np.sum([pp * np.log(pp / qq) for pp, qq in zip(p, q)])

    x = np.linspace(0, 1, 51)[1:-1]
    qx = [0.2, 0.8]
    visualize_plot(
        x=[
            x, x, x
        ],
        y=[
            [kl([xx, 1 - xx], qx) for xx in x],
            [kl(qx, [xx, 1 - xx]) for xx in x],
            [(kl(qx, [xx, 1 - xx]) + kl([xx, 1 - xx], qx)) / 2 for xx in x],
        ],
        labels=['1', '2', '3'],
        fig_path='../c.png',
    )


def main_vis():
    data_f = np.load('/xlearning/pengxin/Temp/Python/Visualization_BAc/Visualnp50.npz')

    feature_vec = data_f['feature_vec']
    type_vec = data_f['type_vec']
    group_vec = data_f['group_vec']
    pred_vec = data_f['pred_vec']
    epoch = data_f['epoch']

    p = 20
    lr = 50
    vis_fea = tsne(feature_vec, perplexity=p, verbose=1, learning_rate=lr, early_exaggeration=6)
    visualize_scatter(vis_fea,
                      fig_path='../../Visualization2/E{:03d}P{:02d}Type.jpg'.format(epoch, p),
                      label_color=type_vec,
                      label_shape=type_vec,
                      )
    visualize_scatter(vis_fea,
                      fig_path='../../Visualization2/E{:03d}P{:02d}Cluster.jpg'.format(epoch, p),
                      label_color=pred_vec,
                      label_shape=type_vec,
                      )
    visualize_scatter(vis_fea,
                      fig_path='../../Visualization2/E{:03d}P{:02d}Group.jpg'.format(epoch, p),
                      label_color=group_vec,
                      label_shape=type_vec,
                      )


if __name__ == '__main__':
    # print(1-0.005**0.1)
    main_vis()
