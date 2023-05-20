import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import distance
import torch.nn.functional as F


def d(p1, p2):
    return distance.cdist(p1, p2, 'euclidean')


def d2(p1, p2):
    # return torch.sqrt( -2 * p1 @ p2.T + np.linalg.norm(p1, axis=1, keepdims=True) + np.linalg.norm(p2, axis=1, keepdims=True).T)
    # return  -2 * p1 @ p2.T + np.linalg.norm(p1, axis=1, keepdims=True) + np.linalg.norm(p2, axis=1, keepdims=True).T
    return torch.sqrt(
        -2 * p1 @ p2.T + torch.sum(p1 ** 2, dim=1, keepdim=True) + torch.sum(p2 ** 2, dim=1, keepdim=True).T)


def d3(p1, p2):
    dis = np.zeros((len(p1), len(p2)))
    for ind, p in enumerate(p2):
        dis[:, ind] = torch.sqrt(torch.sum((p1 - p) ** 2, dim=1))
    return dis


def d4(p1, p2):
    return torch.cdist(p1, p2)


def speed_test():
    t = time.time()
    for f in [d4, d3, d2, d]:
        for _ in range(1000):
            p1 = torch.rand((5, 2))
            p2 = torch.rand((5, 2))
            # p1 = np.random.random((5, 2))
            # p2 = np.random.random((5, 2))
            print(p1)
            print(p2)

            # r = d4(p1, p2)
            # # print(r)
            # r = d3(p1, p2)
            # # print(r)
            # r = d2(p1, p2)
            # print(r)
            # r = d(p1, p2)
            r = f(p1, p2)
            print(r)
            break
        break

        print(time.time() - t)
def dist_test():
    p1 = torch.rand((5, 2))
    p2 = torch.rand((5, 2))
    cd = -torch.cdist(p1, p2)
    print(cd)
    cd = torch.softmax(cd, dim=1)
    print(cd)

def kl(x):
    p = [x, 1 - x]
    q = [0.5, 0.5]
    # q = [x, 1 - x]
    # p = [0.5, 0.5]
    loss = 0
    for i, j in zip(p, q):
        loss += i * np.log(i / j)
    return loss


def main():
    plt.figure()
    x = np.linspace(0, 1, 101)[1:-1]
    # plt.ylim(-1,1)
    # r = [kl(xi) for xi in x]
    # plt.plot(x, r - np.min(r))
    plt.show()


def main2():
    import re, collections
    def get_stats(vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(pair, v_in):
        v_out = {}

        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    vocab = {'l o w </w>': 5, 'l o w e r </w>': 2,
             'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    print(vocab)
    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        print(pairs)
        best = max(pairs, key=pairs.get)
        print(best)
        vocab = merge_vocab(best, vocab)
        print(vocab)
        print()


def random_choose():
    Test = False
    a = np.arange(10 if Test else 100000)
    t = time.time()
    for _ in range(1 if Test else 1000):
        # y = random.sample(list(a), 9 if Test else 10000)
        # 1.4936156272888184

        # y = np.random.choice(a, 9 if Test else 30000, replace=False)
        # 0.16245508193969727 1.681701898574829

        it = np.arange(len(a))
        np.random.shuffle(it)
        y = a[it[:(9 if Test else 30000)]]
        # if Test:
        #     print(y)
        # # 0.1506044864654541 1.5049107074737549
    print(time.time() - t)


def test_kl():
    v = np.asarray([[0.1175, 0.3918, 0.0692, 0.0235, 0.1541, 0.0063, 0.0127, 0.1429, 0.0005,
                     0.0815],
                    [0.0897, 0.3592, 0.0769, 0.0240, 0.1670, 0.0156, 0.0294, 0.1672, 0.0024,
                     0.0686]])
    v2 = (v - 0.1) ** 2
    print(v2)
    print(np.sum(v2, axis=1))


if __name__ == '__main__':
    speed_test()
