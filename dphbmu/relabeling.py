import numpy as np
from scipy.stats import norm
import itertools


def zz_to_P(zz):
    N = len(zz)
    K = len(np.unique(zz))
    P = np.zeros((N, K), dtype = int)
    for i in range(N):
        P[i, zz[i]] = int(1)
    return P


def P_to_zz(P):
    return np.argmax(P, axis = 1)


def relabelling(zz, mu, max_iter = 10):
    '''
    Relabeling algorithm in Stephens (2000).
    '''
    zz, mu = zz.copy(), mu.copy()
    K = len(mu[0])
    assign = np.array([zz_to_P(zz[i]) for i in range(len(zz))])
    print('(Relabelling)')
    for i in range(max_iter):
        Q = assign.mean(axis = 0)
        count = 0
        for n in range(len(assign)):
            klds, inds = [], []
            for ind in itertools.permutations(range(K)):
                P = assign[n][:,ind]
                klds.append(-np.log(Q[P == 1] + 1e-12).sum())  # Kullback-Leibler Div.
                inds.append(ind)
            ind = inds[np.argmin(np.array(klds))]
            if not (ind == np.arange(K)).all():
                count += 1
            assign[n] = assign[n][:,ind]
            mu[n] = mu[n][ind,:]
        print(f'#iter {i}: num. occurence = {count}')
        if count == 0:
            break
    zz = np.array([P_to_zz(assign[i]) for i in range(len(assign))])
    return zz, mu