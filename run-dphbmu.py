import numpy as np
from scipy.stats import norm

import os
import pickle
from copy import deepcopy

from dphbmu.conditionals import *
from dphbmu.splitmerge_collapsed import *
from dphbmu.relabeling import *

from frame2d.smf_3story2bay import smf_3story2bay

# ------------------------------
# util. functions
# ------------------------------

def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

def logit(p):
    EPS = 1e-8
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log(1.0 - p)

# ------------------------------
# define simulator
# ------------------------------

simulator = lambda x: smf_3story2bay(norm.cdf(x))[4:]

def create_obs(ne, sg, rng):
    '''
    For creating synthetic data.
    '''
    uo_1 = np.array([0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90])
    uo_2 = np.array([0.40, 0.70, 0.60, 0.70, 0.90, 0.80, 0.90, 0.90, 0.90])
    uo_3 = np.array([0.20, 0.30, 0.30, 0.50, 0.60, 0.60, 0.70, 0.80, 0.80])
    uo = np.concatenate([
        np.tile(uo_1, (ne, 1)),
        np.tile(uo_2, (ne, 1)),
        np.tile(uo_3, (ne, 1)),
    ])
    uo = uo + norm(0, 0.02).rvs(uo.shape, random_state = rng)
    xo = norm.ppf(uo)
    yo = np.array([simulator(x) for x in xo])
    yo += norm(0, sg).rvs(yo.shape, random_state = rng)
    return xo, yo


if __name__ == '__main__':

    # ------------------------------
    # synthetic data
    # ------------------------------

    # hyperparameters for synthetic data
    sg = 0.10
    ne = 5

    # create observations
    xo, yo = create_obs(ne, sg, 123)

    # ------------------------------
    # hyperparameters for DP-HBMU 
    # ------------------------------

    # hyperparameters for base distribution
    D = 9
    mu0 = norm.ppf(np.array([0.5] * D))
    rh0 = 0.05

    # hyperparameters for gamma distributions
    a0g = 2.0
    b0g = a0g / 0.10**-2
    a0t = 2.0
    b0t = a0t / 0.10**-2
    a0l = 1.0
    b0l = a0l / 1.00

    # hyperparameters for pCN stepsize adaptation
    arate_tar = 0.8
    n_win = 50
    v = 0.6

    # num. of iterations
    n_iter = 20000
    n_burn = 5000

    # ------------------------------
    # RUN DP-HBMU Sampler!
    # ------------------------------

    # init. random number generator
    rng = np.random.default_rng(456)

    # data slots
    xxs, zzs, als, mus, tus, gms = [], [], [], [], [], []
    acs, scs = [], []  # for acceptance rate & scales

    # initialize
    xx = norm(0, 1).rvs(
        size = (len(yo), D),
        random_state = rng
    )
    yy = np.array([simulator(x) for x in xx])
    zz = np.zeros(len(yo)).astype(int)
    tu = a0t / b0t
    gm = a0g / b0g
    al = a0l / b0l
    k_ini = 1
    mu = np.array([sample_mu(tu, mu0, rh0, rng = rng) for _ in range(k_ini)])

    # initialize the pCN scale
    lam = logit(0.5)
    scale = np.full(len(xx), sigmoid(lam))

    for i in range(n_iter):
            
        # metropolis-within-gibbs rule
        zz = update_zz_split_merge_collapsed(zz, xx, tu, mu0, rh0, al, rng = rng)
        al = update_al(al, len(np.unique(zz)), len(zz), a0l, b0l, rng = rng)
        mu = update_mu(zz, xx, tu, mu0, rh0, rng = rng)
        tu = update_tu(zz, xx, mu0, rh0, a0t, b0t, rng = rng)
        xx, yy, accepts = update_xx_pcn(simulator, zz, xx, yy, yo, gm, mu, tu, scale, rng = rng)
        gm = update_gm(yy, yo, a0g, b0g, rng = rng)

        # append
        als.append(deepcopy(al))
        zzs.append(deepcopy(zz))
        xxs.append(deepcopy(xx))
        mus.append(deepcopy(mu))
        tus.append(deepcopy(tu))
        gms.append(deepcopy(gm))
        acs.append(deepcopy(accepts))
        scs.append(deepcopy(scale))

        # tuning stepsize in pCN
        ar = np.array(acs[-np.min((len(acs), n_win)):]).mean()
        if i >= n_win and i < n_burn:
            zet = (i + 1)**-v
            lam = lam + zet * (ar - arate_tar)
            scale = np.full(len(xx), sigmoid(lam))
        
        if i % 20 == 0:
            print(
                f'Iter {i:05}: zz =', zz,
                f'| acc.rate = {ar:.3f}',
            )

    # ------------------------------
    # relabeling!
    # ------------------------------

    # number of clusters
    classnums = np.array([len(m) for m in mus])

    # posterior draws conditional on K_hat (= most probable number of clusters)
    K_hat = 3
    zzr = np.array(zzs[n_burn:])[classnums[n_burn:] == K_hat]
    mur = np.array([mus[i] for i in range(n_burn, len(mus)) if classnums[i] == K_hat])

    # relabeling
    zzr, mur = relabelling(zzr, mur)

    # --------------------
    # save
    # --------------------

    dic = {
        'observation': {
            'sg': sg,
            'ne': ne,
            'xo': xo,
            'yo': yo,
        },
        'posterior': {
            'zzs': np.array(zzs),
            'xxs': np.array(xxs),
            'mus': mus,
            'als': np.array(als),
            'gms': np.array(gms),
            'tus': np.array(tus),
            'classnums': classnums
        },
        'relabeling': {
            'zzs': np.array(zzr),
            'mus': np.array(mur)
        },
        'hyper_step': {
            'arate_tar': arate_tar, 'n_win': n_win, 'v': v,
        },
        'hyper_mcmc': {
            'mu0': mu0,
            'rh0': rh0,
            'a0g': a0g,
            'b0g': b0g,
            'a0t': a0t,
            'b0t': b0t,
            'a0l': a0l,
            'b0l': b0l,
        },
        'n_burn': n_burn,
        'acceptance': np.array(acs),
        'scales': np.array(scs)
    }

    out_dir = 'result/'
    os.makedirs(out_dir, exist_ok = True)
    filepath = os.path.join(out_dir, 'posterior.pickle')
    with open(filepath, mode='wb') as f:
        pickle.dump(dic, f)