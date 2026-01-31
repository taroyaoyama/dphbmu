import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gammaln


def _stable_softmax2(loga, logb):
    m = max(loga, logb)
    a = np.exp(loga - m)
    b = np.exp(logb - m)
    s = a + b
    return a / s, b / s


def lp_x_given_others(x, xx, tu, mu0, rh0):
    '''
    Log. probability of x conditional on other x's in a cluster sharing class label k.
    '''
    n, D = xx.shape[0], x.shape[0]
    if n == 0:
        mvn = multivariate_normal(
            mean = mu0, cov = (1 + 1 / rh0) / tu * np.eye(D)
        )
        return mvn.logpdf(x)
    x_bar = np.mean(xx, axis = 0)
    mvn = multivariate_normal(
        mean = (rh0 * mu0 + n * x_bar) / (rh0 + n),
        cov = (1 + 1 / (rh0 + n)) / tu * np.eye(D)
    )
    return mvn.logpdf(x)


def lp_xx(xx, tu, mu0, rh0):
    '''
    Marginal likelihood for a cluster {x_n}_n
    '''
    n, D = xx.shape
    xbar = np.mean(xx, axis = 0)
    sse = np.sum((xx - xbar)**2)
    lp = 0
    lp += 0.5 * D * (np.log(rh0) - np.log(n + rh0) + n * np.log(tu) - n * np.log(2 * np.pi))
    lp += -0.5 * tu * sse
    lp += -0.5 * tu * (rh0 * n) / (rh0 + n) * np.sum((xbar - mu0)**2)
    return lp


def launch_split_state(i, j, S, rng = None):
    '''
    Randomly initiate split state.
    '''
    if rng is None:
        rng = np.random.default_rng()

    A, B = [i], [j]
    for k in S:
        if k == i or k == j:
            continue
        if rng.uniform() < 0.5:
            A.append(k)
        else:
            B.append(k)
    return A, B


def restricted_gibbs_split_collapsed(i, j, A, B, xx, tu, mu0, rh0, n_scans = 1, rng = None):
    """
    Restricted Gibbs scans for split using collapsed predictive only (no mu).
    Returns A, B, log_q (forward proposal log prob for assignments).
    """
    eps = 1e-300
    if rng is None:
        rng = np.random.default_rng()

    # whole set (including i & j)
    S = np.array(sorted(set(A) | set(B)))

    # run gibbs scans
    log_q = 0.0
    for _ in range(n_scans):
        for k in S:
            if k == i or k == j:
                continue

            # remove k from current side
            if k in A:
                A.remove(k)
            else:
                B.remove(k)

            # compute weights
            log_pa = np.log(len(A)) + lp_x_given_others(xx[k], xx[A], tu, mu0, rh0)
            log_pb = np.log(len(B)) + lp_x_given_others(xx[k], xx[B], tu, mu0, rh0)
            pa, pb = _stable_softmax2(log_pa, log_pb)

            # assign
            if rng.uniform() < pa:
                A.append(k)
                log_q += np.log(pa + eps)
            else:
                B.append(k)
                log_q += np.log(pb + eps)

    return A, B, log_q


def lp_restricted_gibbs_split_collapsed(i, j, A_tar, B_tar, A_lau, B_lau, xx, tu, mu0, rh0):
    """
    Log probability of producing (A_tar, B_tar) by one collapsed restricted Gibbs scan,
    starting from launch partition (A_lau, B_lau).
    """
    eps = 1e-300
    A_lau, B_lau = list(A_lau), list(B_lau)

    # whole set
    S = np.array(sorted(set(A_lau) | set(B_lau)))

    # aggregate log prob.
    log_q = 0.0
    for k in S:
        if k == i or k == j:
            continue

        if k in A_lau:
            A_lau.remove(k)
        else:
            B_lau.remove(k)

        log_pa = np.log(len(A_lau)) + lp_x_given_others(xx[k], xx[A_lau], tu, mu0, rh0)
        log_pb = np.log(len(B_lau)) + lp_x_given_others(xx[k], xx[B_lau], tu, mu0, rh0)
        pa, pb = _stable_softmax2(log_pa, log_pb)

        if k in A_tar:
            A_lau.append(k)
            log_q += np.log(pa + eps)
        else:
            B_lau.append(k)
            log_q += np.log(pb + eps)

    return log_q


def _clean_up_zz(zz):
    uniq = np.unique(zz)
    mapping = {old: new for new, old in enumerate(uniq)}
    return np.array([mapping[z] for z in zz])


def update_zz_split_merge_collapsed(zz, xx, tu, mu0, rh0, alp, rng = None):
    """
    Split & Merge DPMM sampler (Jain & Neal, 2004), fully collapsed w.r.t. mu.
    """
    if rng is None:
        rng = np.random.default_rng()
    zz = zz.copy()

    # randomly select two indices
    i, j = rng.choice(len(zz), size = 2, replace = False)
    zi, zj = zz[i], zz[j]

    if zi == zj:

        # ---------------------
        # split procedure
        # ---------------------

        S = np.where(zz == zi)[0]
        n = len(S)
        if n <= 2:
            return zz

        # launch split state
        A, B = launch_split_state(i, j, S, rng = rng)

        # warm-up scan (optional, keeps your structure)
        A, B, _ = restricted_gibbs_split_collapsed(i, j, A, B, xx, tu, mu0, rh0, n_scans = 1, rng = rng)

        # proposal: one final scan
        A, B, lq_fwd = restricted_gibbs_split_collapsed(i, j, A, B, xx, tu, mu0, rh0, n_scans = 1, rng = rng)
        nA, nB = len(A), len(B)
        if nA == 0 or nB == 0:
            return zz

        # likelihoods
        lp_S = lp_xx(xx[S], tu, mu0, rh0)
        lp_A = lp_xx(xx[A], tu, mu0, rh0)
        lp_B = lp_xx(xx[B], tu, mu0, rh0)

        # reverse proposal for merge: deterministic (q_rev = 1)
        lq_rev = 0.0

        # acceptance ratio
        log_acc = 0.0
        log_acc += np.log(alp) + gammaln(nA) + gammaln(nB) - gammaln(n)  # CRP prior ratio
        log_acc += lq_rev - lq_fwd
        log_acc += lp_A + lp_B - lp_S

        if np.log(rng.uniform()) < log_acc:
            zz_new = zz.copy()
            zz_new[A] = zi
            zz_new[B] = zz.max() + 1
            return _clean_up_zz(zz_new)
        else:
            return zz

    else:

        # ---------------------
        # merge procedure
        # ---------------------

        A = np.where(zz == zi)[0]
        B = np.where(zz == zj)[0]
        nA, nB = len(A), len(B)

        S = np.array(sorted(set(A) | set(B)))
        n = len(S)

        # likelihoods
        lp_S = lp_xx(xx[S], tu, mu0, rh0)
        lp_A = lp_xx(xx[A], tu, mu0, rh0)
        lp_B = lp_xx(xx[B], tu, mu0, rh0)

        # forward proposal for merge: deterministic (q_fwd = 1)
        lq_fwd = 0.0

        # launch split state for reverse proposal
        A_lau, B_lau = launch_split_state(i, j, S, rng = rng)

        # warm-up scan (optional)
        A_lau, B_lau, _ = restricted_gibbs_split_collapsed(i, j, A_lau, B_lau, xx, tu, mu0, rh0, n_scans = 1, rng = rng)

        # reverse proposal density: probability that one scan produces the *current* (A,B)
        lq_rev = lp_restricted_gibbs_split_collapsed(i, j, A, B, A_lau, B_lau, xx, tu, mu0, rh0)

        # acceptance ratio
        log_acc = 0.0
        log_acc += gammaln(n) - np.log(alp) - gammaln(nA) - gammaln(nB)
        log_acc += lq_rev - lq_fwd
        log_acc += lp_S - lp_A - lp_B

        if np.log(rng.uniform()) < log_acc:
            zz_new = zz.copy()
            zz_new[B] = zi
            return _clean_up_zz(zz_new)
        else:
            return zz