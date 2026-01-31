import numpy as np
from scipy.stats import (
    norm, gamma, uniform, multivariate_normal, beta, bernoulli
)


def log_prior(x, mu, tu):
    return multivariate_normal(mean = mu, cov = np.eye(len(mu)) / tu).logpdf(x)


def log_likelihood(y, yo, gm):
    return norm(y, gm**-0.5).logpdf(yo).sum()


def within_support(x):
    return True


def sample_mu(tu, mu0, rh0, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    sg0 = (tu * rh0)**-0.5
    mu = norm(loc = mu0, scale = sg0).rvs(random_state = rng)
    return mu


def update_mu(zz, xx, tu, mu0, rh0, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    n, D = xx.shape
    K = len(np.unique(zz))
    mu = []
    for k in range(K):
        xk = xx[np.where(zz == k)]
        nk = len(xk)
        mu_muk = (nk / (nk + rh0)) * xk.mean(axis = 0) + (rh0 / (nk + rh0)) * mu0
        Sg_muk = np.eye(D) / (nk + rh0) / tu
        mu.append(multivariate_normal(mean = mu_muk, cov = Sg_muk).rvs(random_state = rng))
    return np.array(mu)


def update_tu(zz, xx, mu0, rh0, a0, b0, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    n, D = xx.shape
    a1 = a0 + n * D / 2
    b1 = b0 + 0
    K = len(np.unique(zz))
    for k in range(K):
        xk = xx[np.where(zz == k)]
        nk = len(xk)
        xk_bar = xk.mean(axis = 0)
        b1 += 0.5 * np.sum((xk - xk_bar)**2)
        b1 += nk * rh0 / 2 / (rh0 + nk) * np.sum((xk_bar - mu0)**2)
    tu = gamma(a = a1, scale = 1 / b1).rvs(random_state = rng)
    return tu


def update_gm(yy, yo, a0, b0, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    res = (yy - yo).ravel()
    a1 = a0 + len(res) / 2
    b1 = b0 + 0.5 * np.sum(res**2)
    gm = gamma(a = a1, scale = 1 / b1).rvs(random_state = rng)
    return gm


def update_xx_pcn(simulator, zz, xx, yy, yo, gm, mu, tu, scale, rng = None):
    '''
    Update unobserved FE model parameters using preconditioned Crank-Nikolson MCMC.
    '''
    if rng is None:
        rng = np.random.default_rng()
    accepts = np.zeros(len(xx))
    for j in range(len(xx)):
        x_cur, y_cur, y_obs = xx[j], yy[j], yo[j]
        l_cur = log_likelihood(y_cur, y_obs, gm)
        # pCN proposal
        eta = multivariate_normal(
            mean = np.zeros_like(x_cur), cov = np.eye(len(x_cur)) / tu
        ).rvs(random_state = rng)
        x_new = mu[zz[j]] + np.sqrt(1 - scale[j]**2) * (x_cur - mu[zz[j]]) + scale[j] * eta
        # m-h rule
        if within_support(x_new):
            y_new = simulator(x_new)
            l_new = log_likelihood(y_new, y_obs, gm)
            log_rat = l_new - l_cur
            if np.log(rng.uniform()) < log_rat:
                xx[j] = x_new
                yy[j] = y_new
                accepts[j] = 1
    return xx, yy, accepts


def update_xx_pcn_non_hierarchical(simulator, xx, yy, yo, gm, scale, rng = None):
    '''
    Update unobserved FE model parameters using preconditioned Crank-Nikolson MCMC.
    (Non-hierarchical case)
    '''
    if rng is None:
        rng = np.random.default_rng()
    accepts = np.zeros(len(xx))
    for j in range(len(xx)):
        x_cur, y_cur, y_obs = xx[j], yy[j], yo[j]
        l_cur = log_likelihood(y_cur, y_obs, gm)
        # pCN proposal
        eta = multivariate_normal(
            mean = np.zeros_like(x_cur), cov = np.eye(len(x_cur))
        ).rvs(random_state = rng)
        x_new = np.sqrt(1 - scale[j]**2) * x_cur + scale[j] * eta
        # m-h rule
        if within_support(x_new):
            y_new = simulator(x_new)
            l_new = log_likelihood(y_new, y_obs, gm)
            log_rat = l_new - l_cur
            if np.log(rng.uniform()) < log_rat:
                xx[j] = x_new
                yy[j] = y_new
                accepts[j] = 1
    return xx, yy, accepts


def update_al(al, K, N, c1 = 1.0, c2 = 1.0, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    eta = beta(al + 1.0, N).rvs(random_state = rng)
    pi = (c1 + K - 1.0) / (N * (c2 - np.log(eta)) + c1 + K - 1.0)
    if bernoulli(pi).rvs(random_state = rng) == 1:
        shape = c1 + K
    else:
        shape = c1 + K - 1.0
    rate = c2 - np.log(eta)
    al_new = gamma(a = shape, scale = 1.0 / rate).rvs(random_state = rng)
    return al_new