import pandas as pd
import numpy as np
from scipy.stats import binom, norm, poisson

def hc_vals(pv, alpha=0.25, minPv='1/n', stbl=True):
    """
    Higher Criticism test (see
    [1] Donoho, D. L. and Jin, J.,
     "Higher criticism for detecting sparse hetrogenous mixtures", 
     Annals of Stat. 2004
    [2] Donoho, D. L. and Jin, J. "Higher critcism thresholding: Optimal 
    feature selection when useful features are rare and weak", proceedings
    of the national academy of sciences, 2008.
     )

    Parameters:
        pv -- list of p-values. P-values that are np.nan are exluded.
        alpha -- lower fruction of p-values to use.
        stbl -- use expected p-value ordering (stbl=True) or observed 
                (stbl=False)

    Return :
        hc_star -- sample adapted HC (HC\dagger in [1])
        p_star -- HC threshold: upper boundary for collection of
                 p-value indicating the largest deviation from the
                 uniform distribution.

    """
    pv = np.asarray(pv)
    n = len(pv)  #number of features including NA
    pv = pv[~np.isnan(pv)]  
    #n = len(pv)
    hc_star = np.nan
    p_star = np.nan

    if n > 0:
        ps_idx = np.argsort(pv)
        ps = pv[ps_idx]  #sorted pvals

        uu = np.linspace(1 / n, 0.999, n)  #expectation of p-values under H0
        i_lim_up = np.maximum(int(np.floor(alpha * n + 0.5)), 1)

        ps = ps[:i_lim_up]
        uu = uu[:i_lim_up]
        
        if minPv == '1/n' :
            i_lim_low = np.argmax(ps > 0.999/n)
        else :
            i_lim_low = minPv

        if stbl:
            z = (uu - ps) / np.sqrt(uu * (1 - uu)) * np.sqrt(n)
        else:
            z = (uu - ps) / np.sqrt(ps * (1 - ps)) * np.sqrt(n)

        i_lim_up = max(i_lim_low + 1, i_lim_up)

        i_max_star = np.argmax(z[i_lim_low:i_lim_up]) + i_lim_low

        z_max_star = z[i_max_star]

        hc_star = z[i_max_star]
        p_star = ps[i_max_star]

    return hc_star, p_star


def hc_vals_full(pv, alpha=0.45):
    pv = np.asarray(pv)
    n = len(pv)
    pv = pv[~np.isnan(pv)]
    hc_star = np.nan
    p_star = np.nan

    if n > 0:
        ps_idx = np.argsort(pv)
        ps = pv[ps_idx]  #sorted pvals

        uu = np.linspace(1 / n, 0.999, n)  #expectation of p-values
        i_lim_up = np.maximum(int(np.floor(alpha * n + 0.5)), 1)

        uu = uu[:i_lim_up]
        ps = ps[:i_lim_up]

        z_stbl = (uu - ps) / np.sqrt(uu * (1 - uu)) * np.sqrt(n)
        z = (uu - ps) / np.sqrt(ps * (1 - ps)) * np.sqrt(n)

        i_lim = np.maximum(int(np.floor(alpha * n + 0.5)), 1)
        i_max = np.argmax(z[:i_lim])
        z_max = z[i_max]

        #compute HC
        i_lim_low = 0 #np.argmax(ps > 0.99/n)
        # try:
        #     i_lim_low = np.arange(n)[ps > 0.99 / n][0]
        # except:
        #     i_lim_low = 0

        #i_max = np.argmax(z[:i_lim_up])
        i_lim_up = max(i_lim_low + 1, i_lim_up)

        i_max_star = np.argmax(z[i_lim_low:i_lim_up]) + i_lim_low

        hc_star = z[i_max_star]
        p_star = ps[i_lim_low]

        p_star_full = ps[np.argmax(z[:i_lim_up])]

        #i_max = np.argmax(z_stbl[:i_lim_up])
        i_max_star = np.argmax(z_stbl[i_lim_low:i_lim_up]) + i_lim_low

        p_star_full_stbl = ps[np.argmax(z_stbl[:i_lim_up])]

        hc_star_stbl = z_stbl[i_max_star]

        p_star_stbl = ps[i_max_star]


    import pandas as pd
    df = pd.DataFrame({
        'pval': ps,
        'z': z,
        'z_stbl': z_stbl,
        'u': uu,
        'HC': hc_star,
        'HC_stbl': hc_star_stbl,
        'p_star_full' : p_star_full,
        'p_star_full_stbl' : p_star_full_stbl,
        'p_star': p_star,
        'p_star_stbl': p_star_stbl
    })
    return df

def binom_test_two_sided_slow(x, n, p) :
    #slower
    def my_func(r) :
        from scipy.stats import binom_test
        return binom_test(r[0],r[1],r[2])

    a = np.concatenate([np.expand_dims(x,1),
                    np.expand_dims(n,1),
                    np.expand_dims(p,1)],
                    axis = 1)

    pv = np.apply_along_axis(my_func,1,a)

    return pv

def binom_test_two_sided(x, n, p) :
    x_low = n * p - np.abs(x-n*p)
    x_high = n * p + np.abs(x-n*p)

    p_up = binom.cdf(x_low, n, p)\
        + binom.sf(x_high-1, n, p)
        
    prob = np.minimum(p_up, 1)
    return prob * (n != 0) + 1. * (n == 0)


def poisson_test_random(x, lmd) :
    p_down = 1 - poisson.cdf(x, lmd)
    p_up = 1 - poisson.cdf(x, lmd) + poisson.pmf(x, lmd)
    U = np.random.rand(x.shape[0])
    prob = np.minimum(p_down + (p_up-p_down)*U, 1)
    return prob * (n != 0) + U * (n == 0)


def binom_test_two_sided_random(x, n, p) :
    x_low = n * p - np.abs(x-n*p)
    x_high = n * p + np.abs(x-n*p)

    p_up = binom.cdf(x_low, n, p)\
        + binom.sf(x_high-1, n, p)
    
    p_down = binom.cdf(x_low-1, n, p)\
        + binom.sf(x_high, n, p)
    
    U = np.random.rand(x.shape[0])
    prob = np.minimum(p_down + (p_up-p_down)*U, 1)
    return prob * (n != 0) + U * (n == 0)

def two_sample_test(
    X,
    Y,
    alpha=0.45,
    stbl=True,
    randomize=False
    ):
    # Input: X, Y, are two lists of integers of equal length :
    # Output: data frame: "X, Y, T1, n2, T2, pval, pval_z, hc"
    counts = pd.DataFrame()
    counts['n1'] = X
    counts['n2'] = Y
    T1 = counts['n1'].sum()
    T2 = counts['n2'].sum()
    counts['p'] = (T1 - counts.n1) / (T1 + T2 - counts.n1 - counts.n2)

    counts['T1'] = T1
    counts['T2'] = T2

    counts['pval'] = two_counts_pvals(
        counts['n1'],
        counts['n2'],
        randomize=randomize
        )
    counts['sign'] = np.sign(counts.n1 - (counts.n1 + counts.n2) * counts.p)
    hc_star, p_val_thresh = hc_vals(counts['pval'], alpha=alpha, stbl=stbl)
    counts['HC'] = hc_star

    counts['thresh'] = True
    counts.loc[counts['pval'] >= p_val_thresh, ('thresh')] = False
    counts.loc[np.isnan(counts['pval']), ('thresh')] = False
    
    return counts

def two_counts_pvals(c1, c2, randomize=False):
    # feature by feature exact binomial test
    T1 = c1.sum()
    T2 = c2.sum()
    p = (T1 - c1) / (T1 + T2 - c1 - c2)

    if randomize :
        pvals = binom_test_two_sided_random(c1, c1 + c2, p)
    else :
        pvals = binom_test_two_sided(c1, c1 + c2, p)
        #pvals = binom_test_two_sided_slow(c1, c1 + c2, p)

    return pvals


