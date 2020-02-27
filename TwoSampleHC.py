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
        minPv -- integer or string '1/n' (default). Ignote smallest
                 minPv-1 when computing HC score.

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

def binom_test_two_sided_slow(x, n, p) :
    """
     Calls scipy.stats.binom_test on each entry of
     an array. Slower than binom_test_two_sided but 
     perhaps more accurate. 
    """
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

def poisson_test_random(x, lmd) :
    p_down = 1 - poisson.cdf(x, lmd)
    p_up = 1 - poisson.cdf(x, lmd) + poisson.pmf(x, lmd)
    U = np.random.rand(x.shape[0])
    prob = np.minimum(p_down + (p_up-p_down)*U, 1)
    return prob * (n != 0) + U * (n == 0)


def binom_test_two_sided(x, n, p) :
    """
    Returns:

    Prob( |Bin(n,p) - np| >= |x-np| )

    Note: for small values of Prob there are differences
    fron scipy.python.binom_test. It is unclear which one is 
    more accurate.
    """
    x_low = n * p - np.abs(x-n*p)
    x_high = n * p + np.abs(x-n*p)

    p_up = binom.cdf(x_low, n, p)\
        + binom.sf(x_high-1, n, p)
        
    prob = np.minimum(p_up, 1)
    return prob * (n != 0) + 1. * (n == 0)


def binom_test_two_sided_random(x, n, p) :
    """
    Returns:
    pval  -- random number such that 
               Prob(|Bin(n,p) - np| >= 
                |InvCDF(pval|Bin(n,p)) - n p|) ~ U(0,1)
    """

    x_low = n * p - np.abs(x-n*p)
    x_high = n * p + np.abs(x-n*p)

    p_up = binom.cdf(x_low, n, p)\
        + binom.sf(x_high-1, n, p)
    
    p_down = binom.cdf(x_low-1, n, p)\
        + binom.sf(x_high, n, p)
    
    U = np.random.rand(x.shape[0])
    prob = np.minimum(p_down + (p_up-p_down)*U, 1)
    return prob * (n != 0) + U * (n == 0)

def two_sample_test(X, Y, alpha=0.25,
                stbl=True, randomize=False):
    """
    Two-sample HC test using binomial P-values. See 
    [1] Alon Kipnis, ``Higher Criticism for Discriminating Word-Frequency
    Tables and Testing Authorship'', 2019
    
    This function combines two_sample_pvals and hc_vals.

    Parameters:
    X, Y : list of integers of equal length -- represnts counts 
            from two samples.
    alpha : number in (0,1) -- parameter of HC statistics
    stbl : Boolean -- standardize P-values by i/N or p_{(i)}/N
    randomize : Boolean -- randomized P-valus of not

    Returns:
    HC : HC score under Binmial P-values
    p_thresh : HC threshold
    """
    
    pvals = two_sample_pvals(X, Y, randomize=randomize)
    hc_star, p_thresh = hc_vals(pvals, alpha=alpha, stbl=stbl)
    
    return hc_star, p_thresh


def two_sample_pvals(c1, c2, randomize=False):
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


def two_sample_test_df(X, Y, alpha=0.25,
                stbl=True, randomize=False):
    """
    Same as two_sample_test but returns all information for computing
     HC score of the two samples. Requires pandas.
    """

    counts = pd.DataFrame()
    counts['n1'] = X
    counts['n2'] = Y
    T1 = counts['n1'].sum()
    T2 = counts['n2'].sum()
    counts['p'] = (T1 - counts.n1) / (T1 + T2 - counts.n1 - counts.n2)

    counts['T1'] = T1
    counts['T2'] = T2

    counts['pval'] = two_sample_pvals(
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

def hc_vals_full(pv, alpha=0.25):
    """
    Higher Criticism test with intermediate computation information
    (see
    [1] Donoho, D. L. and Jin, J.,
     "Higher criticism for detecting sparse hetrogenous mixtures", 
     Annals of Stat. 2004
    )

    Parameters:
        pv -- list of p-values. P-values that are np.nan are exluded.
        alpha -- lower fruction of p-values to use.
        
    Return :
        df -- DataFrame with fields describing HC computation

    """
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

        def get_HC(z, i_low, i_high) :
            i_max = np.argmax(z[i_low:i_high]) + i_low
            HC = z[i_max]
            return HC, i_max
            
        #compute HC
        HC, i_star = get_HC(z, 0, i_lim_up)

        HC_stbl, i_star_stbl = get_HC(z_stbl, 0, i_lim_up)

        i_lim_low_dagger=np.argmax(ps > 0.999/n)
        i_lim_up_dagger = max(i_lim_low_dagger + 1, i_lim_up)
        HC_dagger, i_star_dagger = get_HC(
            z, i_lim_low_dagger, i_lim_up_dagger)
        
        HC_stbl_dagger, i_star_stbl_dagger = get_HC(
            z_stbl, i_lim_low_dagger, i_lim_up_dagger)
        

    import pandas as pd
    df = pd.DataFrame({
        'pval': ps,
        'z': z,
        'z_stbl': z_stbl,
        'u': uu,
        'HC': HC,
        'HC_stbl': HC_stbl,
        'HC_dagger' : HC_dagger,
        'HC_stbl_dagger' : HC_stbl_dagger,
        'thresh' : ps < ps[i_star],
        'thresh_stbl' : ps < ps[i_star_stbl],
        'thresh_dagger' : ps < ps[i_star_dagger],
        'thresh_stbl_dagger' : ps < ps[i_star_stbl_dagger],
    })
    return df


