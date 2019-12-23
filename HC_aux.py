import pandas as pd
import numpy as np
from scipy.stats import binom

def hc_vals(pv, alpha=0.45, stbl=True):
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
        stbl -- use expected p-value ordering (stbl=True) or observed (stbl=False)

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

        uu = np.linspace(1 / n, 0.999, n)  #approximate expectation of p-values
        i_lim_up = np.maximum(int(np.floor(alpha * n + 0.5)), 1)

        ps = ps[:i_lim_up]
        uu = uu[:i_lim_up]
        
        i_lim_low = np.argmax(ps > 0.999/n)

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
        i_lim_low = np.argmax(ps > 0.99/n)
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


def pval_bin(n1, n2, T1, T2, min_counts=3, randomize = False):
    if ((n1 + n2 >= min_counts) and (T1 > n1)):
        if randomize :
            pval = binom_test_two_sided_random(x=np.array([n1]),
                          n=n1 + n2,
                          p=(T1 - n1) / np.float((T1 + T2 - n1 - n2)))[0]
        else :
            pval = binom_test(x=n1,
                              n=n1 + n2,
                              p=(T1 - n1) / np.float((T1 + T2 - n1 - n2)))
    else:
        pval = np.nan
    return pval


def binom_test_two_sided2(x, n, p) :
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

    prob = binom.cdf(x_low, n, p)\
        + binom.cdf(n-x_high, n, 1-p)

    return prob * (n!=0) + 1.* (n==0)

def binom_test_two_sided_random(x, n, p) :
    x_low = n * p - np.abs(x-n*p)
    x_high = n * p + np.abs(x-n*p)

    p_up = binom.cdf(x_low, n, p)\
        + binom.cdf(n-x_high, n, 1-p)
    
    p_down = binom.cdf(x_low-0.5, n, p)\
        + binom.cdf(n-x_high-0.5, n, 1-p)
        
    p = np.minimum(p_down + (p_up-p_down)*np.random.rand(x.shape[0]), 1)
    return p * (n != 0) + 1. * (n == 0)

    

def z_score(n1, n2, T1, T2):
    p = (n1 + n2) / (T1 + T2)  #pooled prob of success
    se = np.sqrt(p * (1 - p) * (1. / T1 + 1. / T2))
    return (n1 / T1 - n2 / T2) / se

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
    counts['p'] = (T1 - counts['n1']) / (T1 + T2 - counts['n1'] - counts['n2'])

    counts['T1'] = T1
    counts['T2'] = T2

    #counts['pval'] = #counts.apply(lambda row: pval_bin(
        #row['n1'], row['n2'], row['T1'], row['T2'],
        # min_counts=min_counts, randomize = randomize),
        #                          axis=1)

    if randomize :
        counts['pval'] = binom_test_two_sided_random(counts.n1.values,
                                          counts.n1.values + counts.n2.values,
                                          counts['p']
                                           )
    else :
        counts['pval'] = binom_test_two_sided(counts.n1.values,
                                          counts.n1.values + counts.n2.values,
                                          counts['p']
                                           )
    counts['sign'] = np.sign(counts.n1 - (counts.n1 + counts.n2) * counts.p)
    hc_star, p_val_thresh = hc_vals(counts['pval'], alpha=alpha, stbl=stbl)
    counts['HC'] = hc_star

    counts['thresh'] = True
    counts.loc[counts['pval'] >= p_val_thresh, ('thresh')] = False
    counts.loc[np.isnan(counts['pval']), ('thresh')] = False
    
    return counts


def two_counts_pvals(c1, c2, randomize=False):

    T1 = c1.sum()
    T2 = c2.sum()
    p = (T1 - c1) / (T1 + T2 - c1 - c2)

    #Joining unit1 and unit2 for the HC computation
    if randomize :
        pvals = binom_test_two_sided_random(c1, c1 + c2, p)
    else :
        pvals = binom_test_two_sided(c1, c1 + c2, p)

    return pvals

def two_list_test(term_cnt1,
                  term_cnt2,
                  lo_terms=pd.DataFrame(),
                  alpha=0.35,
                  min_counts=3,
                  stbl=True):
    #  HC test based on terms in lo_terms (or all terms otherwise)
    #  Input: term_cnt1, term_cnt2 -- list of the form term-count (with possible multiplicities)
    # lump counts
    unit1 = term_cnt1.groupby(['term']).sum()
    unit2 = term_cnt2.groupby(['term']).sum()

    #if list of terms is not provided, use all terms
    if lo_terms.shape[0] == 0:
        ls = unit1.index.tolist() + unit2.index.tolist()
        lo_terms = pd.DataFrame({'term': list(set(ls))})

    #merge based on list of terms
    lo_terms = lo_terms.filter(['term'])
    unit1_red = unit1.merge(lo_terms, how='right', on=['term']).fillna(0)
    unit2_red = unit2.merge(lo_terms, how='right', on=['term']).fillna(0)

    counts = pd.DataFrame()
    counts['term'] = unit1_red.term
    counts['n1'] = unit1_red.n
    counts['n2'] = unit2_red.n
    T1 = counts['n1'].sum()
    T2 = counts['n2'].sum()
    counts['p'] = (T1 - counts['n1']) / (T1 + T2 - counts['n1'] - counts['n2'])

    counts['T1'] = T1
    counts['T2'] = T2
    
    counts['pval'] = binom_test_two_sided(counts.n1.values,
                                          counts.n1.values + counts.n2.values,
                                          counts['p']
                                           )

    counts['sign'] = np.sign(counts.n1 - (counts.n1 + counts.n2) * counts.p)
    hc_star, p_val_thresh = hc_vals(counts['pval'], alpha=alpha, stbl=stbl)
    counts['HC'] = hc_star
    counts['thresh'] = True
    counts.loc[counts['pval'] >= p_val_thresh, ('thresh')] = False
    counts.loc[np.isnan(counts['pval']), ('thresh')] = False

    return counts

def binom_test(x,n=None,p=0.5):
    """
    Perform a test that the probability of success is p.
    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.
    Parameters
    ----------
    x : integer or array_like
        the number of successes, or if x has length 2, it is the
        number of successes and the number of failures.
    n : integer
        the number of trials.  This is ignored if x gives both the
        number of successes and failures
    p : float, optional
        The hypothesized probability of success.  0 <= p <= 1. The
        default value is p = 0.5
    Returns
    -------
    p-value : float
        The p-value of the hypothesis test
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Binomial_test
    """
    
    d = scipy.stats.binom.pmf(x,n,p)
    rerr = 1+1e-7
    if (x < p*n):
        i = np.arange(np.ceil(p*n),n+1)
        y = np.sum(scipy.stats.binom.pmf(i,n,p) <= d*rerr,axis=0)
        pval = scipy.stats.binom.cdf(x,n,p) + scipy.stats.binom.sf(n-y,n,p)
    else:
        i = np.arange(np.floor(p*n) + 1)
        y = np.sum(scipy.stats.binom.pmf(i,n,p) <= d*rerr,axis=0)
        pval = scipy.stats.binom.cdf(y-1,n,p) + scipy.stats.binom.sf(x-1,n,p)

    return min(1.0,pval)
