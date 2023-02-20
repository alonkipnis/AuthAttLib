from scipy.stats import chi2_contingency, norm
from scipy.spatial.distance import cosine

from TwoSampleHC import two_sample_pvals, HC
import numpy as np


def HC_sim(c1, c2, gamma=0.15, randomize=False,
           pval_thresh=1.1, HCtype='HCstar'):
    """
    Higher-Criticism (HC) similarity of two discrete samples

    Args:
    -----
    c1, c2 : two lists of integers of equal length
    gamma : HC parameter
    randomize : randomized Pvalues or normalization
    pval_thresh : only use P-values below this value. Has not effect
                  if pval_thresh > 1. 

    Returns: 
    -------
    HCstar of the binomial allocation P-values of the two lists
    """
    pvals = two_sample_pvals(c1, c2, randomize=randomize)
    pvals_red = pvals[pvals < pval_thresh]

    if len(pvals_red) == 0:
        return np.nan

    if HCtype == 'HCstar':
        hc, _ = HC(pvals_red).HCstar(gamma=gamma)
    else:
        hc, _ = HC(pvals_red).HC(gamma=gamma)
    return hc


def BJ_sim(c1, c2, gamma=0.1, randomize=False, pval_thresh=1.1):
    """
    Berk-Jones (BJ) similarity of two discrete samples

    Args:
    -----
    c1, c2 : two lists of integers of equal length
    gamma : lower fraction of P-values
    randomize : randomized Pvalues or normalization
    pval_thresh : only use P-values below this value. Has not effect
                  if pval_thresh > 1. 

    Returns:
    -------
    HCstar of the binomial allocation P-values of the two lists
    """
    pvals = two_sample_pvals(c1, c2, randomize=randomize)
    pvals_red = pvals[pvals < pval_thresh]

    if len(pvals_red) == 0:
        return np.nan

    bj, _ = HC(pvals_red).BJ(gamma=gamma)
    return bj


def two_sample_chi_square(c1, c2, lambda_="pearson"):
    """returns the Chi-Square score of the two samples c1 and c2
     (representing counts). Null cells are ignored. 

    Args: 
     c1, c2 : list of integers
        representing two 1-way contingency tables
     lambda_ : one of :
            "pearson"             1     Pearson's chi-squared statistic.
                                        In this case, the function is
                                        equivalent to `stats.chisquare`.
            "log-likelihood"      0     Log-likelihood ratio. Also known as
                                        the G-test [Rf6c2a1ea428c-3]_.
            "freeman-tukey"      -1/2   Freeman-Tukey statistic.
            "mod-log-likelihood" -1     Modified log-likelihood ratio.
            "neyman"             -2     Neyman's statistic.
            "cressie-read"        2/3   
    
    Returns
    -------
    chisq : score
        score divided by degree of freedom. 
        this normalization is useful in comparing multiple
        chi-squared scores. See Ch. 9.6.2 in 
        Yvonne M. M. Bishop, Stephen E. Fienberg, and Paul 
        W. Holland ``Discrete Multivariate Analysis''  
    log_pval: log of p-value
    """

    if (sum(c1) == 0) or (sum(c2) == 0):
        return np.nan, 1
    else:
        obs = np.array([c1, c2])
        if lambda_ in ['mod-log-likelihood',
                       'freeman-tukey',
                       'neyman']:
            obs_nz = obs[:, (obs[0] != 0) & (obs[1] != 0)]
        else:
            obs_nz = obs[:, (obs[0] != 0) | (obs[1] != 0)]

        chisq, pval, dof, exp = chi2_contingency(
            obs_nz, lambda_=lambda_)
        if pval == 0:
            Lpval = -np.inf
        else:
            Lpval = np.log(pval)
        return chisq / dof, Lpval


def cosine_sim(c1, c2):
    """
    returns the cosine similarity of the two sequences
    (c1 and c2 are assumed to be numpy arrays of equal length)
    """
    return cosine(c1, c2)


def z_test(n1, n2, T1, T2):
    p = (n1 + n2) / (T1 + T2)  # pooled prob of success
    se = np.sqrt(p * (1 - p) * (1. / T1 + 1. / T2))
    return (n1 / T1 - n2 / T2) / se


def two_sample_proportion(c1, c2):
    T1 = c1.sum()
    T2 = c2.sum()

    p = (c1 + c2) / (T1 + T2)  # pooled prob of success
    se = np.sqrt(p * (1 - p) * (1. / T1 + 1. / T2))  # pooled std

    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.divide(c1 / T1 - c2 / T2, se)

    return 2 * norm.cdf(-np.abs(z))
