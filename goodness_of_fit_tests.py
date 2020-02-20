from scipy.stats import chi2_contingency, ks_2samp, norm
from scipy.spatial.distance import cosine
import numpy as np

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
    log_pval : log of p-value
    """
    
    if (sum(c1) == 0) or (sum(c2) == 0) :
        return np.nan, 1
    else :
        obs = np.array([c1, c2])
        if lambda_ in ['mod-log-likelihood',
                         'freeman-tukey',
                          'neyman'] :
            obs_nz = obs[:, (obs[0]!=0) & (obs[1]!=0)]
        else :
            obs_nz = obs[:, (obs[0]!=0) | (obs[1]!=0)]

        chisq, pval, dof, exp = chi2_contingency(
                                    obs_nz, lambda_=lambda_)
        if pval == 0:
            Lpval = -np.inf
        else :
            Lpval = np.log(pval)
        return chisq / dof, Lpval
        

def two_sample_KS(s1, s2) :
    """ 2-sample Kolmogorov-Smirnov test
    """
    return ks_2samp(s1, s2)

def cosine_sim(c1, c2):
    """
    returns the cosine similarity of the two sequences
    (c1 and c2 are assumed to be numpy arrays of equal length)
    """
    return cosine(c1, c2)


def z_test(n1, n2, T1, T2):
    p = (n1 + n2) / (T1 + T2)  #pooled prob of success
    se = np.sqrt(p * (1 - p) * (1. / T1 + 1. / T2))
    return (n1 / T1 - n2 / T2) / se


def two_sample_proportion(c1, c2) :
    T1 = c1.sum()
    T2 = c2.sum()
    
    p = (c1 + c2) / (T1 + T2) #pooled prob of success
    se = np.sqrt(p * (1 - p) * (1. / T1 + 1. / T2)) #pooled std

    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.divide(c1 / T1 - c2 / T2, se)
    
    return 2*norm.cdf(-np.abs(z))
