from scipy.stats import chi2_contingency, ks_2samp
from scipy.spatial.distance import cosine
import numpy as np

def two_sample_chi_square(c1, c2, lambda_="pearson"):
    """returns the Chi-Square score of the two samples c1 and c2
     (representing counts). Null cells are ignored. 

    Args: 
     c1, c2  -- two arrays of integers of equal length
    
    Returns:
        chisq -- centralized chi-square score (score - dof)
        log of pvalue -- p-value
    """
    
    if (sum(c1) == 0) or (sum(c2) == 0) :
        return np.nan, 1
    else :
        obs = np.array([c1, c2])
        chisq, pval, dof, exp = chi2_contingency(
                                        obs[:,obs.sum(0)!=0],
                                        lambda_=lambda_
                                                            )
        return chisq - dof, np.log(pval)

def two_sample_KS(c1, c2) :
    """ 2-sample Kolmogorov-Smirnov test
    """
    return ks_2samp(c1, c2)

def cosine_sim(c1, c2):
    """
    returns the cosine similarity of the two sequences
    (c1 and c2 are assumed to be numpy arrays of equal length)
    """
    return cosine(c1, c2)
