import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

import re


def hc_vals(pv, alpha=0.45, stbl=True):
    pv = np.asarray(pv)
    pv = pv[~np.isnan(pv)]
    n = len(pv)
    hc_star = np.nan
    p_star = np.nan

    if n > 0:
        uu = np.linspace(1 / n, 1, n)  #approximate expectation of p-values
        ps_idx = np.argsort(pv)
        ps = pv[ps_idx]  #sorted pvals

        if stbl:
            z = (uu - ps) / np.sqrt(uu * (1 - uu) + 1e-20) * np.sqrt(n)
        else:
            z = (uu - ps) / np.sqrt(ps * (1 - ps) + 1e-20) * np.sqrt(n)

        i_lim_up = np.maximum(int(np.floor(alpha * n + 0.5)), 1)
        try:
            i_lim_low = np.where(ps > 0.99 / n)[0][0]
        except:
            i_lim_low = 0

        i_lim_up = max(i_lim_low + 1, i_lim_up)

        i_max_star = np.argmax(z[i_lim_low:i_lim_up]) + i_lim_low

        z_max_star = z[i_max_star]

        hc_star = z[i_max_star]
        p_star = ps[i_max_star]

    return hc_star, p_star


def hc_vals_full(pv, alpha=0.45):
    pv = np.asarray(pv)
    pv = pv[~np.isnan(pv)]
    n = len(pv)
    hc_star = np.nan
    p_star = np.nan

    if n > 0:
        uu = np.linspace(1 / n, 1, n)  #approximate expectation of p-values
        ps = np.sort(pv)  #sorted pvals
        ps_idx = np.argsort(pv)

        z_stbl = (uu - ps) / np.sqrt(uu * (1 - uu) + 1e-10) * np.sqrt(n)
        z = (uu - ps) / np.sqrt(ps * (1 - ps) + 1e-10) * np.sqrt(n)

        i_lim = np.maximum(int(np.floor(alpha * n + 0.5)), 1)
        i_max = np.argmax(z[:i_lim])
        z_max = z[i_max]

        #compute HC
        i_lim_up = np.maximum(int(np.floor(alpha * n + 0.5)), 1)
        try:
            i_lim_low = np.arange(n)[ps > 0.99 / n][0]
        except:
            i_lim_low = 0

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


def pval_bin(n1, n2, T1, T2, min_counts=3):
    from scipy.stats import binom_test
    if ((n1 + n2 >= min_counts) and (T1 > n1)):
        pval = binom_test(x=n1,
                          n=n1 + n2,
                          p=(T1 - n1) / np.float((T1 + T2 - n1 - n2)))
    else:
        pval = np.nan
    return pval


def z_score(n1, n2, T1, T2):
    p = (n1 + n2) / (T1 + T2)  #pooled prob of success
    se = np.sqrt(p * (1 - p) * (1. / T1 + 1. / T2))
    if (se > 0) and (~np.isnan(se)):
        return (n1 / T1 - n2 / T2) / se
    else:
        return np.nan


def z_prop_test(n1, n2, T1, T2):
    from scipy.stats import norm
    z = z_score(n1, n2, T1, T2)
    return 2 * norm.cdf(-np.abs(z))


def two_sample_test(X, Y, alpha=0.45, stbl=True,min_counts=3):
    # Input: X, Y, are two lists of integers of equal length :
    # Output: data frame: "X, Y, T1, n2, T2, pval, pval_z, hc"
    counts = pd.DataFrame()
    counts['n1'] = X
    counts['n2'] = Y
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    counts['pval'] = counts.apply(lambda row: pval_bin(
        row['n1'], row['n2'], row['T1'], row['T2'], min_counts=min_counts),
                                  axis=1)
    counts['z'] = counts.apply(
        lambda row: z_score(row['n1'], row['n2'], row['T1'], row['T2']),
        axis=1)

    hc_star, p_val_thresh = hc_vals(counts['pval'], alpha=alpha, stbl=stbl)
    counts['hc'] = hc_star
    counts.loc[counts['pval'] > p_val_thresh, ('z')] = np.nan
    counts.loc[np.isnan(counts['pval']), ('z')] = np.nan
    return counts


def lo_terms_to_counts(term_df1, term_df2, lo_terms=[]):
    "unit1 and unit2 are dataframes with one term in each row"
    "lo_terms is a dataframe with at least one column 'token' (or None) "

    tc1 = term_df1['token'].value_counts()  #count terms in df1
    tc2 = term_df2['token'].value_counts()  #count terms in df2

    unit1 = pd.DataFrame({'token': tc1.index, 'n': tc1.values})
    unit2 = pd.DataFrame({'token': tc2.index, 'n': tc2.values})

    #if list of terms is not provided, use all terms
    if len(lo_terms) == 0:
        ls = term_df1.token.tolist() + term_df1.token.tolist()
        lo_terms = pd.DataFrame({'token': list(set(ls))})
        #lo_terms.term = lo_terms.term.astype(str)

    #merge based on list of terms
    counts = pd.DataFrame({'token': lo_terms.token})
    counts = counts.merge(unit1, how='left', on=['token']).fillna(0)
    counts = counts.merge(unit2, how='left', on=['token']).fillna(0)

    counts = counts.rename(columns={"n_x": "n1", "n_y": "n2"})

    return counts


def two_list_test(term_cnt1,
                  term_cnt2,
                  lo_terms=pd.DataFrame(),
                  alpha=0.45,
                  stbl = True,
                  min_counts=3):
    #  HC test based on terms in lo_terms (or all terms otherwise)
    #  Input: term_cnt1, term_cnt2 -- list of the form term-count 
    # (with possible multiplicities)

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
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    counts['pval'] = counts.apply(lambda row: pval_bin(
        row['n1'], row['n2'], row['T1'], row['T2'], min_counts=min_counts),
                                  axis=1)
    counts['z'] = counts.apply(
        lambda row: z_score(row['n1'], row['n2'], row['T1'], row['T2']),
        axis=1)

    hc_star, p_val_thresh = hc_vals(counts['pval'], alpha=alpha, stbl = stbl)
    counts['hc'] = hc_star
    counts.loc[counts['pval'] > p_val_thresh, ('z')] = np.nan
    counts.loc[np.isnan(counts['pval']), ('z')] = np.nan
    return counts


def two_counts_pvals(c1, c2, min_counts=3):
    counts = pd.DataFrame()
    counts['n1'] = np.atleast_1d(c1)
    counts['n2'] = np.atleast_1d(c2)
    counts['T1'] = counts['n1'].sum()
    counts['T2'] = counts['n2'].sum()

    #Joining unit1 and unit2 for the HC computation
    counts['pval'] = counts.apply(lambda row: pval_bin(
        row['n1'], row['n2'], row['T1'], row['T2'], min_counts=min_counts),
                                  axis=1)
    return counts


def two_counts_pvals_df(counts_df, min_counts=1):
    #add pval row to dataframe with columns n1, n2, T1, T2
    counts_df['pval'] = counts_df.apply(lambda row: pval_bin(
        row['n1'], row['n2'], row['T1'], row['T2'], min_counts=min_counts),
                                        axis=1)
    counts_df['z'] = counts_df.apply(
        lambda row: z_score(row['n1'], row['n2'], row['T1'], row['T2']),
        axis=1)
    return counts_df
