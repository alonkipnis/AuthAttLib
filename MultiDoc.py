from sklearn.feature_extraction.text import CountVectorizer
import logging
from scipy.stats import chisquare, poisson, binom
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.special import binom as nkchoose
import scipy

from TwoSampleHC import two_sample_pvals, HC, binom_test_two_sided
from .goodness_of_fit_tests import two_sample_chi_square

def n_label(cls) :
            return f"n ({cls})"

def T_label(cls) :
    return f"T ({cls})"

def multinomial_test(x, p) : # slow
    import met
    assert(len(x) == len(p))
    n_max = 40
    r_max = 1e6
    p = np.array(p) / np.sum(p)
    n = sum(x)
    if sum(x) == 0 or nkchoose(n,len(p)) > r_max :
        logging.warning(f"Number of combinations is too large."
        " Approximating exact binomial test using chisquared test.")
        return chisquare(x, p * n)[1]
    all_multi_cases = met.all_multinom_cases(len(p), n)
    probs = scipy.stats.multinomial.pmf(x=all_multi_cases, n=n, p=p)
    pval = probs[probs <= probs[all_multi_cases.index(list(x))]].sum()
    return pval


def poisson_test_two_sided_matrix_appx(x, lm) :
    return np.select([x < lm, x >= lm], [poisson.cdf(x, lm), poisson.sf(x, lm)])

def poisson_test_two_sided_matrix(x, lm) :
    pl = np.select([x < lm, x > lm, x==lm], [poisson.cdf(x, lm), poisson.sf(x, lm), 1])
    pu = np.select([x < lm, x > lm], [poisson.sf(poisson.isf(pl, lm), lm), poisson.sf(x, lm)])

    return pl + pu


def poisson_test_two_sided(x, lm) :
    pl = poisson.cdf(x, lm) * (x < lm) + poisson.sf(x, lm) * (x > lm) + (x==lm)
    pu = poisson.sf(poisson.isf(pl, lm), lm) * (x < lm) +  poisson.sf(x, lm) * (x > lm) 

    # if x < lm :
    #     assert((pl >= pu/2).all())
    # if x > lm :
    #     assert((pu >= pl/2).all())

    return pl + pu

def binom_test_two_sided(x, n, p) :
    """
    Returns:
    --------
    Prob( |Bin(n,p) - np| >= |x-np| )

    Note: for small values of Prob there are differences
    fron scipy.python.binom_test. It is unclear which one is 
    more accurate.
    """

    n = n.astype(int)

    x_low = n * p - np.abs(x-n*p)
    x_high = n * p + np.abs(x-n*p)

    p_up = binom.cdf(x_low, n, p)\
        + binom.sf(x_high-1, n, p)
        
    prob = np.minimum(p_up, 1)
    return prob * (n != 0) + 1. * (n == 0)

class CompareDocs :
    """
    Class to compare documents in terms of word frequencies. 
    Can select distinguishing words using Higher Criticism threshold. 
    Also tests the similarity of a new document using binomial allocation. 
    """
    def __init__(self, **kwargs) :
        self.pval_type = kwargs.get('pval_type', 'multinom')
        self.vocabulary = kwargs.get('vocabulary', [])
        self.max_features = kwargs.get('max_features', 3000)
        self.min_cnt = kwargs.get('min_cntt', 3)
        self.ngram_range = kwargs.get('ngram_range', (1,1))
        self.measures = kwargs.get('measures', ['HC'])
        self.stbl = kwargs.get('stbl', True)
        self.gamma = kwargs.get('gamma', .2)

        self.counts_df = pd.DataFrame()
        self.num_of_cls = np.nan
        self.cls_names = []
                
    def count_words(self, data) :
        df = pd.DataFrame()
        
        if type(data) == pd.DataFrame :
            df_vocab = pd.DataFrame({'feature' : self.vocabulary})\
                         .set_index('feature')
            df = pd.DataFrame(data.feature.value_counts())\
                    .rename(columns={'feature' : 'n'})
            return df_vocab.join(df, how='left').fillna(0).astype(int)

        pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"
        pat = r"\b\w\w+\b"
        # term counts
        if len(self.vocabulary) == 0:
            tf_vectorizer = CountVectorizer(token_pattern=pat, 
                    max_features=self.max_features, ngram_range=self.ngram_range)
        else:
            tf_vectorizer = CountVectorizer(token_pattern=pat,
                    vocabulary=self.vocabulary, ngram_range=self.ngram_range)
            
        tf = tf_vectorizer.fit_transform([data])
        vocab = tf_vectorizer.get_feature_names()
        tc = np.array(tf.sum(0))[0].astype(int)

        df = pd.concat([df, pd.DataFrame({'feature': vocab, 'n' : tc})])\
               .set_index('feature')
        return df.astype(int)
            
    def fit(self, data) :
        """
        ARGS:
        -----
        data    :   dictionary. One entry per class. Values : string. 
        """
        df = pd.DataFrame()
        if self.vocabulary == [] :
            logging.error("You must provide a vocabulary.")
            raise ValueError
        
        df['feature'] = self.vocabulary
        df['n'] = 0
        df['T'] = 0
        df = df.set_index('feature')

        def n_label(cls) :
            return f"{cls}:n"

        def T_label(cls) :
            return f"{cls}:T"
            
        for cls in data :
            assert(cls != 'tested')  # this name is reserved
            assert(":" not in cls)   # colon symbol is reserved
            
            self.cls_names += [cls]
            logging.debug(f"Processing {cls}...")
            txt = data[cls]
            dfi = self.count_words(txt)
            dfi['n'] = dfi['n'].astype(int)
            logging.debug(f"Found {dfi.n.sum()} terms.")
            df['n'] += dfi.n
            df['T'] += dfi.n.sum()

            dfi[T_label(cls)] = dfi.n.sum()
            dfi = dfi.rename(columns = {'n' : n_label(cls)})
            df = df.join(dfi, how='left', rsuffix=cls)
            values = {n_label(cls) : 0, 
                     T_label(cls) : max(df[T_label(cls)])}
            df = df.fillna(value=values)
        
        self.num_of_cls = len(self.cls_names)
        
        self.counts_df = df
        
    def get_pvals(self) :
        if self.num_of_cls < 2 :
            logging.error("Not enough columns.")
            return np.nan
        df = self.counts_df.copy()
        if self.num_of_cls > 2 :
            logging.info("Using multinomial tests. May be slow.")

            df['x'] = df.filter(regex=r":n$").to_records(index=False).tolist()
            df['p'] = df.filter(regex=r":T$").to_records(index=False).tolist()
            pv = df.apply(lambda r : multinomial_test(r['x'], r['p']), axis = 1)
        
        else : # num_cls == 2
            logging.info("Using binomial tests.")
            pv = two_sample_pvals(df[f"{self.cls_names[0]}:n"],
                df[f"{self.cls_names[1]}:n"])
            
        df['pval'] = pv
        return df

    def HCT(self, **kwrgs) :
        """
        Return results after applying HC threshold to fitted data
        Report whether a feature is selected by HC threshold

        """
        
        stbl = kwrgs.get('stbl', self.stbl)
        gamma = kwrgs.get('gamma', self.gamma)
        
        df = self.get_pvals()

        hc, thr = HC(df['pval'], stbl=stbl).HCstar(gamma=gamma)
        df['HC'] = hc
        df['thresh'] = df['pval'] < thr
        return df

    def test_cls_Poiss(self, cls_name, **kwrgs) :
        """
        HC Test of one class against the rest. Returns HC value 
        and indicates if feature is selected by HCT

        """
        stbl = kwrgs.get('stbl', self.stbl)
        gamma = kwrgs.get('gamma', self.gamma)
        
        df1 = pd.DataFrame()
        col_name_n = f"{cls_name}:n"
        col_name_T = f"{cls_name}:T" 
        df1 = self.counts_df.filter([col_name_n, col_name_T])
        df1['frequency'] = self.counts_df['n'] / self.counts_df["T"] 
            # observed feature frequency
    
        df1['pval'] = binom_test_two_sided(df1[col_name_n],
                    df1[col_name_T], df1["frequency"]) # can appx by Poisson

        hc, thr = HC(df1['pval'], stbl=stbl).HCstar(gamma=gamma)
        df1['HC'] = hc
        df1['thresh'] = df1['pval'] < thr
        df1['more'] = np.sign(df1[col_name_n] - df1[col_name_T]*df1["frequency"])
        return df1

    def test_cls(self, cls_name, **kwrgs) :
        """
        HC Test of one class against the rest. Returns HC value 
        and indicates if feature is selected by HCT

        """
        stbl = kwrgs.get('stbl', self.stbl)
        gamma = kwrgs.get('gamma', self.gamma)

        df1 = pd.DataFrame()
        col_name_n = f"{cls_name}:n"
        col_name_T = f"{cls_name}:T"
        df1 = self.counts_df.filter([col_name_n, col_name_T])
        df1['rest:n'] = self.counts_df['n'] - df1[col_name_n]
        df1['rest:T'] = self.counts_df["T"] - df1[col_name_T]

        df1['pval'] = two_sample_pvals(df1[col_name_n], df1["rest:n"])

        hc, thr = HC(df1['pval'], stbl=stbl).HCstar(gamma=gamma)
        df1['HC'] = hc
        df1['thresh'] = df1['pval'] < thr
        df1['more'] = np.sign(df1[col_name_n] / df1[col_name_T] \
                                - df1['rest:n'] / df1['rest:T'])
        return df1

    def HCT_vs_many_Poiss(self, **kwrgs) :
        """
        Apply HC threshold to fitted data in a 1-vs-many fashion
        with many Poisson tests. Returns DataFrame with columns 
        'affinity (cls)' indicating the affinity of the class to
        each feature:
         1 : more frequent features
         -1 : less frequent feature
         0 : not selected

        """

        stbl = kwrgs.get('stbl', self.stbl)
        gamma = kwrgs.get('gamma', self.gamma)
        
        dft = self.counts_df
        for nm in self.cls_names :
            col_name = f'{nm}:affinity'
            df = self.test_cls_Poiss(nm)
            df[col_name] = df['more'] * df['thresh']
            dft = dft.join(df[[col_name]])
        return dft.filter(like='affinity')


    def HCT_vs_many(self, **kwrgs) :
        """
        Apply HC threshold to fitted data in a 1-vs-many fashion
        with many binomial tests. Returns DataFrame with columns 
        'affinity (cls)' indicating the affinity of the class to
        each feature:
         1 : more frequent features
         -1 : less frequent feature
         0 : not selected

        """
        stbl = kwrgs.get('stbl', self.stbl)
        gamma = kwrgs.get('gamma', self.gamma)

        dft = self.counts_df
        for nm in self.cls_names :
            col_name = f'{nm}:affinity'
            df = self.test_cls(nm)
            df[col_name] = df['more'] * df['thresh']
            dft = dft.join(df[[col_name]])
        return dft.filter(like='affinity')

    def HCT_vs_many_filtered(self, **kwrgs) :
        """
        Same as CompareDocs.HCT_vs_many, but only return
        features selected at least once
        """
        stbl = kwrgs.get('stbl', self.stbl)
        gamma = kwrgs.get('gamma', self.gamma)
        dft = self.HCT_vs_many(gamma=gamma, stbl=stbl)
        return dft[dft.iloc[:,dft.columns.str.contains('affinity')].abs().any(axis=1)]

    def classify_naive(self, doc, HCT="all", ret_stat=False) :
        """
        Recieves a new document and a list of features.
        Uses a simple classification rule.
        """

        df = self.test_doc(doc, HCT=HCT)

        for cls in self.cls_names :
            df[f'{cls}:naive_score'] = (df['tested:n'] > 0) * df[f'{cls}:affinity']

        scores = {}
        for cls in self.cls_names :
            scores[cls] = df.loc[:, f'{cls}:naive_score'].sum()
        
        if ret_stat :
            return df

        return scores

    def test_doc(self, doc, of_cls=None, **kwrgs) : 
        """
        Test a new document against existing documents by combining
        binomial allocation P-values from each document. 
        
        Args:
        doc     dataframe representing terms in the tested doc
        of_cls  use this to indicate that the tested document is already
                represented by one of the classes in the model
        stbl    type of HC statistic to use
        gamma   parameter of HC statistic

        """
        stbl = kwrgs.get('stbl', self.stbl)
        gamma = kwrgs.get('gamma', self.gamma)
        
        dfi = self.count_words(doc)

        logging.debug(f"Doc contains {dfi.n.sum()} terms.")

        df = self.counts_df

        dfi['tested:T'] = dfi.n.sum()
        dfi = dfi.rename(columns = {'n' : 'tested:n'})
        df = df.join(dfi, how='left')
        
        for cls in self.cls_names:
            cnt1 = df['tested:n'].astype(int)
            cnt2 = df[f'{cls}:n'].astype(int)
            if of_cls == cls : # if tested document is already represented in 
                                # corpus, remove its counts to get a meaningful
                                # comparison. 
                logging.debug(f"Doc is of {of_cls}. Evaluating in Leave-out manner.")
                cnt2 -= cnt1
                assert(np.all(cnt2 >= 0))
                
            if cnt1.sum() + cnt2.sum() > 0 :
                pv, p = two_sample_pvals(cnt1, cnt2, ret_p=True)
            else :
                pv, p = cnt1 * np.nan, cnt1 * np.nan
            
            df[f'{cls}:pval'] = pv
            df[f'{cls}:score'] = -2*np.log(df[f'{cls}:pval'])
            df[f'{cls}:Fisher'] = df[f'{cls}:score'].mean()
            df[f'{cls}:HC'], pth = HC(pv, stbl=stbl).HCstar(gamma=gamma)
            df[f'{cls}:chisq'] = two_sample_chi_square(cnt1, cnt2)[0]
            more = -np.sign(cnt1 - (cnt1 + cnt2) * p)
            thresh = pv < pth
            df[f'{cls}:affinity'] = more * thresh
    
        return df

    def naive_score_doc(self, doc, HCT='all',
                        of_cls=None, **kwrgs) : 
        """
        Test a new document against existing data using a simple
        scoring algorithm based on feature affinity

        """
        stbl = kwrgs.get('stbl', self.stbl)
        gamma = kwrgs.get('gamma', self.gamma)
        
        dfi = self.count_words(doc)

        logging.debug(f"Doc contains {dfi.n.sum()} terms.")

        if HCT == 'multinomial' :
            df = self.HCT(gamma=gamma, stbl=stbl)
            thresh = df['thresh']
        elif HCT == 'one_vs_many' :
            df = self.HCT_vs_many(gamma=gamma, stbl=stbl)
        else :
            df = self.counts_df
            thresh = 1

        dfi['tested:T'] = dfi.n.sum()
        dfi = dfi.rename(columns = {'n' : 'tested:n'})
        df = df.join(dfi, how='left')
        
        for cls in self.cls_names:
            cnt1 = df['tested:n'].astype(int)
            cnt2 = df[f'{cls}:n'].astype(int)
            if of_cls == cls : # if tested document is already represented in 
                                # corpus, remove its counts to get a meaningful
                                # comparison. 
                logging.debug(f"Doc is of {of_cls}. Evaluating in Leave-our manner.")
                print(f"Doc is of {of_cls}. Evaluating in Leave-our manner.")
                cnt2 -= cnt1

            more = np.sign(cnt1 / cnt1.sum() - cnt2 / cnt2.sum())

            if HCT == 'one_vs_many' :
                thresh = df[f'{cls}:affinity']
            df[f'{cls}:score'] = more * thresh
    
        return df
