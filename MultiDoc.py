from sklearn.feature_extraction.text import CountVectorizer
import logging
from scipy.stats import chisquare
import met

from TwoSampleHC import two_sample_pvals, HC, binom_test_two_sided

def exact_multinomial_test(x, p) : # slow
    assert(len(x) == len(p))
    n_max = 40
    p = np.array(p) / np.sum(p)
    n = sum(x)
    if n > n_max :
        return chisquare(x, p * n)[1]
    all_multi_cases = [tup[0] for tup in met.onesided_exact_likelihood(x, [1,1,1])]
    probs = scipy.stats.multinomial.pmf(x=all_multi_cases, n=n, p=p)
    pval = probs[probs <= probs[all_multi_cases.index(list(x))]].sum()
    return pval

class CompareDocs :
    def __init__(self, **kwargs) :
        self.pval_type = kwargs.get('pval_type', 'multinom')
        self.vocab = kwargs.get('vocabulary', [])
        self.max_features = kwargs.get('max_features', 1000)
        self.min_cnt = kwargs.get('min_count', 3)
        self.ng_range = kwargs.get('ngram_range', (1,1))

        self.counts_df = pd.DataFrame()
        self.num_of_docs = np.nan
        
    def count_words(self, text) :
        df = pd.DataFrame()

        pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"
        pat = r"\b\w\w+\b"
        # term counts
        if len(self.vocab) == 0:
            tf_vectorizer = CountVectorizer(token_pattern=pat, 
                                            max_features=self.max_features,
                                            ngram_range=self.ng_range
                                           )
        else:
            tf_vectorizer = CountVectorizer(token_pattern=pat,
                                            vocabulary=self.vocab,
                                            ngram_range=self.ng_range
                                           )
            
        tf = tf_vectorizer.fit_transform([text])
        vocab = tf_vectorizer.get_feature_names()
        tc = np.array(tf.sum(0))[0]

        df = pd.concat([df, pd.DataFrame({'term': vocab, 'n': tc})])
        return df
    
    def fit(self, lo_texts) :
        df = pd.DataFrame()
        if self.vocab == [] : # build vocabulry from data
            logging.debug("Building vocabulary from data")
            df_tot = self.count_words(" ".join(lo_texts))
            self.vocab = list(df_tot[df_tot.n >= self.min_cnt]['term'])
        
        df['term'] = self.vocab
        df['n'] = 0
        df = df.set_index('term')
            
        for i,txt in enumerate(lo_texts) :
            name = f"{i+1}"
            logging.debug(f"Processing {name}...")
            dfi = self.count_words(txt).set_index('term')
            logging.debug(f"Found {dfi.n.sum()} terms.")
            df['n'] += dfi.n 
            dfi['T' + name] = dfi.n.sum()
            dfi = dfi.rename(columns = {'n' : 'n' + name})
            df = df.join(dfi, how='left', rsuffix=name)
        
        self.num_of_docs = i + 1
        
        self.counts_df = df
        
        
    def HCtest(self, gamma=.2) :
        """
        To do: implement multinomial testing.
        """
        if self.num_of_docs < 2 :
            logging.error("Not enough columns.")
            return np.nan
        if self.num_of_docs > 2 :
            logging.info("Using multinomial tests. May be slow.")
            df = self.counts_df.copy()
            acc_x = []
            acc_p = []
            
            acc_x = [df[c] for c in df if c == 'n']

            df['x'] = df.filter(regex='n[0-9]').to_records(index=False).tolist()
            df['p'] = df.filter(regex='T[0-9]').to_records(index=False).tolist()
            pv = df.apply(lambda r : exact_multinomial_test(r['x'], r['p']), axis = 1)
        
        else :
            logging.info("Using binomial tests.")
            pv = two_sample_pvals(self.counts_df.n1, self.counts_df.n2)
            
        self.counts_df['pval'] = pv
        hc, thr = HC(pv).HCstar(gamma=gamma)
        self.counts_df['HC'] = hc
        self.counts_df['thresh'] = pv < thr
        return hc
        