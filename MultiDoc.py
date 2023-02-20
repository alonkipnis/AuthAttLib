from sklearn.feature_extraction.text import CountVectorizer
import logging
from scipy.stats import chisquare, poisson
import pandas as pd
import numpy as np
from scipy.special import binom as nkchoose
import scipy

from twosample import bin_allocation_test
from hctest import HCtest
from goodness_of_fit_tests import two_sample_chi_square
from utils import n_most_frequent_words

def softmax(x):
    z = x - x.max()
    return np.exp(z) / np.sum(np.exp(z))


def n_label(cls):
    return f"n ({cls})"


def T_label(cls):
    return f"T ({cls})"


def multinomial_test(x, p):  # slow
    import met
    assert (len(x) == len(p))
    r_max = 1e6
    p = np.array(p) / np.sum(p)
    n = sum(x)
    if sum(x) == 0 or nkchoose(n, len(p)) > r_max:
        logging.warning(f"Number of combinations is too large."
                        " Approximating exact binomial test using chisquared test.")
        return chisquare(x, p * n)[1]
    all_multi_cases = met.all_multinom_cases(len(p), n)
    probs = scipy.stats.multinomial.pmf(x=all_multi_cases, n=n, p=p)
    pval = probs[probs <= probs[all_multi_cases.index(list(x))]].sum()
    return pval


def poisson_test_two_sided_matrix_appx(x, lm):
    return np.select([x < lm, x >= lm], [poisson.cdf(x, lm), poisson.sf(x, lm)])


def poisson_test_two_sided_matrix(x, lm):
    pl = np.select([x < lm, x > lm, x == lm], [poisson.cdf(x, lm), poisson.sf(x, lm), 1])
    pu = np.select([x < lm, x > lm], [poisson.sf(poisson.isf(pl, lm), lm), poisson.sf(x, lm)])
    return pl + pu


def poisson_test_two_sided(x, lm):
    pl = poisson.cdf(x, lm) * (x < lm) + poisson.sf(x, lm) * (x > lm) + (x == lm)
    pu = poisson.sf(poisson.isf(pl, lm), lm) * (x < lm) + poisson.sf(x, lm) * (x > lm)
    return pl + pu


class ListFeatures(object):
    """
    Returns a list of features from a list of documents
    If class labels are provided than take max_features of most frequent from each class
    """
    def __init__(self, **kwargs):
        self.max_features = kwargs.get('max_features', 3000)
        self.ngram_range = kwargs.get('ngram_range', (1,1))
        self.words_to_ignore = kwargs.get('words_to_ignore', [])
        self.pattern = kwargs.get('pattern', r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]")

    def __call__(self, lo_texts, class_labels=None):
        return self.get_most_frequent(lo_texts, class_labels)

    def get_most_frequent(self, lo_texts, class_labels=None):
        if class_labels is not None:
            assert len(class_labels) == len(lo_texts), "Length of class labels does not match length of data"
            vocab = []
            for cls in class_labels:
                vocab_cls = n_most_frequent_words([d for c,d in zip(class_labels, lo_texts) if c == cls],
                                             n=self.max_features, words_to_ignore=self.words_to_ignore,
                                             ngram_range=self.ngram_range, pattern=self.pattern)
                vocab += vocab_cls
            return list(np.unique(vocab))
        else:
            vocab = n_most_frequent_words(lo_texts, n=self.max_features,
                                          words_to_ignore=self.words_to_ignore,
                                          ngram_range=self.ngram_range)
            return vocab


class CompareDocs(object):
    """
    Class to compare documents by their word frequencies.
    Can select distinguishing words using Higher Criticism threshold. 
    Can also test the similarity of a new document using binomial allocation P-values and
    higher criticism as a similarity index.
    """

    def __init__(self, **kwargs):
        self.vocabulary = kwargs.get('vocabulary', [])
        self.max_features = kwargs.get('max_features', 3000)
        self.ngram_range = kwargs.get('ngram_range', (1, 1))
        self.pattern = kwargs.get('pattern', r"\b\w\w+\b")

        self.counts_df = pd.DataFrame()
        self.num_of_cls = np.nan
        self.cls_names = []
        self.measures = ['HC', 'Fisher', 'chisq']

        if len(self.vocabulary) == 0:
            self.count_vectorizer = CountVectorizer(max_features=self.max_features,
                                                    token_pattern=self.pattern, ngram_range=self.ngram_range)
        else:
            self.count_vectorizer = CountVectorizer(vocabulary=self.vocabulary,
                                                    token_pattern=self.pattern, ngram_range=self.ngram_range)

    def count_words(self, data):
        """
        Count occurrences of features.
        """
        df = pd.DataFrame()
        if type(data) == pd.DataFrame:  # each row indicates the occurrence of a feature
            df_vocab = pd.DataFrame({'feature': self.vocabulary}) \
                .set_index('feature')
            df = pd.DataFrame(data.feature.value_counts()) \
                .rename(columns={'feature': 'n'})
            return df_vocab.join(df, how='left').fillna(0).astype(int)

        tf = self.count_vectorizer.fit_transform([data])
        vocab = self.count_vectorizer.get_feature_names_out()
        tc = np.array(tf.sum(0))[0].astype(int)

        df = pd.concat([df, pd.DataFrame({'feature': vocab, 'n': tc})]) \
            .set_index('feature')
        return df.astype(int)

    def fit(self, data):
        """
        Count words and populate contingency table of each class
        Compute sum-of-squares

        Params:
        :data:    dictionary. One entry per class. Values : string
        """

        df = pd.DataFrame()
        assert (len(self.vocabulary) > 0), "Vocabulary is empty."

        df['feature'] = self.vocabulary
        df['n'] = 0
        df['T'] = 0
        df = df.set_index('feature')

        def n_label(cls):
            return f"{cls}:n"

        def T_label(cls):
            return f"{cls}:T"

        for cls in data:
            assert (cls != 'tested'), "Cannot use `tested` as a class name"
            assert (":" not in cls), "Cannot use `:` inside class names"

            self.cls_names += [cls]
            logging.debug(f"Processing {cls}...")
            txt = data[cls]
            dfi = self.count_words(txt)
            dfi['n'] = dfi['n'].astype(int)
            logging.debug(f"Found {dfi.n.sum()} terms.")
            df['n'] += dfi.n
            df['T'] += dfi.n.sum()

            dfi[T_label(cls)] = dfi.n.sum()
            dfi = dfi.rename(columns={'n': n_label(cls)})
            df = df.join(dfi, how='left', rsuffix=cls)
            values = {n_label(cls): 0, T_label(cls): max(df[T_label(cls)])}
            df = df.fillna(value=values) # fill na with 0

        assert (len(df) == len(self.vocabulary)), "ERROR: some features were ignored"
        self.num_of_cls = len(self.cls_names)
        self.counts_df = df

    def get_pvals(self):
        if self.num_of_cls < 2:
            logging.error("Not enough columns.")
            return np.nan
        df = self.counts_df.copy()
        if self.num_of_cls > 2:
            logging.info("Using multinomial tests. May be slow.")

            df['x'] = df.filter(regex=r":n$").to_records(index=False).tolist()
            df['p'] = df.filter(regex=r":T$").to_records(index=False).tolist()
            pv = df.apply(lambda r: multinomial_test(r['x'], r['p']), axis=1)

        else:  # num_cls == 2
            logging.info("Using binomial tests.")
            pv = bin_allocation_test(df[f"{self.cls_names[0]}:n"],
                                     df[f"{self.cls_names[1]}:n"])
        df['pval'] = pv
        return df

    def HCT(self, **kwargs):
        """
        Apply HC threshold to fitted data
        Report whether a feature is selected by HC threshold

        """

        stbl = kwargs.get('stbl', True)
        gamma = kwargs.get('gamma', 0.2)

        df = self.get_pvals()
        hc, thr = HCtest(df['pval'], stbl=stbl).HCstar(gamma=gamma)
        df['HC'] = hc
        df['thresh'] = df['pval'] < thr
        return df

    def test_cls_Poiss(self, cls_name, **kwargs):
        """
        HC Test of one class against the rest. Returns HC value and indicate if feature is below HCT

        """
        stbl = kwargs.get('stbl', True)
        gamma = kwargs.get('gamma', .2)

        col_name_n = f"{cls_name}:n"
        col_name_T = f"{cls_name}:T"
        df1 = self.counts_df.filter([col_name_n, col_name_T])
        df1['frequency'] = self.counts_df['n'] / self.counts_df["T"]
        # observed feature frequency

        df1['pval'] = poisson_test_two_sided(df1[col_name_n],
                                             df1[col_name_T] * df1["frequency"])

        hc, thr = HCtest(df1['pval'], stbl=stbl).HCstar(gamma=gamma)
        df1['HC'] = hc
        df1['thresh'] = df1['pval'] < thr
        df1['more'] = np.sign(df1[col_name_n] - df1[col_name_T] * df1["frequency"])
        return df1

    def test_cls(self, cls_name, **kwargs):
        """
        HC Test of one class against the rest. Returns HC value and indicate if a feature is below
        HCT or not

        """
        stbl = kwargs.get('stbl', True)
        gamma = kwargs.get('gamma', 0.2)

        col_name_n = f"{cls_name}:n"
        col_name_T = f"{cls_name}:T"
        df1 = self.counts_df.filter([col_name_n, col_name_T])
        df1['rest:n'] = self.counts_df['n'] - df1[col_name_n]
        df1['rest:T'] = self.counts_df["T"] - df1[col_name_T]

        df1['pval'] = bin_allocation_test(df1[col_name_n], df1["rest:n"])

        hc, thr = HCtest(df1['pval'], stbl=stbl).HCstar(gamma=gamma)
        df1['HC'] = hc
        df1['thresh'] = df1['pval'] <= thr
        df1['more'] = np.sign(df1[col_name_n] / df1[col_name_T] \
                              - df1['rest:n'] / df1['rest:T'])
        return df1

    def HCT_vs_many(self, **kwargs):
        """
        Apply HC threshold to fitted data in a 1-vs-many fashion
        with many binomial tests. Returns DataFrame with columns 
        'affinity (cls)' indicating the affinity of the class to
        each feature:
         1 : more frequent features
         -1 : less frequent feature
         0 : not selected

        """
        stbl = kwargs.get('stbl', True)
        gamma = kwargs.get('gamma', 0.2)
        ret_only_selected = kwargs.get('ret_only_selected', False)
        save_mask = kwargs.get('save_mask', True)

        dft = self.counts_df
        for nm in self.cls_names:
            col_name = f'{nm}:affinity'
            col_name_pval = f'{nm}:pval'
            df = self.test_cls(nm, stbl=stbl, gamma=gamma)
            df[col_name] = df['more'] * df['thresh']
            df[col_name_pval] = df['pval']
            dft = dft.join(df[[col_name, col_name_pval]])
            if save_mask:
                self.counts_df[f'{nm}:mask'] = dft[f'{nm}:affinity'] != 0

        if ret_only_selected:
            dft = dft[dft.iloc[:,  # only use features selected at least once
                      dft.columns.str.contains('affinity')].abs().any(axis=1)]
        return dft

    def test_doc_mask(self, doc, of_cls=None, **kwargs):

        stbl = kwargs.get('stbl', True)
        gamma = kwargs.get('gamma', 0.2)

        dfi = self.count_words(doc)
        logging.debug(f"Doc contains {dfi.n.sum()} terms.")
        df = self.counts_df
        assert (len(df) == len(dfi)), "count_words must use the same vocabulary"

        dfi['tested:T'] = dfi.n.sum()
        dfi = dfi.rename(columns={'n': 'tested:n'})
        df = df.join(dfi, how='left')
        res = {}

        for cls in self.cls_names:
            mask = df[f'{cls}:mask']
            dfm = df.loc[mask, :]
            logging.debug(f"Applying feature mask: using {len(dfm)} features")

            cnt1 = dfm['tested:n'].astype(int)
            cnt2 = dfm[f'{cls}:n'].astype(int)
            if of_cls == cls:  # if tested document is already represented in
                # corpus, remove its counts to get a meaningful
                # comparison.
                logging.debug(f"Doc is of {of_cls}. Evaluating in a Leave-out manner.")
                cnt2 -= cnt1
                assert (np.all(cnt2 >= 0))
                dfm.loc[:, f'{cls}:n'] = cnt2
                dfm.loc[:, 'n'] -= cnt2
                dfm.loc[:, 'T'] -= np.sum(cnt2)

            if cnt1.sum() + cnt2.sum() > 0:
                pv, p = bin_allocation_test(cnt1, cnt2, ret_p=True)
            else:
                pv, p = cnt1 * np.nan, cnt1 * np.nan

            dfm[f'{cls}:pval'] = pv
            dfm[f'{cls}:score'] = -2 * np.log(dfm[f'{cls}:pval'])
            obs = cnt1
            ex = cnt2 * dfm['tested:T'] / dfm[f'{cls}:T']
            dfm[f'{cls}:chisq'] = (obs - ex) ** 2 / ex
            more = -np.sign(cnt1 - (cnt1 + cnt2) * p)

            hc, pth = HCtest(pv, stbl=stbl).HCstar(gamma=gamma)
            dfm[f'{cls}:affinity'] = more * (pv < pth)

            fisher = dfm[f'{cls}:score'].mean()
            chisq = two_sample_chi_square(cnt1, cnt2)[0]

            res[cls] = {'df': dfm,
                        'hc': hc,
                        'fisher': fisher,
                        'chisq': chisq
                        }
        return res

    def test_doc(self, doc, of_cls=None, **kwargs):
        """
        Test a new document against existing documents by combining
        P-values from each document.
        
        Params:
        :doc:     dataframe representing terms in the tested doc
        :of_cls:  use this to indicate that the tested document is already
                represented by one of the classes in the model
        :stbl:    type of HC statistic to use
        :gamma:   parameter of HC statistic
        """

        stbl = kwargs.get('stbl', True)
        gamma = kwargs.get('gamma', 0.2)

        dfi = self.count_words(doc)
        logging.debug(f"Doc contains {dfi.n.sum()} terms.")

        self.HCT_vs_many() # evaluate mask
        df = self.counts_df.copy()
        assert (len(df) == len(dfi)), "count_words must use the same vocabulary"

        dfi['tested:T'] = dfi.n.sum()
        dfi = dfi.rename(columns={'n': 'tested:n'})
        df = df.join(dfi, how='left')

        for cls in self.cls_names:
            cnt1 = df['tested:n'].astype(int)
            cnt2 = df[f'{cls}:n'].astype(int)
            if of_cls == cls:  # if tested document is already represented in
                # corpus, remove its counts to get a meaningful
                # comparison.
                logging.debug(f"Doc is of {of_cls}. Evaluating in a Leave-out manner.")
                cnt2 -= cnt1
                assert (np.all(cnt2 >= 0))

            if cnt1.sum() + cnt2.sum() > 0:
                pv, p = bin_allocation_test(cnt1, cnt2, ret_p=True)
            else:
                pv, p = cnt1 * np.nan, cnt1 * np.nan

            df[f'{cls}:pval'] = pv
            df[f'{cls}:score'] = -2 * np.log(df[f'{cls}:pval'])
            df[f'{cls}:Fisher'] = df[f'{cls}:score'].mean()  # the mean of Fisher's method is equivalent
            # to dividing score by number of Dof
            df[f'{cls}:HC'], pth = HCtest(pv, stbl=stbl).HCstar(gamma=gamma)
            df[f'{cls}:chisq'] = two_sample_chi_square(cnt1, cnt2)[0]
            more = -np.sign(cnt1 - (cnt1 + cnt2) * p)
            mask = pv < pth
            df[f'{cls}:affinity'] = more * mask

        return df

    def predict_proba(self, doc: str, similarity_stats='HC'):
        """
        Find similarity score of doc with respect to each class and apply softmax on its negative value.
        Note: there is no good probability model here and this function is here only for compatibility with
        sklearn classifiers.
        """
        test_stat = self.test_doc(doc)
        similarities = test_stat.iloc[:, test_stat.columns.str.contains(similarity_stats)].mean()
        return softmax(-similarities)

    def predict(self, doc: str, similarity_stats='HC') -> str:
        """
        Returns the class most similar to doc based on similarity stats
        """
        r = self.predict_proba(doc, similarity_stats)
        return r.idxmax().split(':')[0]

    def naive_score_doc(self, doc, HCT='all', of_cls=None, **kwargs):
        """
        Test a new document against existing data using a simple
        scoring algorithm based on feature affinity
        """

        stbl = kwargs.get('stbl', True)
        gamma = kwargs.get('gamma', 0.2)

        dfi = self.count_words(doc)

        logging.debug(f"Doc contains {dfi.n.sum()} terms.")

        if HCT == 'multinomial':
            df = self.HCT(gamma=gamma, stbl=stbl)
            thresh = df['thresh']
        elif HCT == 'one_vs_many':
            df = self.HCT_vs_many(gamma=gamma, stbl=stbl)
        else:
            df = self.counts_df
            thresh = 1

        dfi['tested:T'] = dfi.n.sum()
        dfi = dfi.rename(columns={'n': 'tested:n'})
        df = df.join(dfi, how='left')

        for cls in self.cls_names:
            cnt1 = df['tested:n'].astype(int)
            cnt2 = df[f'{cls}:n'].astype(int)
            if of_cls == cls:  # if tested document is already represented in
                # corpus, remove its counts to get a meaningful
                # comparison.
                logging.debug(f"Doc is of {of_cls}. Evaluating in Leave-out manner.")
                cnt2 -= cnt1

            more = np.sign(cnt1 / cnt1.sum() - cnt2 / cnt2.sum())

            if HCT == 'one_vs_many':
                thresh = df[f'{cls}:affinity']
            df[f'{cls}:score'] = more * thresh

        return df
