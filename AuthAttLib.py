import pandas as pd
import numpy as np
import scipy
from tqdm import *
import re
from scipy.spatial.distance import cosine

from HC_aux import hc_vals, two_counts_pvals
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



def two_sample_chi_square(c1, c2):
    """returns the Chi-Square score of the two samples c1 and c2 (representing counts)
    (c1 and c2 are assumed to be numpy arrays of equal length)"""
    T1 = c1.sum()
    T2 = c2.sum()
    k1 = np.sqrt(T2 / T1)
    k2 = np.sqrt(T1 / T2)
    k1 = 1
    k2 = 1

    C = (k1 * c1 / T1 - k2 * c2 / T2)**2 / (c1 / T1 + c2 / T2 + 1e-20)
    return np.sum(C)


def cosine_sim(c1, c2):
    """
    returns the cosine similarity of the two sequences
    (c1 and c2 are assumed to be numpy arrays of equal length)
    """
    return cosine(c1, c2)


def to_dtm(doc_term_counts):
    """
       Convert a dataframe in the form author|doc_id|term|n to 
       a doc-term matrix, feature_names list, doc_id list
    """
    mat = doc_term_counts.pivot_table(index='doc_id',
                                      columns='term',
                                      values=['n'],
                                      fill_value=0).n
    feature_names = mat.columns.tolist()
    doc_id = mat.index.tolist()
    dtm = scipy.sparse.lil_matrix(mat.values)
    return dtm, feature_names, doc_id


def change_vocab(dtm, old_vocab, new_vocab):
    """
       Switch columns in doc-term-matrix dtm according to new_vocab 
       Words not in new_vocab are ignored
       'dtm' is a document-term matrix (sparse format)
       'old_vocab' and 'new_vocab' are lists of words 
    """

    new_dtm = scipy.sparse.lil_matrix(np.zeros((dtm.shape[0], len(new_vocab))))
    for i, w in enumerate(new_vocab):
        try:
            new_dtm[:, i] = dtm[:, old_vocab.index(w)]
        except:
            None
    return new_dtm


def n_most_frequent_words(texts, n, words_to_ignore=[], ngram_range=(1, 1)):
    """
        Returns the 'n' most frequent tokens in the corpus represented by the 
        list of strings 'texts'
    """

    from sklearn.feature_extraction.text import CountVectorizer

    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"

    tf_vectorizer = CountVectorizer(stop_words=words_to_ignore,
                                    token_pattern=pat,
                                    ngram_range=ngram_range)
    tf = tf_vectorizer.fit_transform(list(texts))
    feature_names = np.array(tf_vectorizer.get_feature_names())

    idcs = np.argsort(-tf.sum(0))
    vocab_tf = np.array(feature_names)[idcs][0]
    return list(vocab_tf[:n])


def frequent_words_tfidf(texts, no_words, words_to_ignore=[]):
    """
        Returns the 'no_words' with the LOWEST tf-idf score.
        Useful in removing proper names and rare words. 
    """

    tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                       min_df=0,
                                       sublinear_tf=True,
                                       stop_words=words_to_ignore)
    tfidf = tfidf_vectorizer.fit_transform(list(texts))
    feature_names = tfidf_vectorizer.get_feature_names()

    idcs = np.argsort(tfidf.sum(0))
    vocab_tfidf = np.array(feature_names)[idcs][0]
    return vocab_tfidf[-no_words:]


def term_counts(text, vocab=[], symbols=[]):
    """ return a dataframe of the form feature|n representing 
        counts of terms in text and symbols in text. 
        If vocab = [] use all words in text as the vocabulary.
        """
    from sklearn.feature_extraction.text import CountVectorizer

    df = pd.DataFrame()

    for ch in symbols:
        n1 = len(re.findall(ch, text))
        df = df.append({'feature': ch, 'n': n1}, ignore_index=True)

    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"
    # term counts
    if len(vocab) == 0:
        tf_vectorizer = CountVectorizer(token_pattern=pat, max_features=500)
    else:
        tf_vectorizer = CountVectorizer(token_pattern=pat, vocabulary=vocab)
    tf = tf_vectorizer.fit_transform([text])
    tc = np.array(tf.sum(0))[0]

    df = pd.concat([df, pd.DataFrame({'feature': vocab, 'n': tc})])
    return df


def text_df_to_dtc_df(data, ngram_range=(1, 1), vocab=[], vocab_size=500):
    from sklearn.feature_extraction.text import CountVectorizer

    df = pd.DataFrame()
    for r in data.iterrows():
        text = r[1].text
        pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"
        # term counts
        if len(vocab) == 0:
            tf_vectorizer = CountVectorizer(token_pattern=pat,
                                            max_features=vocab_size,
                                            ngram_range=ngram_range)
        else:
            tf_vectorizer = CountVectorizer(token_pattern=pat,
                                            vocabulary=vocab)
        tf = tf_vectorizer.fit_transform([text])
        tc = tf.sum(0)
        vocab = tf_vectorizer.get_feature_names()

        df = df.append(pd.DataFrame({
            'doc_id': r[1].doc_id,
            'author': r[1].author,
            'dataset': r[1].dataset,
            'term': vocab,
            'n': np.array(tc)[0]
        }),
                       ignore_index=True)
    return df


def to_docTermCounts(lo_texts, vocab=[], max_features=500, ngram_range=(1, 1)):
    """
       converts list of strings to a doc-term matrix
       returns term-counts matrix (sparse) and a list of feature names
    """

    from sklearn.feature_extraction.text import CountVectorizer

    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"

    if len(vocab) == 0:
        tf_vectorizer = CountVectorizer(max_features=max_features,
                                        token_pattern=pat,
                                        ngram_range=ngram_range)
    else:
        tf_vectorizer = CountVectorizer(vocabulary=vocab,
                                        token_pattern=pat,
                                        ngram_range=ngram_range)

    tf = tf_vectorizer.fit_transform(lo_texts)
    feature_names = tf_vectorizer.get_feature_names()

    return tf, feature_names


def change_dtm_dictionary(dtm, old_vocab, new_vocab):
    """
       Switch columns in doc-term matrix according to new_vocab 
       Words not in new_vocab are ignored
       dtm is a document-term matrix (sparse format)
       old_vocab and new_vocab are lists of words (without duplicaties)
    """

    new_dtm = scipy.sparse.lil_matrix(np.zeros((dtm.shape[0], len(new_vocab))))
    for i, w in enumerate(new_vocab):
        try:
            new_dtm[:, i] = dtm[:, old_vocab.index(w)]
        except:
            None
    return new_dtm


class DocTermTable(object):
    """ Facilitate p-value, Higher Criticism (HC), and cosine similarity
    computations with with respect to a document-term matrix.
 
    Args: 
            dtm -- (sparse) doc-term matrix.
            feature_names -- list of names for each column of dtm.
            document_names -- list of names for each row of dtm.
            stbl -- type of HC statistic to use 
    """
    def __init__(self, dtm, feature_names=[], document_names=[], stbl=True):
        """ 
        Args: 
            dtm -- (sparse) doc-term matrix.
            feature_names -- list of names for each column of dtm.
            document_names -- list of names for each row of dtm.
            stbl -- type of HC statistic to use 
        """

        if document_names == []:
            document_names = ["doc" + str(i) for i in range(dtm.shape[0])]
        self._doc_names = dict([
            (s, i) for i, s, in enumerate(document_names[:dtm.shape[0]])
        ])
        self._feature_names = feature_names  #: list of feature names (vocabulary)
        self._dtm = dtm  #: matrix representing doc-term-counts
        self._stbl = stbl  #: type of HC score to use

        if dtm.sum() == 0:
            raise ValueError(
                "seems like all counts are zero. Did you pass the wrong data format?"
            )

        self.__compute_internal_stat()

    def __compute_internal_stat(self):
        """summarize internal doc-term-table"""

        self._terms_per_doc = np.squeeze(np.array(
            self._dtm.sum(1))).astype(int)
        self._counts = np.squeeze(np.array(self._dtm.sum(0))).astype(int)

        #keep HC score of each row w.r.t. the rest
        pv_list = self.__per_doc_Pvals()
        self._internal_scores = []
        for pv in pv_list:
            hc, p_thr = hc_vals(pv, stbl=self._stbl, alpha=0.45)
            self._internal_scores += [hc]

    def __per_doc_Pvals(self):
        """Pvals of each row in dtm with respect to the rest"""

        pv_list = []

        #pvals when removing one row at a time
        counts = self._counts
        if self._dtm.shape[0] == 1:  # internal score is undefined
            return []
        for r in self._dtm:
            c = np.squeeze(np.array(r.todense()))
            pv = two_counts_pvals(c, counts - c).pval
            pv_list += [pv.values]

        return pv_list

    def get_HC_rank_features(self, dtbl, LOO=False, within=False, stbl=None):
        """ returns the HC score of dtm1 wrt to doc-term table,
        as well as its rank among internal scores 
        Args:
        stbl -- indicates type of HC statistic
        LOO -- Leave One Out evaluation of the rank (much slower process
                but more accurate; especially when number of documents
                is small)
         """

        if stbl == None:
            stbl = self._stbl

        if within == True:
            pvals = self._get_Pvals(dtbl.get_counts(), within == True)
        else:
            pvals = self.get_Pvals(dtbl)

        HC, p_thr = hc_vals(pvals, stbl=stbl)

        pvals[np.isnan(pvals)] = 1
        feat = np.array(self._feature_names)[pvals < p_thr]

        if (LOO == False) or (within == True):
            lo_hc = self._internal_scores
            if len(lo_hc) > 0:
                rank = np.mean(np.array(lo_hc) < HC)
            else:
                rank = np.nan
            if (stbl != self._stbl):
                print("Warning: requested HC type (stable / non-stable)\
                 does not match internal HC type of table object.\
                Rank may be meaningless.")

        else:  #LOO == True
            loo_Pvals = self.per_doc_Pvals_LOO(dtbl)[
                1:]  #remove first item (corresponding to test sample)

            lo_hc = []
            if (len(loo_Pvals)) == 0:
                raise ValueError("list of loo Pvals is empty")

            for pv in loo_Pvals:
                hc, _ = hc_vals(pv, stbl=stbl)
                lo_hc += [hc]

            if len(lo_hc) > 0:
                rank = np.mean(np.array(lo_hc) < HC)
            else:
                rank = np.nan

        return HC, rank, feat

    def get_feature_names(self):
        return self._feature_names

    def get_document_names(self):
        return self._doc_names

    def get_counts(self):
        return self._counts

    def _get_Pvals(self, counts, within=False):
        """ Returns pvals from a list counts 

        Args:
            counts -- 1D array of feature counts.
            within -- indicates weather to subtracrt counts of dtm
                      from internal counts (this option is useful 
                        whenever we wish to compute Pvals of a 
                        document wrt to the rest)
        Returns:
            list of P-values
        """
        cnt0 = np.squeeze(np.array(self._counts))
        cnt1 = np.squeeze(np.array(counts))

        assert (cnt0.shape == cnt1.shape)

        if within:
            cnt2 = cnt0 - cnt1
            if np.any(cnt2 < 0):
                raise ValueError("'within == True' is invalid")
            pv = two_counts_pvals(cnt1, cnt2).pval
        else:
            pv = two_counts_pvals(cnt1, cnt0).pval
        return pv.values

    def get_Pvals(self, dtbl):
        """ return a list of p-values of another DocTermTable with 
        respect doc-term table.

        Args: 
            dtbl -- DocTermTable with respect to which to compute 
                    pvals
        """

        if dtbl._feature_names != self._feature_names:
            print(
                "Warning: features of 'dtbl' do not match object",
                " Changing dtbl accordingly. "
            )
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)
            print("Completed.")

        return self._get_Pvals(dtbl.get_counts())

    def per_doc_Pvals_LOO(self, dtbl):
        """ return a list of internal pvals after adding another 
        table to the current one.

        Args:
            dtbl -- DocTermTable to add and to compute score with
                    respect to it
        """

        if dtbl._feature_names != self._feature_names:
            print(
                "Warning: features of 'dtbl' do not match object. Changing dtbl accordingly. "
            )
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)
            print("Completed.")

        return self.__per_doc_Pvals_LOO(dtbl._dtm)

    def change_vocabulary(self, new_vocabulary):
        new_dtm = scipy.sparse.lil_matrix(
            np.zeros((self._dtm.shape[0], len(new_vocabulary))))
        old_vocab = self._feature_names

        no_missing_words = 0
        for i, w in enumerate(new_vocabulary):
            try:
                new_dtm[:, i] = self._dtm[:, old_vocab.index(w)]
            except:  #exception occurs if a word in the
                #new vocabulary does not exists in old one
                no_missing_words += 1

        self._dtm = new_dtm
        self._feature_names = new_vocabulary

        self.__compute_internal_stat()

    def __per_doc_Pvals_LOO(self, dtm1):
        pv_list = []

        dtm_all = vstack([dtm1, self._dtm]).tolil()
        #current sample corresponds to the first row in dtm_all

        #pvals when the removing one document at a time
        s1 = np.squeeze(np.array(dtm1.sum(0)))
        s = self._counts + s1
        for r in dtm_all:
            c = np.squeeze(np.array(r.todense()))  #no dense
            pv = two_counts_pvals(c, s - c).pval
            pv_list += [pv.values]

        return pv_list

    def get_doc_as_table(self, doc_id):
        """ Returns a single row in the doc-term-matrix as a new 
        DocTermTable object. 

        Args:
            doc_id -- row identifier.

        Returns:
            DocTermTable object
        """
        dtm = self._dtm[self._doc_names[doc_id], :]
        new_table = DocTermTable(dtm,
                                 feature_names=self._feature_names,
                                 document_names=[doc_id],
                                 stbl=self._stbl)
        return new_table

    def get_ChiSquare(self, dtbl):
        """
        Args: 
            dtbl -- DocTermTable with respect to which to compute 
                    score

        Returns ChiSquare score 
        """
        if dtbl._feature_names != self._feature_names:
            print(
                "Warning: features of 'dtbl' do not match object. Changing dtbl accordingly. "
            )
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)
            print("Completed.")
        return two_sample_chi_square(self._counts, dtbl._counts)

    def get_CosineSim(self, dtbl):
        """ Returns the cosine similarity of another DocTermTable object 
        'dtbl' with respect to the current one
        """
        if dtbl._feature_names != self._feature_names:
            print(
                "Warning: features of 'dtbl' do not match object. Changing dtbl accordingly. "
            )
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)
            print("Completed.")
        return cosine_sim(self._counts, dtbl._counts)


class AuthorshipAttributionMulti(object):
    """
    model for text classification using word frequency data and HC-based 
    testing. 

    Args:
        data -- is a DataFrame with columns doc_id|author|text
                  (author represents the class idnetifyer or 
                    training label).
        vocab -- reduce word counts for this set (unless vocab == []).
        vocab_size -- extract vocabulary of this size 
                      (only if vocab == []).
        ngram_range -- ngram parameter for term tf_vectorizer
        stbl -- a parameter determinining type of HC statistic.
        words_to_ignore -- tell tokenizer to ignore words in
                           this list.

    """
    def __init__(self,
                 data,
                 vocab=[],
                 vocab_size=100,
                 words_to_ignore=[],
                 ngram_range=(1, 1),
                 stbl=True):
        """
        Args:
            data -- is a DataFrame with columns doc_id|author|text
                      (author represents the class idnetifyer or 
                        training label)
            vocab -- reduce word counts for this set (unless vocab == [])
            vocab_size -- extract vocabulary of this size 
                          (only if vocab == [])
            ngram_range -- ngram parameter for term tf_vectorizer
            stbl -- a parameter determinining type of HC statistic
            words_to_ignore -- tell tokenizer to ignore words in
                               this list
        """

        self._AuthorModel = {}  #:  list of DocTermTable objects, one for
        #: each author.
        self._vocab = vocab  #: joint vocabulary for the model.
        self._ngram_range = ngram_range  #: the n-gram range of text
        #: in the model.
        self._stbl = stbl  #:  type of HC statistic to use.

        if self._vocab == []:  #common vocabulary
            vocab = n_most_frequent_words(list(data.text),
                                          n=vocab_size,
                                          words_to_ignore=words_to_ignore,
                                          ngram_range=self._ngram_range)
        self._vocab = vocab

        #compute author-models

        lo_authors = pd.unique(data.author)
        for auth in lo_authors:
            data_auth = data[data.author == auth]
            print("\t Creating author-model for {} using {} features"\
                .format(auth, len(self._vocab)))

            self._AuthorModel[auth] = self.to_docTermTable(
                list(data_auth.text), document_names=list(data_auth.doc_id))
            print("\t\tfound {} documents and {} relevant tokens"\
            .format(len(data_auth),
                self._AuthorModel[auth]._counts.sum()))

        #self.compute_author_models()

    def to_docTermTable(self, X, document_names=[]):
        """Convert raw input X into a DocTermTable object. 

        Override this fucntion to process other input format. 

        Args:     
            X -- list of texts 
            document_names -- list of strings representing the names
                              of each text in X

        Returs:
            DocTermTable object
        """

        dtm, _ = to_docTermCounts(X,
                                  vocab=self._vocab,
                                  ngram_range=self._ngram_range)
        return DocTermTable(dtm,
                            feature_names=self._vocab,
                            document_names=document_names,
                            stbl=self._stbl)

    def re_compute_author_models(self):
        """ compute author models after a change in vocab """

        for auth in self._AuthorModel:
            am = self._AuthorModel[auth]
            am.change_vocabulary(self._vocab)
            print("Changed vocabulary for {}. Found {} relevant tokens"\
                .format(auth, am._counts.sum()))

    def predict(self, X, method='HC', unk_thresh=1e6, LOO=False):
        """
        Attribute text X with one of the authors or '<UNK>'. 

        Args:
        X -- list of texts representing the test document (or corpus)
            method -- designate which score to use
            unk_thresh -- minimal score below which the text is 
            attributed to one of the authors in the model and not assigned
            the label '<UNK>'.
        LOO -- indicates whether to compute rank in a leave-of-out mode
            It leads to more accurate rank-based testing but require more 
            computations. 

        Returns:
            pred -- one of the lo_authors or '<UNK>'
            marg -- ratio of second smallest score to smallest scroe

        Note:
            Currently scores 'HC', 'rank' and 'cosine'
            are supported. 
        """

        if len(self._AuthorModel) == 0:
            raise IndexError("no pre-trained author models found")
            return None

        cand = '<UNK>'
        min_score = unk_thresh
        margin = unk_thresh

        Xdtb = self.to_docTermTable(X)

        for i, auth in enumerate(self._AuthorModel):
            am = self._AuthorModel[auth]
            score, rank, feat = am.get_HC_rank_features(Xdtb, LOO=LOO)
            chisq = am.get_ChiSquare(Xdtb)
            cosine = am.get_CosineSim(Xdtb)

            if method == 'rank':
                score = rank
            elif method == 'cosine':
                score = cosine

            if score < min_score: # find new minimum
                margin = min_score / score
                min_score = score
                cand = auth
            elif score / min_score < margin : # make sure to track the margin
                margin = score / min_score
        return cand, margin

    def internal_stats_corpus(self):
        """Compute scores of each pair of corpora within the model.
            
        Returns: 
            a dataframe with rows: 
            doc_id|author|HC|ChiSq|cosine|rank|wrt_author

            doc_id -- the document identifyer.
            wrt_author -- author of the corpus against which the
                          document is tested.
            HC, ChiSq, cosine -- HC score, Chi-Square score, and 
                    cosine similarity, respectively, between the 
                    document and the corpus.
            rank -- the rank of the HC score compared to other
                    documents within the corpus.
        """

        
        df = pd.DataFrame()

        for auth0 in tqdm(self._AuthorModel): # go over all corpora
            md0 = self._AuthorModel[auth0]
            for auth1 in self._AuthorModel:  # go over all corpora
                if auth0 < auth1:            # consider each pair only once
                    md1 = self._AuthorModel[auth1]
                    HC, rank, feat = md0.get_HC_rank_features(md1)
                    chisq = md0.get_ChiSquare(md1)
                    cosine = md0.get_CosineSim(md1)
                    df = df.append(
                        {
                            'author': auth1,
                            'wrt_author': auth0,
                            'HC': HC,
                            'ChiSq': chisq,
                            'cosine': cosine,
                            'no_docs': len(md1._counts),
                            'no_tokens': md1._counts.sum(),
                            'feat': list(feat)
                        },
                        ignore_index=True)
        return df

    def internal_stats(self, lo_authors=[], LOO=False):
        """
        Compute scores of each document with respect to the corpus of
        each author. When tested against its own corpus, the document
        is removed from that corpus. 
        
        Args:
        lo_authors -- subset of the authors in the model with respect
                to which the scores of each document are evaluated.
                If empty, evaluate with respect to all authors.
        LOO -- indicates whether to compute rank in a leave-of-out mode
            It leads to more accurate rank-based testing but require more 
            computations.

        Returns: 
            Pandas dataframe with rows: 
            doc_id|author|HC|ChiSq|cosine|rank|wrt_author
            where:
            doc_id -- the document identifyer. 
            wrt_author -- author of the corpus against which the
                          document is tested.
            HC, ChiSq, cosine -- HC score, Chi-Square score, and cosine
                    similarity, respectively, between the document and
                    the corpus.
            rank -- the rank of the HC score compared to other documents 
            within the corpus.
        """

        df = pd.DataFrame()

        if lo_authors == []:
            lo_authors = self._AuthorModel

        for auth0 in tqdm(lo_authors):
            md0 = self._AuthorModel[auth0]
            for auth1 in self._AuthorModel:
                md1 = self._AuthorModel[auth1]
                lo_docs = md1.get_document_names()
                for dn in lo_docs:
                    dtbl = md1.get_doc_as_table(dn)
                    chisq = md0.get_ChiSquare(dtbl)
                    cosine = md0.get_CosineSim(dtbl)
                    if auth0 == auth1:
                        HC, rank, feat = md0.get_HC_rank_features(dtbl,
                                                                  LOO=LOO,
                                                                  within=True)
                    else:
                        HC, rank, feat = md0.get_HC_rank_features(dtbl,
                                                                  LOO=LOO)
                    df = df.append(
                        {
                            'doc_id': dn,
                            'author': auth1,
                            'wrt_author': auth0,
                            'HC': HC,
                            'ChiSq': chisq,
                            'cosine': cosine,
                            'rank': rank,
                            'feat': list(feat)
                        },
                        ignore_index=True)
        return df

    def predict_stats(self, x, LOO=False):
        """ Returns a pandas dataframe with columns representing the 
        statistics: HC score, ChiSquare, rank (of HC), cosine similarity
        where each one is obtained by comparing the input text 'x' to each
        corpus in the model.
        
        Args:
            x -- input text (list of strings)
            LOO -- indicates whether to compute rank in a leave-of-out
                    mode. It leads to more accurate rank-based testing 
                    but require more computations.

        Returns:
            dataframe with rows: 
            doc_id|author|HC|ChiSq|cosine|rank|wrt_author

            doc_id -- the document identifyer.
            wrt_author -- author of the corpus against which the
                         document is tested.
            HC, ChiSq, cosine -- HC score, Chi-Square score, and cosine
                                 similarity, respectively, between the 
                                 document and the corpus.
            rank -- the rank of the HC score compared to other documents 
                    within the corpus.
        """
        # provides statiscs on decision wrt to test sample text
        xdtb = self.to_docTermTable(x)

        df = pd.DataFrame()
        for auth in self._AuthorModel:
            md = self._AuthorModel[auth]
            HC, rank, feat = md.get_HC_rank_features(xdtb, LOO=LOO)
            chisq = md.get_ChiSquare(xdtb)
            cosine = md.get_CosineSim(xdtb)
            predict = self.predict(x)
            df = df.append(
                {
                    'wrt_author': auth,
                    'HC': HC,
                    'ChiSq': chisq,
                    'rank': rank,
                    'feat': feat,
                    'cosine': cosine,
                    'predict': predict,
                },
                ignore_index=True)
        return df

    def predict_stats_list(self, X, document_names=[], LOO=False):
        """
            Same as predict_stats but X represents a list of documents.
            Each document in X is tested seperately.
        """
        res = pd.DataFrame()
        if document_names == []:
            document_names = ["doc" + str(i + 1) for i in list(range(len(X)))]
        for j, x in enumerate(X):
            df = self.predict_stats(x)
            df.loc[:, 'doc_id'] = document_names[j]
            res = res.append(df, ignore_index=True)
        return res

    def reduce_features(self, new_feature_set):
        """
            Update the model to a new set of features. 
        """
        self._vocab = new_feature_set
        self.re_compute_author_models()

    def reduce_features_from_model_pair(self, auth1, auth2, stbl=None):
        """
            Find list of features (tokens) discriminating auth1 and auth2
            Reduce model to those features. 
            'auth1' and 'auth2' are keys within _AuthorModel
            return new list of features
        """
        md1 = self._AuthorModel[auth1]
        md2 = self._AuthorModel[auth2]
        _, _, feat = md1.get_HC_rank_features(md2, stbl=stbl)
        print("Reducting to {} features".format(len(feat)))
        self.reduce_features(list(feat))
        return self._vocab


class AuthorshipAttributionMultiBinary(object):
    """ This class uses pair-wise tests to determine most likely author. 
        It does so by creating AuthorshipAttributionMulti object for each 
        pair of authors and reduces the features of this object. 

        The interface is similar to AuthorshipAttributionMultiBinary
    """
    def __init__(
            self,
            data,
            vocab=[],
            vocab_size=100,
            words_to_ignore=[],
            global_vocab=False,
            ngram_range=(1, 1),
            stbl=True,
            reduce_features=False,
    ):
        # train_data is a dataframe with at least fields: author|doc_id|text
        # vocab_size is an integer controlling the size of vocabulary

        self._AuthorPairModel = {}

        if global_vocab == True:
            if len(vocab) == 0:
                #get top vocab_size terms
                vocab = n_most_frequent_words(list(data.text),
                                              n=vocab_size,
                                              words_to_ignore=words_to_ignore,
                                              ngram_range=ngram_range)
        else:
            vocab = []

        lo_authors = pd.unique(data.author)  #all authors
        lo_author_pairs = [(auth1, auth2) for auth1 in lo_authors\
                             for auth2 in lo_authors if auth1 < auth2 ]

        print("Found {} author-pairs".format(len(lo_author_pairs)))
        for ap in tqdm(lo_author_pairs):  # AuthorPair model for each pair
            print("MultiBinaryAuthorModel: Creating model for {} vs {}"\
                .format(ap[0],ap[1]))

            data_pair = data[data.author.isin(list(ap))]
            ap_model = AuthorshipAttributionMulti(
                data_pair,
                vocab=vocab,
                vocab_size=vocab_size,
                words_to_ignore=words_to_ignore,
                stbl=stbl)
            if reduce_features == True:
                ap_model.reduce_features_from_model_pair(ap[0], ap[1])
            self._AuthorPairModel[ap] = ap_model

    def predict(self, x, method='HC'):
        def predict_max(df1):
            cnt = df1.pred.value_counts()
            imx = cnt.values == cnt.values.max()
            if sum(imx) == 1:
                return cnt.index[imx][0]
            else:
                return '<UNK>'

        df1 = self.predict_stats(x, LOO=False, method=method)

        predict = predict_max(df1)
        return predict

    def predict_stats(self, x, method='HC', LOO=False):
        # provides statiscs on decision wrt to test sample text
        df = pd.DataFrame()

        if len(self._AuthorPairModel) == 0:
            raise IndexError("no pre-trained author models found")

        for ap in self._AuthorPairModel:
            ap_model = self._AuthorPairModel[ap]

            pred, margin = ap_model.predict(x, method=method, LOO=LOO)
            df = df.append(
                {
                    'author1': ap[0],
                    'author2': ap[1],
                    'pred': pred,
                    'margin': margin,
                },
                ignore_index=True)
        return df
