import pandas as pd
import numpy as np
import scipy
from tqdm import *
import re
from scipy.spatial.distance import cosine

from HC_aux import hc_vals, two_counts_pvals
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

def two_sample_chi_square(c1, c2) :
    """returns the Chi-Square score of the two samples c1 and c2 (representing counts)
    (c1 and c2 are assumed to be numpy arrays of equal length)"""
    T1 = c1.sum()
    T2 = c2.sum()
    C = (np.sqrt(T2/T1) * c1 / T1 - np.sqrt(T1/T2) * c2 / T2 ) ** 2 / (c1/T1 + c2/T2 + 1e-20)
    return np.sum(C)

def cosine_sim(c1, c2) :
    """
    returns the cosine similarity of the two sequences
    (c1 and c2 are assumed to be numpy arrays of equal length)
    """
    return cosine(c1,c2)

def to_dtm(doc_term_counts) :
    """
       Convert a dataframe in the form author|doc_id|term|n to 
       a doc-term matrix, feature_names list, doc_id list
    """
    mat = doc_term_counts.pivot_table(index = 'doc_id',
     columns = 'term',
      values = ['n'],
       fill_value=0).n
    feature_names = mat.columns.tolist()
    doc_id = mat.index.tolist()
    dtm = scipy.sparse.lil_matrix(mat.values) 
    return dtm, feature_names, doc_id


def change_vocab(dtm, old_vocab, new_vocab) :
    """
       Switch columns in doc-term-matrix dtm according to new_vocab 
       Words not in new_vocab are ignored
       'dtm' is a document-term matrix (sparse format)
       'old_vocab' and 'new_vocab' are lists of words 
    """

    new_dtm = scipy.sparse.lil_matrix(np.zeros((dtm.shape[0], len(new_vocab))))
    for i,w in enumerate(new_vocab) :
        try :
            new_dtm[:,i] = dtm[:,old_vocab.index(w)]
        except :
            None
    return new_dtm


def n_most_frequent_words(texts, n, words_to_ignore = [],
                                                     ngram_range = (1,1)) :
    """
        Returns the 'n' most frequent tokens in the corpus represented by the 
        list of strings 'texts'
    """
    
    from sklearn.feature_extraction.text import CountVectorizer

    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"

    tf_vectorizer = CountVectorizer(stop_words = words_to_ignore,
                                    token_pattern = pat,
                                   ngram_range = ngram_range)  
    tf = tf_vectorizer.fit_transform(list(texts))
    feature_names = np.array(tf_vectorizer.get_feature_names())

    idcs = np.argsort(-tf.sum(0))
    vocab_tf = np.array(feature_names)[idcs][0]
    return list(vocab_tf[:n])

def frequent_words_tfidf(texts, no_words, words_to_ignore = []) :
    """
        Returns the 'no_words' with the LOWEST tf-idf score.
        Useful in removing proper names and rare words. 
    """

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', 
                         min_df = 0, sublinear_tf=True,
                          stop_words = words_to_ignore)
    tfidf = tfidf_vectorizer.fit_transform(list(texts))
    feature_names = tfidf_vectorizer.get_feature_names()

    idcs = np.argsort(tfidf.sum(0))
    vocab_tfidf = np.array(feature_names)[idcs][0]
    return vocab_tfidf[-no_words:]
        
def term_counts(text, vocab = [], symbols = []) :
    """ return a dataframe of the form feature|n representing 
        counts of terms in text and symbols in text. 
        If vocab = [] use all words in text as the vocabulary.
        """
    from sklearn.feature_extraction.text import CountVectorizer

    df = pd.DataFrame()

    for ch in symbols : 
        n1 = len(re.findall(ch, text))
        df = df.append({'feature' : ch, 'n' : n1}, ignore_index=True)
    
    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"
    # term counts
    if len(vocab) == 0 :
        tf_vectorizer = CountVectorizer(token_pattern = pat, max_features = 500)
    else :
        tf_vectorizer = CountVectorizer(token_pattern = pat, vocabulary = vocab)
    tf = tf_vectorizer.fit_transform([text])
    tc = np.array(tf.sum(0))[0]
    
    df = pd.concat([df,pd.DataFrame({'feature' : vocab, 'n' : tc})])
    return df

def text_df_to_dtc_df(data, ngram_range = (1,1), vocab = [], vocab_size = 500) :
    from sklearn.feature_extraction.text import CountVectorizer

    df = pd.DataFrame()
    for r in data.iterrows() :
        text = r[1].text
        pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"
        # term counts
        if len(vocab) == 0 :
            tf_vectorizer = CountVectorizer(token_pattern = pat,
                                            max_features = vocab_size,
                                             ngram_range = ngram_range)
        else :
            tf_vectorizer = CountVectorizer(token_pattern = pat,
                                                     vocabulary = vocab)
        tf = tf_vectorizer.fit_transform([text])
        tc = tf.sum(0)
        vocab = tf_vectorizer.get_feature_names()

        df = df.append(pd.DataFrame({'doc_id' : r[1].doc_id,
                                         'author' : r[1].author,
                                         'dataset' : r[1].dataset,
                                         'term' : vocab,
                                         'n' : np.array(tc)[0]}),
                                                     ignore_index = True )
    return df


def to_docTermCounts(lo_texts, vocab = [], max_features = 500,
                                                     ngram_range = (1,1)) :
    """
       converts list of strings to a doc-term matrix
       returns term-counts matrix (sparse) and a list of feature names
    """

    from sklearn.feature_extraction.text import CountVectorizer

    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"

    if len(vocab) == 0 :
        tf_vectorizer = CountVectorizer(max_features = max_features,
                                        token_pattern = pat,
                                     ngram_range = ngram_range) 
    else :
        tf_vectorizer = CountVectorizer(vocabulary = vocab,
                                        token_pattern = pat,
                                     ngram_range = ngram_range)
    
    tf = tf_vectorizer.fit_transform(lo_texts)
    feature_names = tf_vectorizer.get_feature_names()

    return tf, feature_names


def change_dtm_dictionary(dtm, old_vocab, new_vocab) :
    """
       Switch columns in doc-term matrix according to new_vocab 
       Words not in new_vocab are ignored
       dtm is a document-term matrix (sparse format)
       old_vocab and new_vocab are lists of words (without duplicaties)
    """

    new_dtm = scipy.sparse.lil_matrix(np.zeros((dtm.shape[0], len(new_vocab))))
    for i,w in enumerate(new_vocab) :
        try :
            new_dtm[:,i] = dtm[:,old_vocab.index(w)]
        except :
            None
    return new_dtm

class DocTermTable(object) :

    def __init__(self, dtm, feature_names = [], document_names =[], stbl = True) :
        """ dtm is a (sparse) doc-term matrix
            feature_names is the list of names for each column of dtm
            document_names is the list of names for each row of dtm
            'stbl' is an option for computing HC scores 
        """
        
        if document_names == [] :
            document_names = ["doc" + str(i) for i in range(dtm.shape[0])]
        self._doc_names = dict([(s,i) for i,s, in enumerate(document_names[:dtm.shape[0]])])
        self._feature_names = feature_names
        self._dtm = dtm #(doc-term counts)
        self._stbl = stbl
        
        if dtm.sum() == 0 :
            raise ValueError("seems like all counts are zero. Did you pass the wrong data format?")
        
        self._compute_internal_stat()
        
        
    def _compute_internal_stat(self) :
        """summarize dtc """

        self._terms_per_doc =  np.squeeze(np.array(self._dtm.sum(1))).astype(int) 
        self._counts = np.squeeze(np.array(self._dtm.sum(0))).astype(int)
        
        "keep HC score of each row w.r.t. the rest"
        pv_list = self._per_doc_Pvals()
        self._internal_scores = []
        for pv in pv_list :
            hc, p_thr = hc_vals(pv, stbl = self._stbl, alpha = 0.45)
            self._internal_scores += [hc]
        

    def _per_doc_Pvals(self) :
        """Pvals of each row in dtm with respect to the rest """
        
        pv_list = []
        
        #pvals when removing one row at a time
        counts = self._counts
        if self._dtm.shape[0]  == 1 :  # internal score is undefined 
            return []
        for r in self._dtm :
            c = np.squeeze(np.array(r.todense()))
            pv = two_counts_pvals(c, counts - c).pval
            pv_list += [pv.values]
        
        return pv_list

    def get_HC_rank_features(self, dtbl, LOO = False, within = False, stbl = None) :
        """ return the HC score of dtm1 wrt to doc-term table,
            as well as its rank among internal scores 
            'stbl' indicates type of HC statistic
            'LOO' stands for Leave One Out evaluation of the rank --
            this is a much slower process but more accurate 
            (especially with a small number of documents)
         """
        
        if stbl == None :
            stbl = self._stbl

        if within == True :
            pvals = self._get_Pvals(dtbl.get_counts(), within == True)
        else :
            pvals = self.get_Pvals(dtbl)
            
        HC, p_thr = hc_vals(pvals, stbl = stbl)
        
        pvals[np.isnan(pvals)] = 1
        feat = np.array(self._feature_names)[pvals < p_thr]


        if (LOO == False) or (within == True) :
            lo_hc = self._internal_scores
            if len(lo_hc) > 0 :
                rank = np.mean(np.array(lo_hc) < HC)
            else:
                rank = np.nan
            if (stbl != self._stbl) :
                print("Warning: requested HC type (stable / non-stable)\
                 does not match internal HC type of table object.\
                Rank may be meaningless.")

        else : #LOO == True
            loo_Pvals = self.per_doc_Pvals_LOO(dtbl)[1:]  #remove first item (corresponding to test sample)

            lo_hc = []
            if(len(loo_Pvals)) == 0 :
                raise ValueError("list of loo Pvals is empty")

            for pv in loo_Pvals :
                hc,_ = hc_vals(pv, stbl = stbl)
                lo_hc += [hc]

            if len(lo_hc) > 0 :
                rank = np.mean(np.array(lo_hc) < HC)
            else :
                rank = np.nan
            
        return HC, rank, feat

    def get_feature_names(self) :
        return self._feature_names

    def get_document_names(self) :
        return self._doc_names
    
    def get_counts(self) :
        return self._counts

    def _get_Pvals(self, counts, within = False) :
        """ counts is a 1D array of feature counts
            'within' indicates weather to subtracrt counts of dtm from internal counts 
            (this option is useful whenever we wish to compute Pvals of a document wrt to the rest)
        
        """
        cnt0 = np.squeeze(np.array(self._counts))
        cnt1 = np.squeeze(np.array(counts))
        
        assert(cnt0.shape == cnt1.shape)

        if within : 
            cnt2 = cnt0 - cnt1
            if np.any(cnt2 < 0) :
                raise ValueError("'within == True' is invalid")
            pv = two_counts_pvals(cnt1, cnt2).pval
        else :
            pv = two_counts_pvals(cnt1, cnt0).pval
        return pv.values
    
    def get_Pvals(self, dtbl) :
        """ return a list of pvals of another DocTermTable with respect doc-term table.
            Use synch = True if features of test table are not synchornized with current one
        """
        
        if dtbl._feature_names != self._feature_names :
            print("Warning: features of 'dtbl' do not match object. Changing dtbl accordingly. ")
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)
            print("Completed.")
        
        return self._get_Pvals(dtbl.get_counts())
    
    def per_doc_Pvals_LOO(self, dtbl) :
        """ return a list of internal pvals after adding another table to the current one
        """
        
        if dtbl._feature_names != self._feature_names :
            print("Warning: features of 'dtbl' do not match object. Changing dtbl accordingly. ")
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)
            print("Completed.")
        
        return self._per_doc_Pvals_LOO(dtbl._dtm)
        
    def change_vocabulary(self, new_vocabulary) :
        new_dtm = scipy.sparse.lil_matrix(np.zeros((self._dtm.shape[0], len(new_vocabulary))))
        old_vocab = self._feature_names
        
        no_missing_words = 0
        for i,w in enumerate(new_vocabulary) :
            try :
                new_dtm[:,i] = self._dtm[:, old_vocab.index(w)]
            except : #exception occurs if a word in the 
                     #new vocabulary does not exists in old one
                no_missing_words += 1

        self._dtm = new_dtm
        self._feature_names = new_vocabulary
        
        self._compute_internal_stat()    
    
    def _per_doc_Pvals_LOO(self, dtm1) :
        pv_list = []
        
        dtm_all = vstack([dtm1, self._dtm]).tolil()
        #current sample corresponds to the first row in dtm_all
                
        #pvals when the removing one document at a time
        s1 = np.squeeze(np.array(dtm1.sum(0)))
        s = self._counts + s1
        for r in dtm_all :
            c = np.squeeze(np.array(r.todense()))  #no dense
            pv = two_counts_pvals(c, s - c).pval
            pv_list += [pv.values]
        
        return pv_list
    
    def get_doc_as_table(self, doc_id) :
        """
        Returns a single row in the doc-term-matrix as a new DocTermTable 
        object. 

        'doc_id' is the rwo identifier 
        """
        dtm = self._dtm[self._doc_names[doc_id],:]
        new_table = DocTermTable(dtm,
                                 feature_names = self._feature_names,
                                 document_names = [doc_id],
                                 stbl = self._stbl)
        return new_table

    def get_ChiSquare(self, dtbl) :
        """
        Returns the ChiSquare score of another DocTermTable object 
        'dtbl' with respect to the current one
        """
        if dtbl._feature_names != self._feature_names :
            print("Warning: features of 'dtbl' do not match object. Changing dtbl accordingly. ")
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)
            print("Completed.")
        return two_sample_chi_square(self._counts,dtbl._counts)

    def get_CosineSim(self, dtbl) :
        """
        Returns the cosine similarity of another DocTermTable object 
        'dtbl' with respect to the current one
        """
        if dtbl._feature_names != self._feature_names :
            print("Warning: features of 'dtbl' do not match object. Changing dtbl accordingly. ")
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)
            print("Completed.")
        return cosine_sim(self._counts,dtbl._counts)
        
class AuthorshipAttributionMultiText(object) :
    """
        model for text classification using word frequency data and HC-based 
        testing

        'data' -- is a DataFrame with columns doc_id|author (class)|text
        'vocab' -- if not empty, reduce word counts for this set
        'vocab_size' -- if vocab is empty, extract vocabulary of this size
        'ngram_range' -- ngram parameter for term tf_vectorizer
        'stbl' -- a parameter determinining type of HC statistic

    """
    # model for authorship atribution problem
    # multiple authors with a joint vocabulary
    
    def __init__(self,
                data, 
                vocab = [],
                 vocab_size = 100, 
                 words_to_ignore =[], 
                 ngram_range = (1,1),
                 stbl = True) :
        # train_data is a dataframe with at least fields: author|doc_id|text
        # vocab_size is an integer controlling the size of vocabulary 

        self._AuthorModel = {}
        self._data = data
        self._dist_feat = []
        self._vocab = vocab
        self._ngram_range = ngram_range
        self._stbl = stbl

        if len(self._vocab) == 0: #common vocabulary
            vocab = n_most_frequent_words(list(data.text),
                                         n = vocab_size,
                           words_to_ignore = words_to_ignore,
                           ngram_range = self._ngram_range
                           )
        self._vocab = vocab        
        self.compute_author_models()
        
    def to_docTermTable(self, X, document_names = []) :
        """Convert input raw data sample X into a DocTermTable object
            with properties derived from the model.
            Override this fucntion to process other types of input. 
        """
        
        dtm, _ = to_docTermCounts(X, vocab = self._vocab,
                                         ngram_range = self._ngram_range)
        return DocTermTable(dtm, feature_names = self._vocab,
                                document_names = document_names,
                                stbl = self._stbl)
        
    def compute_author_models(self) :
        # compute model for HC similarity of each author
        lo_authors = pd.unique(self._data.author)

        for auth in lo_authors :  
            #modification: use different vocabulary for each author
            data_auth = self._data[self._data.author == auth]
            #vocab = frequent_words_tfidf(lo_texts, vocab_size)
            print("\t Creating author-model for {}".format(auth))
            
            self._AuthorModel[auth] = self.to_docTermTable(list(data_auth.text),
                                      document_names = list(data_auth.doc_id))
            print("\t\tfound {} documents, {} features, and {} relevant tokens"\
            .format(len(data_auth),len(self._vocab),
                self._AuthorModel[auth]._counts.sum()))

    def predict(self, X, method = 'max_HC', unk_thresh = 1e6, LOO = False) :
        """
            Attribute text X with one of the authors or '<UNK>'. 
            'unk_thresh' is the minimal HC score below which the text is 
            attributed to one of the authors in the model and not assigned
            the label '<UNK>'. 
            
            Currently only 'max_HC' and 'rank' prediction methods are supported

        """
        
        if len(self._AuthorModel) == 0 :
            raise IndexError("no pre-trained author models found")
            return None
            
        cand = '<UNK>'
        min_score = unk_thresh
        
        Xdtb = self.to_docTermTable(X)
        
        for i, auth in enumerate(self._AuthorModel) :
            am = self._AuthorModel[auth]
            score, rank, feat = am.get_HC_rank_features(Xdtb, LOO = LOO)
            chisq = am.get_ChiSquare(Xdtb)

            if method == 'rank' :
                score = rank

            if score < min_score :
                min_score = score
                cand = auth
        return cand
    

    def internal_stats_corpus(self) :
        """
            Compute HC, ChiSquare, and cosine similarity of each
            pair of corpora within the model. 

            Returns a dataframe with rows: 
            author|HC|ChiSq|cosine|wrt_author|no_docs|no_tokens

            'doc_id' is the document identifyer 
            'author' is the author of corpus A
            'wrt_author' is the author of corpus B 
            'HC', 'ChiSq', 'cosine' are the HC, Chi-Square, and cosine
            similarity, respectively, between corpus A and B
             'no_docs' is the total number of documnets in corpus A
             'no_tokens' is the total number of tokens in corpus A
        """

        from tqdm import tqdm
        
        # entire corpus
        df = pd.DataFrame()
        for auth0 in tqdm(self._AuthorModel) :
            md0 = self._AuthorModel[auth0]
            for auth1 in self._AuthorModel :
                if auth0 < auth1 :
                    md1 = self._AuthorModel[auth1]
                    HC, rank, feat = md0.get_HC_rank_features(md1)
                    chisq = md0.get_ChiSquare(md1)
                    cosine = md0.get_CosineSim(md1)
                    df = df.append({'author' : auth1,
                                    'wrt_author' : auth0,
                                    'HC' : HC, 
                                    'ChiSq' : chisq,
                                    'cosine' : cosine,
                                    'no_docs' : len(md1._counts),
                                    'no_tokens' : md1._counts.sum(),
                                    'feat' : list(feat)
                                   },  ignore_index=True)
        return df

    def internal_stats(self, LOO = False) :
        """
            Compute HC, ChiSquare, rank of HC, and cosine similarity of each
            document with respect to the corpus of each author.
            When tested against its own corpus, the document
            is removed from the the corpus. 

            'LOO' is used to compute the calibrate the rank computation. 
            It leads to more accurate rank-based testing but require more 
            computations. 

            Returns a dataframe with rows: 
            doc_id|author|HC|ChiSq|cosine|rank|wrt_author

            'doc_id' is the document identifyer 
            'author' is the author of the document
            'wrt_author' is the author of the corpus against which the
             document is tested
            'HC', 'ChiSq', 'cosine' are the HC, Chi-Square, and cosine
            similarity, respectively, between the document and the corpus
            'rank' is the rank of the HC score compared to the other
             documents within the corpus
        """

        # cross HC scores, rank, and features
        from tqdm import tqdm

        # individual documents
        df = pd.DataFrame()

        for auth0 in tqdm(self._AuthorModel) :
            md0 = self._AuthorModel[auth0]
            for auth1 in self._AuthorModel :
                md1 = self._AuthorModel[auth1]
                lo_docs = md1.get_document_names()
                for dn in lo_docs :
                    dtbl = md1.get_doc_as_table(dn)
                    chisq = md0.get_ChiSquare(dtbl)
                    cosine = md0.get_CosineSim(dtbl)
                    if auth0 == auth1 :
                        HC, rank, feat = md0.get_HC_rank_features(dtbl, LOO = LOO, within = True)
                    else :
                        HC, rank, feat = md0.get_HC_rank_features(dtbl, LOO = LOO)
                    df = df.append({'doc_id' : dn,
                                    'author' : auth1,
                                    'wrt_author' : auth0,
                                    'HC' : HC, 
                                    'ChiSq' : chisq,
                                    'cosine' : cosine,
                                    'rank' : rank,
                                    'feat' : list(feat)
                                   },  ignore_index=True)
        return df

    def predict_stats(self, x, LOO = False) :
        """
            Returns HC, ChiSquare, rank of HC, and cosine similarity of 
            the text of list of texts x with respect to each corpus in the 
            model. 
            
            'LOO' is used to compute the calibrate the rank computation. 
            It leads to more accurate rank-based testing but require more 
            computations.

            Returns a dataframe with rows: 
            doc_id|author|HC|ChiSq|cosine|rank|wrt_author

            'doc_id' is the document identifyer 
            'wrt_author' is the author of the corpus against which the
             document is tested
            'HC', 'ChiSq', 'cosine' are the HC, Chi-Square, and cosine
            similarity, respectively, between the document and the corpus
            'rank' is the rank of the HC score compared to the other
             documents within the corpus
        """
        # provides statiscs on decision wrt to test sample text
        xdtb = self.to_docTermTable(x)

        df = pd.DataFrame()
        for auth in self._AuthorModel :
            md = self._AuthorModel[auth]
            HC, rank, feat = md.get_HC_rank_features(xdtb, LOO = LOO)
            chisq = md.get_ChiSquare(xdtb)
            cosine = md.get_CosineSim(xdtb)
            predict = self.predict(x)
            df = df.append({'wrt_author' : auth,
                            'HC' : HC,
                            'ChiSq' : chisq,
                            'rank' : rank,
                            'feat' : feat,
                            'cosine' : cosine,
                            'predict' : predict,
                           },  ignore_index=True)
        return df

    def predict_stats_list(self, X, document_names = [], LOO = False) :
        """
            Same as predict_stats but X represents a list of documents.
            Each document in X is tested seperately.
        """
        res = pd.DataFrame()
        if document_names == [] :
            document_names = ["doc" + str(i+1) for i in list(range(len(X)))]
        for j,x in enumerate(X) :
            df = self.predict_stats(x)
            df.loc[:,'doc_id'] = document_names[j]
            res = res.append(df,ignore_index = True)
        return res

    def reduce_features(self, new_feature_set) :
        """
            Update the model to a new set of features. 
        """
        self._vocab = new_feature_set        
        self.compute_author_models()

    def reduce_features_from_model_pair(self, auth1, auth2) :
        """
            Find list of features (tokens) discriminating auth1 and auth2
            Reduce model to those features. 
            'auth1' and 'auth2' are keys within _AuthorModel
            return new list of features
        """
        md1 = self._AuthorModel[auth1]
        md2 = self._AuthorModel[auth2]
        _, _, feat = md1.get_HC_rank_features(md2)
        print("Reducting to {} features".format(len(feat)))
        self.reduce_features(list(feat))
        return self._vocab
