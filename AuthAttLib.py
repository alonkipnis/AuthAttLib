import pandas as pd
import numpy as np
from tqdm import *
from .utils import to_docTermCounts,\
 n_most_frequent_words, extract_ngrams
from .FreqTable import FreqTable
from sklearn.model_selection import train_test_split
import warnings
import scipy

import logging


class AuthorshipAttributionMulti(object):
    """
    A class to measure similarity of documents to multiple authors, 
    especially Higher Criticism-based similarity. 
    ===============================================================

    Params:
        data            is a DataFrame with columns doc_id|author|text
                        (author represents the class idnetifyer or 
                        training label).
        vocab           reduce word counts for this set (unless vocab == []).
        vocab_size      extract vocabulary of this size 
                        (only if vocab == []).
        ngram_range     ngram parameter for tf_vectorizer
        stbl            a parameter determinining type of HC statistic.
        words_to_ignore tell tokenizer to ignore words in this list.
    """
    def __init__(self, data, vocab=None, ngram_range=(1,1), **kwargs) :
        
        # model parameters: 
        #self._pval_thresh = pval_thresh
        self._verbose = kwargs.get('verbose', False)
        self._ngram_range = ngram_range  #: ng-range used
        #: in the model.
        #self._stbl = stbl  #:  type of HC statistic to use.
        #self._randomize = randomize #: randomize pvalue or not
        #self._gamma = gamma
        #self._min_cnt = min_cnt

        self._vocab = vocab
        if (self._vocab is None) or (self._vocab == []) : #get vocabulary unless supplied
            self._get_vocab(data, **kwargs)

        #compute FreqTable for each author
        self._AuthorModel = {}  #: list of FreqTable objects
        self._compute_author_models(data, **kwargs)
        self._inter_similarity = None

    def _get_vocab(self, data, **kwargs) :
        # create vocabulary form data
        # use most frequent words
        self._vocab = n_most_frequent_words(
              list(data.text), 
              n= kwargs.get('vocab_size', 100),
              words_to_ignore=kwargs.get('words_to_ignore', []),
              ngram_range=self._ngram_range
              )

    def _compute_author_models(self, data, **kwargs) :        
        lo_authors = pd.unique(data.author)
        for auth in lo_authors:
            data_auth = data[data.author == auth]
            logging.info("Creating author-model for {} using {} features..."\
                    .format(auth, len(self._vocab)))

            self._AuthorModel[auth] = self._to_docTermTable(
                                list(data_auth.text),
                                document_names=list(data_auth.doc_id),
                                **kwargs)
            
            logging.info(("\t\tfound {} documents and {} relevant tokens."\
            .format(len(data_auth),
                self._AuthorModel[auth]._counts.sum())))


    def _to_docTermTable(self, X, document_names=[], **kwargs):
        """Convert raw input X into a FreqTable object. 

        Override this fucntion to process other input format. 

        Args:     
        ------
            X                 list of texts 
            document_names    list of strings representing the names
                              of each text in X
        Returs:
        -------
            FreqTable object
        """

        dtm, _ = to_docTermCounts(X,
                            vocab=self._vocab,
                            ngram_range=self._ngram_range)

        return FreqTable(dtm,
                    column_labels=self._vocab,
                    row_labels=document_names,
                    stbl=kwargs.get('stbl', True),
                    randomize=kwargs.get('randomize', False),
                    gamma=kwargs.get('gamma', 0.25),
                    pval_thresh=kwargs.get('pval_thresh',1.1),
                    min_cnt=kwargs.get('min_cnt', 3),
                    pval_type=kwargs.get('pval_type', 'cell')
                    )

    def _recompute_author_models(self):
        """ compute author models after a change in vocab """

        for auth in self._AuthorModel:
            am = self._AuthorModel[auth]
            am.change_vocabulary(self._vocab)
            if self._verbose :
                print("Changing vocabulary for {}. Found {} relevant tokens."\
                    .format(auth, am._counts.sum()))
            logging.info("Changing vocabulary for {}. Found {} relevant tokens."\
                    .format(auth, am._counts.sum()))

    def internal_stats_corpus(self):
        """Compute scores of each pair of corpora within the model.
            
        Returns:
        -------- 
            DataFrame with rows: 
            doc_id, author, HC, ChiSq, cosine, rank, wrt_author

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

        for auth0 in tqdm(self._AuthorModel): # go over all authors
            md0 = self._AuthorModel[auth0]
            for auth1 in self._AuthorModel:  # go over all corpora
                if auth0 < auth1:       # test each pair only once
                    md1 = self._AuthorModel[auth1]
                    HC = md0.get_HC(md1)
                    chisq, chisq_pval, chisq_rank = md0.get_ChiSquare(md1)
                    cosine = md0.get_CosineSim(md1)
                    df = df.append(
                        {
                            'author': auth1,
                            'wrt_author': auth0,
                            'HC': HC,
                            'chisq': chisq,
                            'chisq_pval' : chisq_pval,
                            'chisq_rank' : chisq_rank,
                            'cosine': cosine,
                            'no_docs (author)': len(md1.get_row_labels()),
                            'no_docs (wrt_author)': len(md0.get_row_labels()),
                            'no_tokens (author)': md1._counts.sum(),
                        },
                        ignore_index=True)
        return df

    def test_doc_pval(self, doc_id, author, wrt_authors = []) :
        try :
            md0 = self._AuthorModel[author]
            lo_docs = md0.get_row_labels()
            i = lo_docs[doc_id]
            dtbl = md0.get_row_as_FreqTable(doc_id)
        except ValueError:
            logging.error("Document {} by author {}".format(doc_id,author)\
                +" has empty set of features.")
            return None

        if wrt_authors == [] :
            wrt_authors = self._AuthorModel.keys()

        df = pd.DataFrame()
        df['n'] = dtbl._counts
        df['feature'] = dtbl.get_column_labels()
        for auth1 in wrt_authors:
            md1 = self._AuthorModel[auth1]                
            if author == auth1:
                df1 = md1.two_table_HC_test(dtbl, within=True)
            else:
                df1 = md1.two_table_HC_test(dtbl, within=False)
            
            df[f'n({auth1})'] = df1.n1
            df[f'pval({auth1})'] = df1.pval
            df[f'sign({auth1})'] = np.sign(df1.n1 - (df1.n2 + df1.n1) * df1.p)

        return df


    def get_doc_stats(self, doc_id, author, wrt_authors = [], LOO = False) :
        """ 
        document statistics wrt to all authors in list wrt_authors of 
        a single document within the model. 

        """

        try :
            md0 = self._AuthorModel[author]
            lo_docs = md0.get_row_labels()
            i = lo_docs[doc_id]
            dtbl = md0.get_row_as_FreqTable(doc_id)
        except ValueError:
            print("Document {} by author {}".format(doc_id,author)\
                +" has empty set of features.")
            return None

        df = pd.DataFrame()

        if len(wrt_authors) == 0:
            # evaluate with resepct to all authors in the model
            wrt_authors = self._AuthorModel

        for auth1 in wrt_authors:
            md1 = self._AuthorModel[auth1]
                
            if author == auth1:

                HC = md1.get_HC(dtbl, within=True)
                rank = md1.get_rank(dtbl, LOO=LOO, within=True)
                chisq, chisq_pval, chisq_rank = md1.get_ChiSquare(dtbl,
                                                 within=True,
                                                 LOO_rank=LOO
                                                 )
                #bj = md1.get_BJSim(dtbl, within=True)
                CR, CR_pval, _ = md1.get_ChiSquare(
                    dtbl,
                    within=True,
                    lambda_="cressie-read")

                LL, LL_pval, LL_rank = md1.get_ChiSquare(
                    dtbl,
                    within=True,
                    lambda_="log-likelihood",
                    LOO_rank=LOO
                    )
                F = md1.get_FisherComb(dtbl, within=True)
                F = md1.get_FisherComb(dtbl, within=True)

                cosine = md1.get_CosineSim(dtbl, within=True)
            else:
                HC = md1.get_HC(dtbl)
                rank = md1.get_rank(dtbl, LOO=LOO)

                chisq, chisq_pval, chisq_rank = md1.get_ChiSquare(dtbl,
                            LOO_rank=LOO)

                CR, CR_pval, _ = md1.get_ChiSquare(
                    dtbl,
                    lambda_="cressie-read")

                LL, LL_pval, LL_rank = md1.get_ChiSquare(
                    dtbl,
                    lambda_="log-likelihood", LOO_rank=LOO)
                
                F = md1.get_FisherComb(dtbl, within=False)
                #bj = md1.get_BJSim(dtbl)
                cosine = md1.get_CosineSim(dtbl)
            df = df.append(
                {
                    'doc_id': doc_id,
                    'author': author,
                    'wrt_author': auth1,
                    'HC': HC,
                    'Fisher' : F,
                    #'BJ' : bj,
                    'chisq': chisq,
                    'chisq_rank' : chisq_rank,
                    'Cressie-Read' : CR,
                    'log-likelihood' : LL,
                    'log-likelihood_rank' : LL_rank,
                    'cosine': cosine,
                    'HC_rank': rank,
                },
                ignore_index=True)
        return df

    def compute_inter_similarity(self, **kwargs) :
        """
        Compute scores of each document with respect to the corpus of
        each author. When tested against its own corpus, the document
        is removed from that corpus. 
        
        Args:
        -----
        authors -- subset of the authors in the model. Test only documents
                belonging to these authors
        wrt_authors -- subset of the authors in the model with respect
                to which the scores of each document are evaluated.
                If empty, evaluate with respect to all authors.
        LOO -- indicates whether to compute rank in a leave-of-out mode.
            This mode provides more accurate rank-based testing but require more 
            computations.

        Sotores output in a pandas DataFrame 
        'AuthorshipAttributionMulti.doc_stats'

        """

        df = pd.DataFrame()

        authors = kwargs.get('authors', self._AuthorModel)
        wrt_authors = kwargs.get('wrt_authors', self._AuthorModel)
        LOO = kwargs.get('LOO', True)
        verbose = kwargs.get('verbose', False)
        
        for auth0 in authors :
            #tqdm(wrt_authors):
            md0 = self._AuthorModel[auth0]
            #for auth1 in self._AuthorModel:
            #    md1 = self._AuthorModel[auth1]
            lo_docs = md0.get_row_labels()
            for dn in lo_docs:
                logging.info(f"testing {dn} by {auth0} against all corpora.")
                df = df.append(self.get_doc_stats(dn, auth0,
                        wrt_authors = wrt_authors, LOO = LOO),
                        ignore_index=True)

        self._inter_similarity = df
        return self._inter_similarity



    def internal_stats(self, authors = [], wrt_authors=[], 
            LOO=False, verbose=False):
        """
        Compute scores of each document with respect to the corpus of
        each author. When tested against its own corpus, the document
        is removed from that corpus. 
        
        Args:
        authors -- subset of the authors in the model. Test only documents
                belonging to these authors
        wrt_authors -- subset of the authors in the model with respect
                to which the scores of each document are evaluated.
                If empty, evaluate with respect to all authors.
        LOO -- indicates whether to compute rank in a leave-of-out mode.
            This mode provides more accurate rank-based testing but require more 
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

        warnings.warn("Use 'AuthorshipAttributionMulti.compute_inter_similarity'"
        " and 'AuthorshipAttributionMulti.get_inter_similarity' instead.",
         DeprecationWarning, stacklevel=1)

        df = pd.DataFrame()

        if len(authors) == 0:
            # evaluate with resepct to all authors in the model
            authors = self._AuthorModel

        for auth0 in authors :
            #tqdm(wrt_authors):
            md0 = self._AuthorModel[auth0]
            #for auth1 in self._AuthorModel:
            #    md1 = self._AuthorModel[auth1]
            lo_docs = md0.get_row_labels()
            for dn in lo_docs:
                if verbose :
                    print("testing {} by {}".format(dn,auth0))
                logging.info("testing {} by {}".format(dn,auth0))
                df = df.append(self.get_doc_stats(dn, auth0,
                 wrt_authors = wrt_authors,
                  LOO = LOO), ignore_index=True)

        return df

    def predict_stats(self, x, wrt_authors=[], LOO=False):
        """ Returns DataFrame with columns representing the 
        statistics: HC score, ChiSquare, rank (of HC), cosine similarity
        where each one is obtained by comparing the input text 'x' to each
        corpus in the model.
        
        Args:
            x -- input text (list of strings)
            wrt_authors -- subset of the authors in the model with respect
                to which the scores of each document are evaluated.
                If empty, evaluate with respect to all authors.
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
        xdtb = self._to_docTermTable([x])

        if len(wrt_authors) == 0:
            # evaluate with resepct to all authors in the model
            wrt_authors = self._AuthorModel

        df = pd.DataFrame()
        for auth in tqdm(wrt_authors):
            md = self._AuthorModel[auth]
            HC = md.get_HC(xdtb)
            rank = md.get_rank(xdtb, LOO=LOO)
            chisq, chisq_pval = md.get_ChiSquare(xdtb)
            cosine = md.get_CosineSim(xdtb)
            df = df.append(
                {
                    'wrt_author': auth,
                    'HC': HC,
                    'chisq': chisq,
                    'chisq_pval' : -chisq_pval,
                    'HC_rank': rank,
                    'cosine': cosine,
                },
                ignore_index=True)
        return df

    def stats_list(self, data, wrt_authors=[], LOO=False):
        """
        Same as internal_stats but for a list of documents 

        Args:
        -----
            data    list of documents with columns: doc_id|author|text

        Returns:
        -------
            dataframe with rows: 
            doc_id|author|HC|ChiSq|cosine|rank|wrt_author

            doc_id      the document identifyer.
            wrt_author  author of the corpus against which the document 
                        is tested.
            HC, ChiSq, cosine    HC score, Chi-Square score, and cosine
                                 similarity, respectively, between the 
                                 document and the corpus.
            rank        the rank of the HC score compared to other documents 
                        within the corpus.
        """

        df = pd.DataFrame()

        if len(wrt_authors) == 0:
            # evaluate with resepct to all authors in the model
            wrt_authors = self._AuthorModel

        for auth0 in tqdm(wrt_authors):
            md0 = self._AuthorModel[auth0]
            for r in data.iterrows() :
                dtbl =  self._to_docTermTable([r[1].text])
                chisq, chisq_pval, chisq_rank = md0.get_ChiSquare(dtbl)
                cosine = md0.get_CosineSim(dtbl)
                HC = md0.get_HC(dtbl)
                rank = md0.get_rank(dtbl, LOO=LOO)
                F = md0.get_FisherComb(dtbl, LOO=LOO)
                df = df.append(
                    {
                        'doc_id': r[1].doc_id,
                        'author': r[1].author,
                        'wrt_author': auth0,
                        'HC': HC,
                        'Fisher' : F,
                        'chisq': chisq,
                        'chisq_rank' : chisq_rank,
                        'chisq_pval' : chisq_pval,
                        'cosine': cosine,
                        'HC_rank': rank,
                    },
                    ignore_index=True)
        return df

    def two_author_test(self, auth1, auth2, within=False, **kwargs) :
        return self._AuthorModel[auth1]\
                  .two_table_HC_test(self._AuthorModel[auth2],
                   within=within, **kwargs)

    def two_doc_test(self, auth_doc_pair1, auth_doc_pair2, **kwargs) :
        """ Test two documents/corpora against each other.

        Args:
        -----
        auth_doc_pairx  (tuple) first coordinate represents the corpus
                        name and second coorindate represents the document
                        name. If document name is None, entire corpus is used. 
                        For testing a corpus agains a dcoument of that 
                        corpus use:
                        auth_doc_pair1 = (<corpus_name>, None)
                        auth_doc_pair2 = (<corpus_name>, <doc_id>)
        """

        if auth_doc_pair1[1] == None :
            md1 = self._AuthorModel[auth_doc_pair1[0]]
        else :
            md1 = self._AuthorModel[auth_doc_pair1[0]]\
            .get_row_as_FreqTable(auth_doc_pair1[1])

        if auth_doc_pair2[1] == None :
            md2 = self._AuthorModel[auth_doc_pair2[0]]
        else :
            md2 = self._AuthorModel[auth_doc_pair2[0]]\
            .get_row_as_FreqTable(auth_doc_pair2[1])
        
        if auth_doc_pair1[0] == auth_doc_pair2[0] :
            if auth_doc_pair1[1] == None:
                logging.debug('Removing counts of document 2 from corpus 1.')
                return md1.two_table_HC_test(md2, within=True, **kwargs)            
            
        return md1.two_table_HC_test(md2, within=False, **kwargs)

    def reduce_features(self, new_feature_set):
        """
            Update the model to a new set of features. 
        """
        self._vocab = new_feature_set
        self._recompute_author_models()

    def get_data_labels(self, cls) :
        dtm = self._AuthorModel[cls]._dtm
        n, _ = dtm.shape
        return dtm, [cls] * n

    def check_classifier(self, classifier, split=0.75) :
        # get data and labels:
        y = []
        X = []
        for cls in self._AuthorModel :
            X1, y1 = self.get_data_labels(cls)
            X = scipy.sparse.vstack([X, X1])
            y += y1
        X = X.tocsr()[1:]

        X_train, X_test,y_train, y_test = train_test_split(
                                X, y, test_size=1-split)
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)


class CheckClassifier(object) :
    """
    Check performance of a classifier operating on an 
    AuthorshipAttributionMulti model
    =================================================

    Args:
    -----
    classifier     has methods fit and score
    model          AuthorshipAttributionMulti
    nMonte         number of checks
    split          train/test split 
    """
    
    def __init__(self, classifier, model, nMonte=1,
                 split=0.75) :
        
        self.classifier = classifier
        self.model = model
        
        acc = []
        X, y = self.get_data_labels()
        
        for i in tqdm(range(nMonte)) :
            X_train, X_test,y_train, y_test = train_test_split(
                                    X, y, test_size=1-split)
            self.fit(X_train, y_train)
            acc += [self.evaluate(X_test, y_test)]
            
        self.acc = acc
            
    def get_data_labels(self) :
        
        def dic_to_values(X) :
            return [list(x.values()) for x in X]

        def dtm_to_featureset(dtm) :
            fs = []
            for sm_id in dtm.get_row_labels() :
                dtl = dtm.get_row_as_FreqTable(sm_id)
                fs += [dtl.get_featureset()]
            return fs
        
        ds = []
        for auth in self.model._AuthorModel :
            mdd =  self.model._AuthorModel[auth]
            fs = dtm_to_featureset(mdd)
            ds += [(f, auth) for f in fs]
            
        ls = list(zip(*ds))
        X = dic_to_values(ls[0])
        y = ls[1]
        return X, y

    def fit(self, X, y) :
        self.classifier.fit(X, y)
    
    def evaluate(self, X, y) :
        return self.classifier.score(X, y)



class AuthorshipAttributionDTM(AuthorshipAttributionMulti) :
    """ 
    Same as AuthorshipAttributionMulti but input is a 
    pd.DataFrame of the form <author>, <doc_id>, <lemma>

    Overrides methods :
        compute_author_models 
        to_docTermTable
    
    """
    
    def _get_vocab(self, ds, **kwargs) :
        """
        Create shared vocabulary from data

        Current version takes all words with at least MIN_CNT 
        appearances
        """

        MIN_CNT = kwargs.get('min_cnt', 3)
        cnt = ds.term.value_counts() 
        vocab = cnt[cnt >= MIN_CNT].index.tolist()
        self._vocab = vocab

    def _compute_author_models(self, ds, **kwargs) :
        
        lo_authors = pd.unique(ds.author)
        for auth in lo_authors:
            ds_auth = ds[ds.author == auth]
            logging.info("Creating author-model for {}...".format(auth))
            
            dtm = self._to_docTermTable(ds_auth, **kwargs)
            dtm.change_vocabulary(new_vocabulary=self._vocab)
            self._AuthorModel[auth] = dtm
            
            logging.info("Found {} documents and {} relevant tokens."\
                .format(len(self._AuthorModel[auth].get_row_labels()),
                    self._AuthorModel[auth]._counts.sum()))
    

    def _to_docTermTable(self, df, **kwargs):
        """Convert raw input X into a FreqTable object. 

        Override this fucntion to process other input format. 

        Args: 
            X                 list of texts 
            document_names    list of strings representing the names
                              of each text in X

        Returs:
            FreqTable object
        """
        
        def df_to_FreqTable(df) :
            df = pd.DataFrame(df.groupby(['doc_id']).\
                term.value_counts()).\
                rename(columns={'term' : 'n'}).\
                reset_index().\
                pivot_table(index = 'doc_id', columns='term',
                 values='n', fill_value=0)
            feature_nams = df.columns.tolist()
            document_names = df.index.tolist()
            mat = df.to_numpy()
            return mat, document_names, feature_nams

        df = df[df.term.isin(self._vocab)]
        mat, dn, fn = df_to_FreqTable(df)
        dtm = FreqTable(mat,
                    column_labels=fn,
                    row_labels=dn,
                    stbl=kwargs.get('stbl', True),
                    randomize=kwargs.get('randomize', False),
                    gamma=kwargs.get('gamma', 0.25),
                    pval_thresh=kwargs.get('pval_thresh',1.1),
                    min_cnt=kwargs.get('min_cnt', 3),
                    pval_type=kwargs.get('pval_type', 'cell')
                    )
        return dtm




class AuthorshipAttributionMultiBinary(object):
    """ 
        A class to measure document and author similarity based on pair-wise 
        testing of authors. It creates an AuthorshipAttributionMulti object 
        for each pair of authors.

        The interface is similar to AuthorshipAttributionMultiBinary
        except that prediction can be made by majority voting.
        =================================================================

    """
    def __init__(self, data, vocab=[], vocab_size=100,
            words_to_ignore=[], global_vocab=False,
            ngram_range=(1, 1), stbl=True, reduce_features=False,
            randomize=False,
                ):
        # train_data is a dataframe with at least fields: author|doc_id|text
        # vocab_size is an integer controlling the size of vocabulary

        self._AuthorPairModel = {}
        self._stbl = stbl
        self._randomize = randomize

        if (len(vocab) == 0) and global_vocab :
            #get top vocab_size terms
            vocab = n_most_frequent_words(
                            list(data.text),
                            n=vocab_size,
                            words_to_ignore=words_to_ignore,
                            ngram_range=ngram_range)
            
        lo_authors = pd.unique(data.author)  #all authors
        lo_author_pairs = [(auth1, auth2) for auth1 in lo_authors\
                             for auth2 in lo_authors if auth1 < auth2 ]

        print("Found {} author-pairs".format(len(lo_author_pairs)))
        for ap in lo_author_pairs:  # AuthorPair model for each pair
            print("MultiBinaryAuthorModel: Creating model for {} vs {}..."\
                .format(ap[0],ap[1]), end =" ")

            data_pair = data[data.author.isin(list(ap))]
            ap_model = AuthorshipAttributionMulti(
                                    data_pair,
                                    vocab=vocab,
                                    vocab_size=vocab_size,
                                    words_to_ignore=words_to_ignore,
                                    ngram_range=ngram_range,
                                    stbl=stbl,
                                    randomize=self._randomize
                                    )
            print("Done.")

            self._AuthorPairModel[ap] = ap_model
            if reduce_features == True:
                feat = self.reduce_features_for_author_pair(ap)
                print("\t\tReduced to {} features...".format(len(feat)))
                
    def reduce_features_for_author_pair(self, auth_pair) :
        """
            Find list of features (tokens) discriminating two authors
            and reduce model to those features. 
            'auth_pair' is a key in self._AuthorPairModel
            returns the new list of features
        """
        ap_model = self._AuthorPairModel[auth_pair]

        md1 = ap_model._AuthorModel[auth_pair[0]]
        md2 = ap_model._AuthorModel[auth_pair[1]]
        
        _, _, feat = md1.get_HC_rank_features(md2)
        ap_model.reduce_features(list(feat))
        return ap_model._vocab

    def predict(self, x, method='HC', LOO=False):
        def predict_max(df1):
            # whoever has more votes or <UNK> in the 
            # case of a draw
            cnt = df1.pred.value_counts()
            imx = cnt.values == cnt.values.max()
            if sum(imx) == 1: 
                return cnt.index[imx][0] 
            else: #in the case of a draw
                return '<UNK>'

        df1 = self.predict_stats(x, LOO=LOO, method=method)

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

    def two_author_test(self, auth1, auth2, **kwargs) :
        md = self._AuthorPairModel[(auth1, auth2)]
        return md.two_author_test(auth1, auth2, **kwargs)


