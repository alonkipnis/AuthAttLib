import pandas as pd
import numpy as np
from tqdm import *

from utils import to_docTermCounts, n_most_frequent_words

from DocTermTable import DocTermTable

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
                 stbl=True,
                 flat=False
                 ):
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
        self._flat = flat

        if len(self._vocab) == 0:  #common vocabulary
            vocab = n_most_frequent_words(list(data.text),
                                          n=vocab_size,
                                          words_to_ignore=words_to_ignore,
                                          ngram_range=self._ngram_range)
        self._vocab = vocab

        #compute author-models
        lo_authors = pd.unique(data.author)
        for auth in lo_authors:
            data_auth = data[data.author == auth]
            print("\t Creating author-model for {} using {} features..."\
                .format(auth, len(self._vocab)))

            self._AuthorModel[auth] = self.to_docTermTable(
                                list(data_auth.text),
                                document_names=list(data_auth.doc_id)
                                )
            print("\t\tfound {} documents and {} relevant tokens."\
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
        if self._flat == True:
            dtm = dtm.sum(0)
            document_names = ["Sum of {} docs".format(len(document_names))]

        return DocTermTable(dtm,
                            feature_names=self._vocab,
                            document_names=document_names,
                            stbl=self._stbl)

    def re_compute_author_models(self):
        """ compute author models after a change in vocab """

        for auth in self._AuthorModel:
            am = self._AuthorModel[auth]
            am.change_vocabulary(self._vocab)
            print("Changing vocabulary for {}. Found {} relevant tokens."\
                .format(auth, am._counts.sum()))

    def predict(self, x, method='HC', features_to_mask = [],
                 unk_thresh=1e6, LOO=False):
        """
        Attribute text x with one of the authors or '<UNK>'. 

        Args:
            x -- string representing the test document 
            method -- designate which score to use
            unk_thresh -- minimal score below which the text is 
            attributed to one of the authors in the model and not assigned
            the label '<UNK>'.
            LOO -- indicates whether to compute rank in a leave-of-out mode
            It leads to more accurate rank-based testing but require more 
            computations. 
            features_to_mask -- mask these features from HC test. 

        Returns:
            pred -- one of the keys in self._AuthorModel or '<UNK>'
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

        Xdtb = self.to_docTermTable([x])

        for i, auth in enumerate(self._AuthorModel):
            am = self._AuthorModel[auth]
            score, rank, feat = am.get_HC_rank_features(Xdtb, 
                                    features_to_mask = features_to_mask,
                                    LOO=LOO)
            chisq, chisq_pval = am.get_ChiSquare(Xdtb)
            cosine = am.get_CosineSim(Xdtb)

            if method == 'rank':
                score = rank
            elif method == 'cosine':
                score = cosine
            elif method == 'chisq' :
                score = chisq
            elif method == 'chisq_pval' :
                score = 1 - chisq_pval

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

        for auth0 in tqdm(self._AuthorModel): # go over all authors
            md0 = self._AuthorModel[auth0]
            for auth1 in self._AuthorModel:  # go over all corpora
                if auth0 < auth1:       # test each pair only once
                    md1 = self._AuthorModel[auth1]
                    HC, rank, feat = md0.get_HC_rank_features(md1)
                    chisq, chisq_pval = md0.get_ChiSquare(md1)
                    cosine = md0.get_CosineSim(md1)
                    df = df.append(
                        {
                            'author': auth1,
                            'wrt_author': auth0,
                            'HC': HC,
                            'chisq': chisq,
                            'chisq_pval' : chisq_pval,
                            'cosine': cosine,
                            'no_docs (author)': len(md1.get_document_names()),
                            'no_docs (wrt_author)': len(md0.get_document_names()),
                            'no_tokens (author)': md1._counts.sum(),
                            'feat': list(feat)
                        },
                        ignore_index=True)
        return df

    def internal_stats(self, wrt_authors=[], LOO=False):
        """
        Compute scores of each document with respect to the corpus of
        each author. When tested against its own corpus, the document
        is removed from that corpus. 
        
        Args:
        wrt_authors -- subset of the authors in the model with respect
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

        if len(wrt_authors) == 0:
            # evaluate with resepct to all authors in the model
            wrt_authors = self._AuthorModel

        for auth0 in tqdm(wrt_authors):
            md0 = self._AuthorModel[auth0]
            for auth1 in self._AuthorModel:
                md1 = self._AuthorModel[auth1]
                lo_docs = md1.get_document_names()
                for dn in lo_docs:
                    dtbl = md1.get_doc_as_table(dn)
                    if auth0 == auth1:
                        HC, rank, feat = md0.get_HC_rank_features(
                            dtbl,LOO=LOO,within=True
                            )
                        chisq, chisq_pval = md0.get_ChiSquare(dtbl,
                                                         within=True)
                        cosine = md0.get_CosineSim(dtbl, within=True)
                    else:
                        HC, rank, feat = md0.get_HC_rank_features(
                            dtbl, LOO=LOO)
                        chisq, chisq_pval = md0.get_ChiSquare(dtbl)
                        cosine = md0.get_CosineSim(dtbl)
                    df = df.append(
                        {
                            'doc_id': dn,
                            'author': auth1,
                            'wrt_author': auth0,
                            'HC': HC,
                            'chisq': chisq,
                            'chisq_pval' : chisq_pval,
                            'cosine': cosine,
                            'rank': rank,
                            'feat': list(feat)
                        },
                        ignore_index=True)
        return df

    def predict_stats(self, x, wrt_authors=[], LOO=False):
        """ Returns a pandas dataframe with columns representing the 
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
        xdtb = self.to_docTermTable([x])

        if len(wrt_authors) == 0:
            # evaluate with resepct to all authors in the model
            wrt_authors = self._AuthorModel

        df = pd.DataFrame()
        for auth in tqdm(wrt_authors):
            md = self._AuthorModel[auth]
            HC, rank, feat = md.get_HC_rank_features(xdtb, LOO=LOO)
            chisq, chisq_pval = md.get_ChiSquare(xdtb)
            cosine = md.get_CosineSim(xdtb)
            df = df.append(
                {
                    'wrt_author': auth,
                    'HC': HC,
                    'chisq': chisq,
                    'chisq_pval' : chisq_pval,
                    'rank': rank,
                    'feat': feat,
                    'cosine': cosine,
                },
                ignore_index=True)
        return df

    def stats_list(self, data, wrt_authors=[], LOO=False):
        """
        Same as internal_stats but for a list of documents 

        Arguments:
            data -- list of documents with columns: doc_id|author|text

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

        df = pd.DataFrame()

        if len(wrt_authors) == 0:
            # evaluate with resepct to all authors in the model
            wrt_authors = self._AuthorModel

        for auth0 in tqdm(wrt_authors):
            md0 = self._AuthorModel[auth0]
            for r in data.iterrows() :
                dtbl =  self.to_docTermTable([r[1].text])
                chisq, chisq_pval = md0.get_ChiSquare(dtbl)
                cosine = md0.get_CosineSim(dtbl)
                HC, rank, feat = md0.get_HC_rank_features(dtbl,
                                                        LOO=LOO)
                df = df.append(
                    {
                        'doc_id': r[1].doc_id,
                        'author': r[1].author,
                        'wrt_author': auth0,
                        'HC': HC,
                        'chisq': chisq,
                        'chisq_pval' : chisq_pval,
                        'cosine': cosine,
                        'rank': rank,
                        'feat': list(feat)
                    },
                    ignore_index=True)
        return df

    def reduce_features(self, new_feature_set):
        """
            Update the model to a new set of features. 
        """
        self._vocab = new_feature_set
        self.re_compute_author_models()

    def reduce_features_from_author_pair(self, auth1, auth2, stbl=None):
        """
            Find list of features (tokens) discriminating auth1 and auth2
            Reduce model to those features. 
            'auth1' and 'auth2' are keys within _AuthorModel
            return new list of features
        """
        md1 = self._AuthorModel[auth1]
        md2 = self._AuthorModel[auth2]
        _, _, feat = md1.get_HC_rank_features(md2, stbl=stbl)
        print("Reducing to {} features...".format(len(feat)))
        self.reduce_features(list(feat))
        return self._vocab

    def get_discriminating_features(self, x,
                             wrt_authors = [], stbl=None) :
        """ 
        Find list of features discriminating x and all
        authors in a list.
        
        Args:
            x -- input text (list of strings)
            wrt_authors -- subset of the authors in the model
                with respect to which to compute HC and features.
                If empty, evaluate with respect to all authors.

        Returns:
            dictionary of scores and features 
        """

        if stbl == None :
            stbl = self._stbl

        xdtb = self.to_docTermTable([x])

        if len(wrt_authors) == 0:
            # evaluate with resepct to all authors in the model
            wrt_authors = self._AuthorModel

        #aggregate models
        agg_model = None
        for auth in tqdm(wrt_authors):
            md = self._AuthorModel[auth]
            agg_model = md.add_table(agg_model)
            
        HC, _, feat = agg_model.get_HC_rank_features(xdtb, stbl=stbl)
        chisq, chisq_pval = agg_model.get_ChiSquare(xdtb)
        cosine = agg_model.get_CosineSim(xdtb)
        
        return {'HC': HC, 'feat': feat,
             'chisq': chisq, 'chisq_pval' : chisq_pval,
            'cosine': cosine}

    def flatten(self) :
        """ Merge all documents to a single one
        """
        self._flat = True
        for auth in self._AuthorModel :
            self._AuthorModel[auth].collapse_dtm()


class AuthorshipAttributionMultiBinary(object):
    """ Use pair-wise tests to determine most likely author. 
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
        for ap in lo_author_pairs:  # AuthorPair model for each pair
            print("MultiBinaryAuthorModel: Creating model for {} vs {}..."\
                .format(ap[0],ap[1]))

            data_pair = data[data.author.isin(list(ap))]
            ap_model = AuthorshipAttributionMulti(
                data_pair,
                vocab=vocab,
                vocab_size=vocab_size,
                words_to_ignore=words_to_ignore,
                stbl=stbl)
            if reduce_features == True:
                ap_model.reduce_features_from_author_pair(ap[0], ap[1])
            self._AuthorPairModel[ap] = ap_model

    def predict(self, x, method='HC'):
        def predict_max(df1):
            "Whoever win most"
            cnt = df1.pred.value_counts()
            imx = cnt.values == cnt.values.max()
            if sum(imx) == 1:
                return cnt.index[imx][0]
            else: #in the case of a draw
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
