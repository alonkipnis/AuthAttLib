import numpy as np
import scipy
from scipy.sparse import vstack, coo_matrix
from goodness_of_fit_tests import *
from sklearn.neighbors.base import NeighborsBase

from HC_aux import hc_vals, two_counts_pvals, two_sample_test

class FreqTable(object):
    """ Interface for p-value, Higher Criticism (HC), and other tests
    of homogenity with with respect to a document-term matrix.
 
    Parameters:
    ---------- 
    dtm : doc-term matrix.
    feature_names : list of names for each column of dtm.
    sample_ids :  list of names for each row of dtm.
    stbl : boolean) -- type of HC statistic to use 
    randomize : boolean -- randomized P-values 
    alpha  : boolean 

    To Do:
        - rank of ChiSquare in LOO

    """
    def __init__(self, dtm, feature_names=[], sample_ids=[],
        stbl=True, alpha=0.2, randomize=False) :
        """ 
        Parameters
        ---------- 
        dtm : (sparse) doc-term matrix.
        feature_names : list of names for each column of dtm.
        sample_ids : list of names for each row of dtm.

        """

        if sample_ids == []:
            sample_ids = ["smp" + str(i) for i in range(dtm.shape[0])]
        self._smp_ids = dict([
            (s, i) for i, s, in enumerate(sample_ids[:dtm.shape[0]])
        ])
        self._sparse = scipy.sparse.issparse(dtm) # check if matrix is sparse
        if len(feature_names) < dtm.shape[1] :
            feature_names = np.arange(1,dtm.shape[1]+1).astype(str).tolist()
        self._feature_names = feature_names  #: feature name (list)
        self._dtm = dtm  #: doc-term-table (matrix)
        self._stbl = stbl  #: type of HC score to use 
        self._randomize = randomize #: randomize P-values or not
        self._alpha = alpha
        #self._alpha = alpha # alpha parameter for HC statistic
        self._pval_thresh = 1 #only consider P-values smaller than this

        if dtm.sum() == 0:
            raise ValueError(
                "Seems like all counts are zero. "\
                +"Did you pass the wrong data format?"
            )

        self.__compute_internal_stat()


    def __compute_internal_stat(self, compute_pvals=True):
        """summarize internal doc-term-table"""

        self._terms_per_doc = np.squeeze(np.array(
            self._dtm.sum(1))).astype(int)
        self._counts = np.squeeze(np.array(self._dtm.sum(0))).astype(int)

        #keep HC score of each row w.r.t. the rest
        #pv_list = self.__per_smp_Pvals()
        self._internal_scores = []
        for row in self._dtm:
            if self._sparse : 
                cnt = np.squeeze(np.array(row.todense())).astype(int)
            else :
                cnt = np.squeeze(np.array(row)).astype(int)
            pv = self._get_Pvals(cnt, within = True)
            hc, p_thr = self.__compute_HC(pv)
            self._internal_scores += [hc]

    def __per_smp_Pvals(self):
        """Pvals of each row in dtm with respect to the rest"""

        pv_list = []

        #pvals when removing one row at a time
        counts = self._counts
        if self._dtm.shape[0] == 1:  # internal score is undefined
            return []
        for r in self._dtm:
            if self._sparse :
                c = np.squeeze(np.array(r.todense()))
            else :
                c = np.squeeze(np.array(r))
            pv = two_counts_pvals(c, counts - c, 
                            randomize=self._randomize)
            pv_list += [pv]

        return pv_list

    def __compute_HC(self, pvals) :
        pv = pvals[pvals < self._pval_thresh]
        return hc_vals(pv, stbl=self._stbl,
             alpha=self._alpha)

    def get_feature_names(self):
        "returns name of each column in table"
        return self._feature_names

    def get_featureset(self) :
        return dict(zip(self._feature_names,
            np.squeeze(np.array(self._counts))))

    def get_per_sample_featureset(self) :
        ls = []
        for smp_id in self._smp_ids :
            if self._sparse :
                counts = np.squeeze(
        np.array(self._dtm[self._smp_ids[smp_id], :].todense())
            ).tolist() 
            else :
                counts = np.squeeze(
               np.array(self._dtm[self._smp_ids[smp_id], :])
                    ).tolist() 
            #get counts from a single line
            ls += [dict(zip(self._feature_names,counts))]
        return ls

    def get_sample_ids(self):
        "returns id of each row in table"
        return self._smp_ids

    def get_counts(self):
        "returns cound of entry of each "
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
            pv = two_counts_pvals(cnt1, cnt2,
                     randomize=self._randomize,
                     )
        else:
            pv = two_counts_pvals(cnt1, cnt0,
                 randomize=self._randomize,
                        )
        return pv 

    def _get_counts(self, dtbl, within=False) :
        """ Returns two list of counts, one from an 
        external table and one from 'self' while considering
         'within' parameter to reduce counts from 'self'.

        Args: 
            dtbl -- FreqTable representing another frequency 
                    counts table
            within -- indicates whether counts of dtbl should be 
                    reduced from from counts of self._dtm

        Returns:
            cnt0 -- adjusted counts of self
            cnt1 -- adjusted counts of dtbl
        """

        if list(dtbl._feature_names) != list(self._feature_names):
            print(
            "Features of 'dtbl' do not match current FreqTable"
            "intance. Changing dtbl accordingly."
            )
            #Warning for changing the test object
            dtbl.change_vocabulary(self._feature_names)

        cnt0 = self._counts
        cnt1 = dtbl._counts
        if within:
            cnt0 = cnt0 - cnt1
            if np.any(cnt0 < 0):
                raise ValueError("'within == True' is invalid")
        return cnt0, cnt1

    def get_Pvals(self, dtbl):
        """ return a list of p-values of another FreqTable with 
        respect to 'self' doc-term table.

        Args: 
            dtbl -- FreqTable object with respect to which to
                    compute pvals
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

    def two_table_test(self, dtbl,
                 within=False, stbl=None,
                 randomize=False) :
        """counts, p-values, and HC with 
        respect to another FreqTable
        """
        if stbl == None :
            stbl = self._stbl

        cnt0, cnt1 = self._get_counts(dtbl, within=within)
        df = two_sample_test(cnt0, cnt1,
             stbl=stbl,
            randomize=self._randomize,
            alpha=self._alpha)
        df.loc[:,'feat'] = self._feature_names
        return df


    def change_vocabulary(self, new_vocabulary):
        """ Shift and remove columns of self._dtm so that it 
        represents counts with respect to new_vocabulary
        """
        if self._sparse :
            new_dtm = scipy.sparse.lil_matrix(
            np.zeros((self._dtm.shape[0], len(new_vocabulary))))
        else : 
            new_dtm = np.zeros((self._dtm.shape[0],
                                 len(new_vocabulary)))
        old_vocab = self._feature_names


        no_missing_words = 0
        for i, w in enumerate(new_vocabulary):
            try:
                new_dtm[:, i] = self._dtm[:, old_vocab.index(w)]
            except:  # occurs if a word in the
                # new vocabulary does not exists in old one.
                no_missing_words += 1

        self._dtm = new_dtm
        self._feature_names = new_vocabulary

        self.__compute_internal_stat()

    def __per_smp_Pvals_LOO(self, dtm1):
        pv_list = []

        if self._sparse :
            dtm_all = vstack([dtm1, self._dtm]).tolil()
        else :     
            dtm_all = np.concatenate([dtm1, self._dtm], axis = 0)
        #current sample corresponds to the first row in dtm_all

        #pvals when the removing one document at a time
        s1 = np.squeeze(np.array(dtm1.sum(0)))
        s = self._counts + s1
        for r in dtm_all:
            if self._sparse :
                c = np.squeeze(np.array(r.todense()))  #no dense
            else :
                c = np.squeeze(np.array(r))
            pv = two_counts_pvals(c, s - c,
                         randomize=self._randomize,
                            )
            pv_list += [pv]

        return pv_list

    def get_sample_as_table(self, smp_id):
        """ Returns a single row in the doc-term-matrix as a new 
        FreqTable object. 

        Args:
            smp_id -- row identifier.

        Returns:
            FreqTable object
        """
        dtm = self._dtm[self._smp_ids[smp_id], :]
        new_table = FreqTable(dtm,
                            feature_names=self._feature_names,
                            sample_ids=[smp_id], alpha=self._alpha,
                            randomize=self._randomize, stbl=self._stbl)
        return new_table

    def collapse_dtm(self) :
        """ Reduce table to a single row 
        Returns
        -------
        new_table : new instance of FreqTable

        """

        self._dtm = self._dtm.sum(0)

    def copy(self) :
        new_table = FreqTable(
                     self._dtm,
                     feature_names=self._feature_names,
                     sample_ids=list(self._smp_ids.keys()), 
                     alpha=self._alpha, randomize=self._randomize,
                     stbl=self._stbl)
        return new_table

    def add_tables(self, lo_dtbl):
        """ Returns a new FreqTable object after adding
        a second FreqTable to the current one. 

        Parameters:
        -----------
        dtbl : Another FreqTable.

        Returns
        -------
        FreqTable : current instance (self)
        """
        
        curr_feat = self._feature_names

        for dtbl in lo_dtbl :
            
            feat1 = dtbl._feature_names

            if curr_feat != feat1 :
                dtbl = dtbl.change_vocabulary(feat)

            if self._sparse :
                try :
                    dtm_tall = vstack([self._dtm, dtbl._dtm]).tolil()
                except :
                    dtm_tall = vstack([self._dtm, 
                        coo_matrix(dtbl._dtm)]).tolil()
            else :
                dtm_tall = np.concatenate([self._dtm, dtbl._dtm], axis=0)

            self._dtm=dtm_tall
            self._smp_ids.update(dtbl._smp_ids)

        self.__compute_internal_stat() 
        return self


    def get_ChiSquare(self, dtbl, within=False, lambda_ = None):
        """ ChiSquare score with respect to another FreqTable 
        object 'dtbl'
        """
        cnt0, cnt1 = self._get_counts(dtbl, within=within)
        return two_sample_chi_square(cnt0, cnt1, lambda_ = lambda_)

    def get_CosineSim(self, dtbl, within=False):
        """ Cosine similarity with respect to another FreqTable 
        object 'dtbl'
        """
        cnt0, cnt1 = self._get_counts(dtbl, within=within)

        return cosine_sim(cnt0, cnt1)

    def get_HC_rank_features(self,
        dtbl,           #type: FreqTable
        LOO=False,             
        within=False,
                        ):
        """ returns the HC score of dtm1 wrt to doc-term table,
        as well as its rank among internal scores 
        Args:
            LOO -- Leave One Out evaluation of the rank (much slower process
                    but more accurate; especially when number of documents
                    is small)
            stbl -- indicates type of HC statistic
            within -- indicate whether tested table is included in current 
                    FreqTable object. if within==True then tested _count
                    are subtracted from FreqTable._dtm
         """
        
        pvals = self._get_Pvals(dtbl.get_counts(), within=within)

        HC, p_thr = self.__compute_HC(pvals)

        pvals[np.isnan(pvals)] = 1
        feat = np.array(self._feature_names)[pvals < p_thr]

        if (LOO == False) or (within == True):
            # internal pvals are always evaluated in a LOO manner
            lo_hc = self._internal_scores
            if len(lo_hc) > 0: # at least 1 doc
                s = np.sum(np.array(lo_hc) < HC) 
                #rank = s / (len(lo_hc) + 1 - within)
                rank = s / len(lo_hc)
            else:
                rank = np.nan
            
        elif LOO == True :
            loo_Pvals = self.__per_smp_Pvals_LOO(dtbl._dtm)[1:]
              #remove first item (corresponding to test sample)

            lo_hc = []
            if (len(loo_Pvals)) == 0:
                raise ValueError("list of LOO Pvals is empty")

            for pv in loo_Pvals:
                hc, _ = self.__compute_HC(pv)
                lo_hc += [hc]

            if len(lo_hc) > 0:
                s = np.sum(np.array(lo_hc) < HC) 
                #rank = s / (len(lo_hc) + 1 - within)
                rank = s / len(lo_hc)
            else:
                rank = np.nan

        return HC, rank, feat



class FreqTableClassifier(NeighborsBase) :
    """ nearset neighbor classifcation for frequency tables 

    """

    def __init__(self, metric='HC', **kwargs):
        """
        Parameters:
        -----------
        metric : string -- what similarity measure to use
        **kwargs : argument to FreqTable
        """
        
        self._inf = 1e6
        self._class_tables = {}
        self._sparse = False
        self._metric = metric
        self._kwargs = kwargs

    def fit(self, X, y) :                
        """ store data in a way convinient for similarity evaluation
        ----------
        X : array of FreqTable objects, shape (n_queries)
        y : array of shape [n_queries] 
            Class labels for each data sample.
        """
        
        self._sparse = scipy.sparse.issparse(X[0])
        
        temp_dt = {}
        for x, cls_name in zip(X,y) :
            if cls_name in temp_dt:
                temp_dt[cls_name] += [x]
            else :
                temp_dt[cls_name] = [x]
                
        for cls_name in temp_dt :
            mat = np.array(temp_dt[cls_name])
            self._class_tables[cls_name] = FreqTable(mat, **self._kwargs)
        
    
    def predict_prob(self, X) :
        """Predict the class labels for the provided data.
        Parameters
        ----------
        X : array of FreqTable objects, shape (n_queries), 

        Returns
        -------
        y : array of shape [n_queries] 
            Class labels for each data sample.
        """ 
        
        def sim_HC(x1, x2) :
            r = x1.two_table_test(x2)
            return r['HC'].values[0]

        def chisq(x1, x2) :
            return x1.get_ChiSquare(x2)[0]

        def chisq_pval(x1, x2) :
            return x1.get_ChiSquare(x2)[1]

        def cosine(x1, x2) :
            return x1.get_CosineSim(x2)

        metric = self._metric
        if metric == 'chisq' :
            sim_measure = chisq
        elif metric == 'chisq_pval' :
            sim_measure = chisq_pval
        elif metric == 'cosine' :
            sim_measure = cosine
        else :
            sim_measure = sim_HC
        
        y_pred = []
        y_score = []
        for x in X :
            dtbl = FreqTable(np.expand_dims(x,0))
            min_cls = None
            min_score = self._inf
            for cls in self._class_tables :
                curr_score = sim_measure(self._class_tables[cls], dtbl)
                if curr_score < min_score :
                    min_score = curr_score
                    min_cls = cls
            y_pred += [min_cls]
            y_score += [min_score]

        return y_pred, y_score

    def set_metric(self, metric, **kwargs) :
        self._metric = metric
        if kwargs != None :
            self._kwargs = kwargs
        #note: if changing kwargs may need to fit model again


    def predict(self, X) :
        """Predict the class labels for the provided data.
        Parameters
        ----------
        X : array of FreqTable objects, shape (n_queries), 

        Returns
        -------
        y : array of shape [n_queries] 
            Class labels for each data sample.
        """
        
        y, _ = self.predict_prob(X)
        return y

    def score(self, X, y) :
        y_hat = self.predict(X)
        return np.mean(y_hat == y)
    
