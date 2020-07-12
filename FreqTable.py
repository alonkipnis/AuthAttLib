import numpy as np
import pandas as pd
import scipy
from scipy.sparse import vstack, coo_matrix
from sklearn.neighbors import NearestNeighbors
    
import sys
#sys.path.append('./TwoSampleHC')
from .TwoSampleHC import HC, binom_test_two_sided,\
         two_sample_pvals, two_sample_test_df, binom_var_test
from .goodness_of_fit_tests import *
    
#To do :
# complete class MultiTable


class FreqTable(object):
    """ 
    A class to represent contingency table of associated with multiple datasets
    Interface for checking the similarity of the table to other tables
    using Higher Criticism (HC) and other statistics. Designed to 
    accelerate computation of HC
    ==========================================================================

    Parameters:
    ---------- 
    dtm             feature-count matrix.
    column_labels   list of names for each column of dtm (feature name)
    row_labels      list of names for each row of dtm (e.g., document ID)
    stbl            Indiacate type of HC statistic to use 
    randomize       indicate whether to randomized P-values or not 
    gamma           HC lower P-value fraction limit
    min_cnt         ignore features whose total count is below this number

    """

    def __init__(self, dtm, column_labels=[], row_labels=[],
        min_cnt=0, stbl=True, gamma=0.25, randomize=False,
         pval_thresh=1.1, pval_type='both') :
        
        if len(row_labels) < dtm.shape[0] :
            row_labels = ["smp" + str(i) for i in range(dtm.shape[0])]
        self._row_labels = dict([
            (s, i) for i, s, in enumerate(row_labels[:dtm.shape[0]])
        ])
        self._sparse = scipy.sparse.issparse(dtm) # check if matrix is sparse

        if len(column_labels) < dtm.shape[1] :
            column_labels = np.arange(1,dtm.shape[1]+1).astype(str).tolist()
        self._column_labels = column_labels  #: feature name (list)
        if not self._sparse :
            self._dtm = np.asarray(dtm)  #: doc-term-table (matrix)
        else :
            self._dtm = dtm
        self._stbl = stbl  #: type of HC score to use 
        self._randomize = randomize #: randomize P-values or not
        self._gamma = gamma # gamma parameter for HC statistic
        self._min_cnt = min_cnt # ignore features whose total count is below 
                                # this number when getting p-vals from counts
        self._pval_thresh = pval_thresh #only consider P-values smaller than
        #import pdb; pdb.set_trace()
        self._pval_type = pval_type 

        if dtm.sum() == 0:
            raise ValueError(
                "Seems like all counts are zero. "\
                +"Did you pass the wrong data format?"
            )
        
        self.__compute_internal_stat()


    @staticmethod     
    def two_sample_pvals_loc(c1, c2, randomize=False,
                         min_cnt=0, pval_type='both') :
        if pval_type == 'variance' :
            return binom_var_test(c1, c2).values
        if pval_type == 'exact' :
            return two_sample_pvals(c1, c2, randomize=randomize)

        pv_bin_var = binom_var_test(c1, c2).values
        pv_exact = two_sample_pvals(c1, c2, randomize=randomize)
        pv_exact = pv_exact[c1 + c2 >= min_cnt]

        pv_all = np.concatenate([pv_bin_var, pv_exact])
        return pv_all

    @staticmethod
    def get_mat_sum(mat) :
        """
        mat can be 2D numpy array or a scipy matrix
        """
        if scipy.sparse.issparse(mat) :
            return np.squeeze(np.array(mat.sum(0))).astype(int)
        else :
            return np.squeeze(mat.sum(0))

    @staticmethod
    def get_mat_row(mat, r) :
        """
        mat can be 2D numpy array or a scipy matrix
        """
        if scipy.sparse.issparse(mat) :
            return np.squeeze(mat[r,:].toarray()).astype(int)
        else :
            return mat[r,:]

    def row_similarity(self, c1, c2) :
        hc = HC_sim(c1, c2, gamma=self._gamma, 
                randomize=self._randomize,
                pval_thresh=self._pval_thresh)
        return hc

    def __compute_internal_stat(self, compute_pvals=True):
        """ summarize internal doc-term-table """
        
        self._terms_per_doc = np.asarray(self._dtm.sum(1).ravel())\
                            .squeeze().astype(int)
        self._counts = np.asarray(self._dtm.sum(0).ravel())\
                    .squeeze().astype(int)
        
        self._internal_scores = []

        
        self._internal_scores = self._per_row_similarity_LOO(
            self.row_similarity)
    

    def __compute_HC(self, pvals) :
        np.warnings.filterwarnings('ignore') # when more than one pval is 
        # np.nan numpy show a warning. The current code supress this warning
        pv = pvals[pvals < self._pval_thresh]
        pv = pv[~np.isnan(pv)]
        np.warnings.filterwarnings('always')
        if len(pv) > 0 :
            hc = HC(pv, stbl=self._stbl)
            return hc.HCstar(gamma=self._gamma)
        else :
            return np.nan, np.nan
        #return hc_vals(pv, stbl=self._stbl,
        #     gamma=self._gamma)

    def get_column_labels(self):
        "returns name of each column in table"
        return self._column_labels

    def get_featureset(self) :
        """
        Returns a dictionary with keys = column_labels 
        and values = total count per column """
        return dict(zip(self._column_labels,
            np.squeeze(np.array(self._counts))))

    def get_per_sample_featureset(self) :
        ls = []
        for smp_id in self._row_labels :
            if self._sparse :
                counts = np.squeeze(
        np.array(self._dtm[self._row_labels[smp_id], :].todense())
            ).tolist() 
            else :
                counts = np.squeeze(
               np.array(self._dtm[self._row_labels[smp_id], :])
                    ).tolist() 
            #get counts from a single line
            ls += [dict(zip(self._column_labels,counts))]
        return ls

    def get_row_labels(self):
        "returns id of each row in table"
        return self._row_labels

    def _Pvals_from_counts(self, counts, within=False):
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
            pv = FreqTable.two_sample_pvals_loc(cnt1, cnt2,
                     randomize=self._randomize, min_cnt=self._min_cnt,
                     pval_type=self._pval_type
                     )
        else:
            pv = FreqTable.two_sample_pvals_loc(cnt1, cnt0,
                 randomize=self._randomize, min_cnt=self._min_cnt,
                 pval_type=self._pval_type)
        return pv 

    def __get_counts(self, dtbl, within=False) :
        """ Returns two list of counts, one from an 
        external table and one from class instance while considering
         'within' parameter to reduce counts from class instance

        Args: 
        -----
            dtbl -- FreqTable representing another frequency 
                    counts table
            within -- indicates whether counts of dtbl should be 
                    reduced from from counts of self._dtm

        Returns:
        -------
            cnt0 -- adjusted counts of object instance
            cnt1 -- adjusted counts of dtbl
        """

        if list(dtbl._column_labels) != list(self._column_labels):
            print(
            "Features of 'dtbl' do not match FreqTable"
            "instance. Changing dtbl accordingly."
            )
            #Warning for changing the test object
            dtbl.change_vocabulary(self._column_labels)

        cnt0 = self._counts
        cnt1 = dtbl._counts
        if within:
            cnt0 = cnt0 - cnt1
            if np.any(cnt0 < 0):
                raise ValueError("'within == True' is invalid")
        return cnt0, cnt1

    def get_Pvals(self, dtbl, within=False):
        """ return a list of binomial allocation 
            p-values of another FreqTable 'dtbl' with 
            respect to doc-term table of class instance.

        Args: 
            dtbl -- FreqTable object 
        """
        cnt0, cnt1 = self.__get_counts(dtbl, within=within)
        pv = FreqTable.two_sample_pvals_loc(cnt1, cnt0,
                 randomize=self._randomize, min_cnt=self._min_cnt,
                 pval_type=self._pval_type
                        )
        return pv

    def two_table_HC_test(self, dtbl, **kwargs) :
        """
        counts, p-values, and HC with 
        respect to another FreqTable

        Args:
        -----
        dtbl : another FreqTable to test agains


        Returns:
        -------
        DataFrame with columns representing counts, 
        binomial allocation P-values,
        binom_var_p-values, 
        and HC score

        """
        stbl = kwargs.get('stbl', self._stbl)
        randomize = kwargs.get('randomize', self._stbl)
        gamma = kwargs.get('gamma', self._gamma)
        within = kwargs.get('within', False)
        min_cnt = kwargs.get('min_cnt', self._min_cnt)
        pvals_type = kwargs.get('pvals', 'binomial allocation')

        cnt0, cnt1 = self.__get_counts(dtbl, within=within)
        
        if pvals_type == 'binomial variance' :
            df = pd.DataFrame(binom_var_test(cnt0, cnt1) )

        else :
            df = two_sample_test_df(cnt0, cnt1,
                 stbl=stbl,
                randomize=randomize,
                gamma=gamma,
                min_cnt=min_cnt
                )
            lbls = self._column_labels
            try :
                df.loc[:,'feature'] = lbls
            except :
                df.loc[:,'feature'] = [lbls]
        return df

    def change_vocabulary(self, new_vocabulary):
        """ Shift and remove columns of self._dtm so that it 
        represents counts with respect to new_vocabulary
        """
        if self._sparse :
            new_dtm = scipy.sparse.lil_matrix(
            np.zeros((self._dtm.shape[0], len(new_vocabulary)))
                                            )
        else : 
            new_dtm = np.zeros((self._dtm.shape[0],
                                 len(new_vocabulary)))
        old_vocab = self._column_labels

        no_missing_words = 0
        for i, w in enumerate(new_vocabulary):
            
            try:
                new_dtm[:, i] = self._dtm[:, old_vocab.index(w)]
                                
            except:  # num of words in new vocabulary that 
                     # do not exists in old one
                no_missing_words += 1

        self._dtm = new_dtm
        self._column_labels = new_vocabulary

        self.__compute_internal_stat()

    def __dtm_plus_row(self, row) :
        """
        Args: 
        -----
        row : matrix of size (1, no_columns)

        Returns: 
        -------
        copy of the object matrix plus another row
        """

        if len(np.shape(row)) < 2 :
            row = np.atleast_2d(row)

        assert(row.shape[1] == self._dtm.shape[1])

        if self._sparse :
            dtm_all = vstack([row, self._dtm]).tolil()
        else :     
            dtm_all = np.concatenate([np.array(row), self._dtm], axis = 0)
        return dtm_all

    def _per_row_similarity_LOO(self, sim_measure, new_row = [],
                                                 within=False) :
        """
        Similarity of each row against all others. 

        Args:
        -------
        new_row : is a (optional) new row (array of size (1,# of columns))
        sim_measure(c1 : [int], c2 : [int]) -> float
        within : indicates weather 'new_row' is already a 
                 row in the table
        """

        lo_scores = []

        if (within == False) and (len(new_row) > 0) :
            mat = self.__dtm_plus_row(new_row)
        elif len(new_row) > 0 :
            mat = self._dtm
            # similarity of new_row
            
            cnt0 = np.squeeze(new_row)
            cnt1 = FreqTable.get_mat_sum(mat) - cnt0
            if np.any(cnt1 < 0):
                raise ValueError("'within == True' is invalid")

            lo_scores += [sim_measure(cnt0, cnt1)]
        else :
            mat = self._dtm

        r, _ = mat.shape

        cnt_total = FreqTable.get_mat_sum(mat) 
        
        for i in range(r) :
            cnt0 = FreqTable.get_mat_row(mat, i)
            cnt1 = cnt_total - cnt0
                
            #import pdb; pdb.set_trace()
            lo_scores += [sim_measure(cnt0, cnt1)]

        return lo_scores
    

    def __per_smp_Pvals_LOO(self, row) :
        pv_list = []
        
        mat = self.__dtm_plus_row(row)
        
        def func(c1, c2) :
            return FreqTable.two_sample_pvals_loc(c1, c2, 
                            randomize=self._randomize,
                            min_cnt=self._min_cnt,
                            pval_type=self._pval_type
                            )

        r,c = mat.shape
        pv_list = []

        for i in range(r) :
            if self._sparse :
                cnt0 = np.squeeze(mat[i,:].toarray())
            else :
                cnt0 = np.squeeze(mat[i,:])
            cnt1 = np.squeeze(np.asarray(mat.sum(0))) - cnt0
            pv_list += [func(cnt0, cnt1)]

        return pv_list

    def get_row_as_FreqTable(self, smp_id : str) :
        """ Returns a single row in the doc-term-matrix as a new 
        FreqTable object. 

        Args:
            smp_id -- row identifier.

        Returns:
            FreqTable object
        """
        if self._sparse :
            dtm = self._dtm[self._row_labels[smp_id], :]
        else :
            dtm = np.atleast_2d(self._dtm[self._row_labels[smp_id], :])

        new_table = FreqTable(dtm,
                            column_labels=self._column_labels,
                            row_labels=[smp_id], gamma=self._gamma,
                            randomize=self._randomize, stbl=self._stbl)
        return new_table


    def copy(self) : 
        # create a copy of FreqTable instance
        new_table = FreqTable(
                     self._dtm,
                     column_labels=self._column_labels,
                     row_labels=list(self._row_labels.keys()), 
                     gamma=self._gamma, randomize=self._randomize,
                     stbl=self._stbl)
        return new_table

    def add_tables(self, lo_dtbl) :
        """ 
        Returns a new FreqTable object after adding
        a second FreqTable to the current one. 

        Parameters:
        -----------
        dtbl : Another FreqTable.

        Returns :
        -------
        FreqTable : current instance (self)
        """
        
        warnings.warn("FreqTable::add_table is deprecated",
                     DeprecationWarning, stacklevel=2)

        curr_feat = self._column_labels

        for dtbl in lo_dtbl :
            
            feat1 = dtbl._column_labels

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
            self._row_labels.update(dtbl._row_labels)

        self.__compute_internal_stat() 
        return self


    def get_ChiSquare(self, dtbl, within=False,
        lambda_ = None, LOO_rank=False):
        """ ChiSquare score with respect to another FreqTable 
        object 'dtbl'

        Returns:
        ------- 
        Chi-squares test score
        pvalue of this test
        rank of test scores among other documents

        """
        cnt0, cnt1 = self.__get_counts(dtbl, within=within)
        score, pval = two_sample_chi_square(cnt0, cnt1, lambda_ = lambda_)

        def sim_measure(c1, c2) : 
            return two_sample_chi_square(c1, c2, lambda_ = lambda_)[0]

        rank = np.nan
        if LOO_rank == True :
            rank = self.get_rank(dtbl, sim_measure=sim_measure,
                within=within, LOO=True)

        return score, pval, rank
        
    def get_BJSim(self, dtbl, within=False):
        """ Berk-Jones similarity with respect to another FreqTable 
        object 'dtbl'
        """
        cnt0, cnt1 = self.__get_counts(dtbl, within=within)

        return BJ_sim(cnt0, cnt1)


    def get_CosineSim(self, dtbl, within=False):
        """ Cosine similarity with respect to another FreqTable 
        object 'dtbl'
        """
        cnt0, cnt1 = self.__get_counts(dtbl, within=within)

        return cosine_sim(cnt0, cnt1)

    def get_HC(self, dtbl, within=False):
        """ returns the HC score of dtm1 wrt to doc-term table,
        as well as its rank among internal scores 
        Args:
            stbl -- indicates type of HC statistic
            within -- indicate whether tested table is included in current 
                    FreqTable object. if within==True then tested _count
                    are subtracted from FreqTable._dtm
         """
        cnt0, cnt1 = self.__get_counts(dtbl, within=within)
        pvals = FreqTable.two_sample_pvals_loc(cnt0, cnt1, 
            randomize=self._randomize, min_cnt=self._min_cnt,
            pval_type=self._pval_type)
        #pvals = self.get_Pvals(dtbl, within=within)
        HC, p_thr = self.__compute_HC(pvals)

        return HC

    
    def get_rank(self, dtbl, sim_measure=None, within=False, LOO=True) :
        """ returns the rank of the similarity of dtbl compared to each
            row in the data-table. 
        Args:
            dtbl : another FreqTable 
            LOO : Leave One Out evaluation of the rank (much slower process
                    but more accurate; especially when number of documents
                    is small)
            within -- indicate whether tested table is included in current 
                    FreqTable object. if within==True then tested _count
                    are subtracted from FreqTable._dtm
        Return :
            rank of score among internal ranks

        Todo: 
            provide the option to use similarity measures other than HC
         """
        if sim_measure == None :
            sim_measure = self.row_similarity
        else : 
            LOO = True # because internal scores are only meaningful
                       # under the default similarity measure

        if LOO == False : # rank in stored HC scores
            lo_scores = self._internal_scores
            cnt0, cnt1 = self.__get_counts(dtbl, within=within)
            score = sim_measure(cnt0, cnt1)
            lo_scores = [score] + lo_scores

        elif LOO == True :
            lo_scores = self._per_row_similarity_LOO(sim_measure,
                             dtbl._counts, within=within)

        if len(lo_scores) > 1:
            score = lo_scores[0]
            rank = np.mean(np.array(lo_scores[1:]) < score) 
        else:
            rank = np.nan
        
        return rank

    def get_HC_rank_features(self,
        dtbl,           
        LOO=False,             
        within=False):
        """ returns the HC score of dtm1 wrt to doc-term table,
        as well as its rank among internal scores 
        Args:
            LOO : Leave One Out evaluation of the rank (much slower process
                    but more accurate; especially when number of documents
                    is small)
            within : indicate whether tested table is included in current 
                    FreqTable object. if within==True then tested _count
                    are subtracted from FreqTable._dtm
         """
        
        pvals = self.get_Pvals(dtbl, within=within)

        HC, p_thr = self.__compute_HC(pvals)

        pvals[np.isnan(pvals)] = 1
        feat = np.array(self._column_labels)[pvals < p_thr]

        if (LOO == False) or (within == True):
            # internal pvals are always evaluated in a LOO manner,
            # hence we used internal HC scores in these cases

            lo_hc = self._internal_scores
            if len(lo_hc)- within > 0: # at least 1 doc not including
                                       # tested doc
                s = np.sum(np.array(lo_hc) < HC) 
                rank = s / (len(lo_hc) - within)
            else:
                rank = np.nan
            
        elif LOO == True :
            loo_Pvals = self.__per_smp_Pvals_LOO(dtbl._dtm)[1:]
              #remove first item (corresponding to tested table)

            lo_hc = []
            if (len(loo_Pvals)) == 0:
                raise ValueError("list of LOO Pvals is empty")

            for pv in loo_Pvals:
                hc, _ = self.__compute_HC(pv)
                lo_hc += [hc]

            if len(lo_hc) > 0:
                rank = np.mean(np.array(lo_hc) < HC) 
            else:
                rank = np.nan

        return HC, rank, feat



class FreqTableClassifier(NearestNeighbors) :
    """ nearset neighbor classifcation for frequency tables 
        TODO: 
         - implement SVD or LDA classifier based on one of the 
           metrics

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
        for x, cls_name in zip(X, y) :
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
            r = x1.two_table_HC_test(x2)
            return r['HC'].values[0]

        def chisq(x1, x2) :
            return x1.get_ChiSquare(x2)[0]

        def chisq_pval(x1, x2) :
            return x1.get_ChiSquare(x2)[1]

        def cosine(x1, x2) :
            return x1.get_CosineSim(x2)

        def LogLikelihood(x1, x2) :
            return x1.get_ChiSquare(x2, lambda_ = "log-likelihood")[0]

        def FreemanTukey(x1, x2) :
            return x1.get_ChiSquare(x2, lambda_ = "freeman-tukey")[0]

        def modLogLikelihood(x1, x2) :
            return x1.get_ChiSquare(x2, lambda_ = "mod-log-likelihood")[0]

        def Neyman(x1, x2) :
            return x1.get_ChiSquare(x2, lambda_ = "neyman")[0]

        def CressieRead(x1, x2) :
            return x1.get_ChiSquare(x2, lambda_ = "cressie-read")[0]

        metric = self._metric

        lo_metrics = {'chisq' : chisq,
                      'cosine' : cosine,
                      'chisq_pval' : chisq_pval,
                      'HC' : sim_HC,
                      'log-likelihood' : LogLikelihood,
                      "freeman-tukey" : FreemanTukey,
                      "mod-log-likelihood" : modLogLikelihood,
                      "neyman" : Neyman,
                      "cressie-read" : CressieRead
                     }
        
        sim_measure = lo_metrics[metric]
        
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
    
