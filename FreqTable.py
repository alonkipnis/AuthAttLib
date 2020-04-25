import numpy as np
import scipy
from scipy.sparse import vstack, coo_matrix
from goodness_of_fit_tests import *
from sklearn.neighbors import NearestNeighbors
    
from TwoSampleHC import HC, binom_test_two_sided,\
         two_sample_pvals, two_sample_test_df
    
#To do :
# complete class MultiTable
# make _Pvals_from_counts to use __get_counts

def binom_var_test(smp1, smp2, max_cnt = 50) :
    """
    Args : 
    smp1, smp2 : numpy arrays or lists of integer of equal legth
    max_cnt : maximum diagonal value smp1 + smp2 to consider

    Returns:
    series with index = m and value = P-value

    Note: 
    Current implementation assumes equals sample sizes for smp1 and smp2
    """
    # Binomal varaince test.   Requires Pandas

    import pandas as pd
    
    df_smp = pd.DataFrame({'n1' : smp1, 'n2' : smp2})
    df_smp.loc[:,'N'] = df_smp.agg('sum', axis = 'columns')
    df_smp = df_smp[(df_smp.N <= max_cnt) & (df_smp.N > 0)]
    df_hist = df_smp.groupby(['n1', 'n2']).count().reset_index()

    df_hist.loc[:,'m'] = df_hist.n1 + df_hist.n2

    df_hist.loc[:,'N1'] = df_hist.n1 * df_hist.N
    df_hist.loc[:,'N2'] = df_hist.n2 * df_hist.N

    df_hist.loc[:,'NN1'] = df_hist.N1.sum()
    df_hist.loc[:,'NN2'] = df_hist.N2.sum()

    df_hist = df_hist.join(df_hist.filter(
        ['m', 'N1', 'N2', 'N']).groupby('m').agg('sum'),
                           on = 'm', rsuffix='_m')

    df_hist.loc[:,'p'] = (df_hist['NN1'] - df_hist['N1_m'])\
            / (df_hist['NN1'] + df_hist['NN2'] - df_hist['N1_m'] - df_hist['N2_m'])

    df_hist.loc[:,'s'] = \
            (df_hist.n1 - df_hist.m * df_hist.p) ** 2 * df_hist.N
    df_hist.loc[:,'Es'] = \
            df_hist.N_m * df_hist.m * df_hist.p * (1 - df_hist.p)
    df_hist.loc[:,'Vs'] =  2 * df_hist.N_m \
        * df_hist.m * (df_hist.m)*(df_hist.p * (1-df_hist.p)) ** 2
    df_hist = df_hist.join(df_hist.groupby('m').agg('sum').s,
                         on = 'm', rsuffix='_m')
    df_hist.loc[:,'z'] = (df_hist.s_m - df_hist.Es) / np.sqrt(df_hist.Vs)
    df_hist.loc[:,'pval'] = \
        df_hist.z.apply(lambda z : scipy.stats.norm.cdf(-np.abs(z)))

    # handle the case m=1 seperately
    n1 = df_hist[(df_hist.n1 == 1) & (df_hist.n2 == 0)].N.values
    n2 = df_hist[(df_hist.n1 == 0) & (df_hist.n2 == 1)].N.values
    if len(n1) + len(n2) >= 2 :
        df_hist.loc[df_hist.m == 1,'pval'] =\
                     binom_test_two_sided(n1, n1 + n2 , 1/2)

    return df_hist.groupby('m').pval.mean()
     
def two_sample_pvals_loc(c1, c2, randomize=False) :
    #pv_bin_var = binom_var_test(c1, c2).values
    pv_exact = two_sample_pvals(c1, c2)
    #pv_all = np.concatenate([pv_bin_var, pv_exact])
    return pv_exact

def get_row_sim_LOO(mat, sim_measure) :
    # compute similarity of each row in matrix mat
    # to the sum of all other rows
    r,c = mat.shape
    lo_scores = []

    for i in range(r) :
        cnt0 = mat[i,:]
        cnt1 = mat.sum(0) - cnt0
        lo_scores += [sim_measure(cnt0, cnt1)]

class FreqTable(object):
    """ 
    Represents 1-way contingency table of multiple dataset
    Each feature has a unique name 
    Each dataset has a unique name
    Allows to check similarity of the table to other tables
    using Higher Criticism (HC) and other statistics. Designed to 
    accelerate computation of HC

 
    Parameters:
    ---------- 
    dtm : doc-term matrix.
    column_names : list of names for each column of dtm.
    row_names :  list of names for each row of dtm.
    stbl : boolean) -- type of HC statistic to use 
    randomize : boolean -- randomized P-values 
    alpha  : boolean 

    To Do:
        - rank of every stat in LOO

    """
    def __init__(self, dtm, column_names=[], row_names=[],
        stbl=True, alpha=0.2, randomize=False) :
        """ 
        Args
        ---------- 
        dtm : (sparse) doc-term matrix.
        column_names : list of names for each column of dtm.
        row_names : list of names for each row of dtm.

        """

        if len(row_names) == []:
            row_names = ["smp" + str(i) for i in range(dtm.shape[0])]
        self._smp_ids = dict([
            (s, i) for i, s, in enumerate(row_names[:dtm.shape[0]])
        ])
        self._sparse = scipy.sparse.issparse(dtm) # check if matrix is sparse

        if len(column_names) < dtm.shape[1] :
            column_names = np.arange(1,dtm.shape[1]+1).astype(str).tolist()
        self._column_names = column_names  #: feature name (list)
        if not self._sparse :
            self._dtm = np.asarray(dtm)  #: doc-term-table (matrix)
        else :
            self._dtm = dtm
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
        """ summarize internal doc-term-table """
        
        
        self._terms_per_doc = np.asarray(self._dtm.sum(1).ravel())\
                            .squeeze().astype(int)
        self._counts = np.asarray(self._dtm.sum(0).ravel())\
                    .squeeze().astype(int)
        
        self._internal_scores = []
        for row in self._dtm:
            if self._sparse : 
                cnt = np.squeeze(np.array(row.todense())).astype(int)
            else :
                cnt = np.squeeze(np.array(row)).astype(int)
            pv = self._Pvals_from_counts(cnt, within = True)
            hc, p_thr = self.__compute_HC(pv)
            self._internal_scores += [hc]

    def __compute_HC(self, pvals) :
        np.warnings.filterwarnings('ignore') # when more than one pval is 
        # np.nan numpy show a warning. The current code supress this warning
        pv = pvals[pvals < self._pval_thresh]
        pv = pv[~np.isnan(pv)]
        np.warnings.filterwarnings('always')
        if len(pv) > 0 :
            hc = HC(pv, stbl=self._stbl)
            return hc.HCstar(alpha=self._alpha)
        else :
            return [np.nan], np.nan
        #return hc_vals(pv, stbl=self._stbl,
        #     alpha=self._alpha)

    def get_column_names(self):
        "returns name of each column in table"
        return self._column_names

    def get_featureset(self) :
        """Returns a dictionary with keys = column_names 
        and values =  total count per column """
        return dict(zip(self._column_names,
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
            ls += [dict(zip(self._column_names,counts))]
        return ls

    def get_row_names(self):
        "returns id of each row in table"
        return self._smp_ids

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
            pv = two_sample_pvals_loc(cnt1, cnt2,
                     randomize=self._randomize,
                     )
        else:
            pv = two_sample_pvals_loc(cnt1, cnt0,
                 randomize=self._randomize,
                        )
        return pv 

    def __get_counts(self, dtbl, within=False) :
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

        if list(dtbl._column_names) != list(self._column_names):
            print(
            "Features of 'dtbl' do not match current FreqTable"
            "intance. Changing dtbl accordingly."
            )
            #Warning for changing the test object
            dtbl.change_vocabulary(self._column_names)

        cnt0 = self._counts
        cnt1 = dtbl._counts
        if within:
            cnt0 = cnt0 - cnt1
            if np.any(cnt0 < 0):
                raise ValueError("'within == True' is invalid")
        return cnt0, cnt1

    def get_Pvals(self, dtbl, within=False):
        """ return a list of p-values of another FreqTable with 
        respect to 'self' doc-term table.

        Args: 
            dtbl -- FreqTable object with respect to which to
                    compute pvals
        """
        cnt0, cnt1 = self.__get_counts(dtbl, within=within)
        pv = two_sample_pvals_loc(cnt1, cnt0,
                 randomize=self._randomize,
                        )
        return pv

    def two_table_test(self, dtbl,
                 within=False, stbl=None,
                 randomize=False) :
        """counts, p-values, and HC with 
        respect to another FreqTable
        """
        if stbl == None :
            stbl = self._stbl

        cnt0, cnt1 = self.__get_counts(dtbl, within=within)
        df = two_sample_test_df(cnt0, cnt1,
             stbl=stbl,
            randomize=self._randomize,
            alpha=self._alpha)
        df.loc[:,'feat'] = self._column_names
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
        old_vocab = self._column_names

        no_missing_words = 0
        for i, w in enumerate(new_vocabulary):
            #import pdb; pdb.set_trace()
            try:
                new_dtm[:, i] = self._dtm[:, old_vocab.index(w)]
                                
            except:  # num of words in new vocabulary that 
                     # do not exists in old one
                no_missing_words += 1

        self._dtm = new_dtm
        self._column_names = new_vocabulary

        self.__compute_internal_stat()

    def __dtm_plus_row(self, row) :
        # returns the a copy of the object matrix plus another row
        # row is a matrix of size (1, no_columns)

        if len(np.shape(row)) < 2 :
            row = np.atleast_2d(row)

        assert(row.shape[1] == self._dtm.shape[1])

        if self._sparse :
            dtm_all = vstack([row, self._dtm]).tolil()
        else :     
            dtm_all = np.concatenate([np.array(row), self._dtm], axis = 0)
        return dtm_all

    def __per_row_similairy(self, row) :
        # similarity of each row compared to the rest

        mat = self.__dtm_plus_row(row)
        
        r,c = mat.shape
        lo_scores = []

        for i in range(r) :
            if self._sparse :
                cnt0 = np.squeeze(mat[i,:].toarray())
            else :
                cnt0 = np.squeeze(mat[i,:])
            cnt1 = np.squeeze(np.asarray(mat.sum(0))) - cnt0
            lo_scores += [ self.__similarity(cnt0, cnt1)]

        return lo_scores

    def __per_smp_Pvals_LOO(self, row) :
        pv_list = []
        
        mat = self.__dtm_plus_row(row)
        
        def func(c1, c2) :
            return two_sample_pvals_loc(c1, c2, 
                            randomize=self._randomize)

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

    def __per_smp_Pvals_LOO(self, row) :
        pv_list = []
        
        mat = self.__dtm_plus_row(row)
        
        def func(c1, c2) :
            return two_sample_pvals_loc(c1, c2, 
                            randomize=self._randomize)

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
            dtm = self._dtm[self._smp_ids[smp_id], :]
        else :
            dtm = np.atleast_2d(self._dtm[self._smp_ids[smp_id], :])

        new_table = FreqTable(dtm,
                            column_names=self._column_names,
                            row_names=[smp_id], alpha=self._alpha,
                            randomize=self._randomize, stbl=self._stbl)
        return new_table


    def copy(self) : 
        # create a copy of FreqTable instance
        new_table = FreqTable(
                     self._dtm,
                     column_names=self._column_names,
                     row_names=list(self._smp_ids.keys()), 
                     alpha=self._alpha, randomize=self._randomize,
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
        
        warnings.warn(message, DeprecationWarning, stacklevel=2)

        curr_feat = self._column_names

        for dtbl in lo_dtbl :
            
            feat1 = dtbl._column_names

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
        cnt0, cnt1 = self.__get_counts(dtbl, within=within)
        return two_sample_chi_square(cnt0, cnt1, lambda_ = lambda_)

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
        pvals = two_sample_pvals_loc(cnt0, cnt1)
        #pvals = self.get_Pvals(dtbl, within=within)
        HC, p_thr = self.__compute_HC(pvals)

        return HC

    def get_rank(self, dtbl , LOO=False, within=False) :
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

        if (LOO == False) or (within == True):
            # internal scores are always evaluated in a LOO manner,
            # hence we used internal HC scores in these cases
            
            score = self.get_HC(dtbl, within=within)

            lo_scores = self._internal_scores
            if len(lo_scores) - within > 0: # at least 1 doc not including
                                       # tested doc
                s = np.sum(np.array(lo_scores) <= score) 
                rank = (s - within) / len(lo_scores)
            else:
                rank = np.nan
            
        elif LOO == True :
            cnts = dtbl._counts
            if len(np.shape(cnts)) < 2 :
                cnts = np.atleast_2d(cnts)
            mat = self.__dtm_plus_row(cnts)

            r,c = mat.shape
            lo_scores = []
            for i in range(r) :
                if self._sparse :
                    cnt0 = np.squeeze(mat[i,:].toarray())
                else :
                    cnt0 = np.squeeze(mat[i,:])
                cnt1 = np.squeeze(np.asarray(mat.sum(0))) - cnt0

                pv = two_sample_pvals_loc(cnt0, cnt1,
                             randomize=self._randomize)
                HC,_ = self.__compute_HC(pv)
                lo_scores += [HC]

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
        feat = np.array(self._column_names)[pvals < p_thr]

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
            r = x1.two_table_test(x2)
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
    
