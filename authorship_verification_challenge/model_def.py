
import sys
lib_path = '/Users/kipnisal/'
sys.path.append(lib_path)
from AuthAttLib.FreqTable import *
from scipy.stats import norm
from AuthAttLib.utils import to_docTermCounts

# training == creating and instance of HCsimCls with the right vocabulary 

class HCsimCls(object) :
    def __init__(self, vocab, radius=0.05, ng_range=(1,1), gamma=0.2) :
        self.radius = radius
        self.vocab = vocab
        self.ng_range = ng_range
        self.gamma = gamma
        self.prob1 = None
        self.prob2 = None
        self.p00 = 1/2
        self.p01 = 1/2
        self.param = None
        

    def set_prob(self, **kwargs) :
        self.param = kwargs

        
    def HC2prob(self, hc) :
        kwargs = self.param
       
        f0 = lambda x : norm.pdf(x, self.param['mu0'], self.param['sigma0'])
        f1 = lambda x : norm.pdf(x, self.param['mu1'], self.param['sigma1'])

        prob = self.p00 * f0(hc) \
                / (self.p00 * f0(hc) + self.p01 * f1(hc))
        return prob
    
    def get_HCsim(self, txt1, txt2) :
        """
        Evaluate HC discrepancy between two texts
        """
        dtc, feat = to_docTermCounts([txt1, txt2], ngram_range=self.ng_range, vocab=self.vocab)
        ft = FreqTable(dtc, column_labels=feat, row_labels=['smp1', 'smp2'], gamma=self.gamma)
        ft1 = ft.get_row_as_FreqTable('smp1')
        ft2 = ft.get_row_as_FreqTable('smp2')
        return ft1.get_HC(ft2)
    
    def predict_proba(self, txt1, txt2, value='HC') :        
        hc = self.get_HCsim(txt1, txt2)
        prob = self.HC2prob(hc)
        return 0.5 * (np.abs(prob-0.5) <= self.radius) + prob * (np.abs(prob-0.5) > self.radius)
