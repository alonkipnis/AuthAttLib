import logging
import pandas as pd
import numpy as np
from MultiDoc import CompareDocs, ListFeatures


class HCsim(object):
    def __init__(self, gamma=0.2, stbl=True, **kwargs):
        self._gamma = gamma
        self._stbl = stbl
        self._vocab_params = kwargs

    def __call__(self, text1, text2):
        return self.documents_similarity(text1, text2)

    def compare_docs(self, text1, text2):
        vocab = ListFeatures(**self._vocab_params)([text1, text2])
        cd = CompareDocs(vocabulary=vocab, measures=['HC'])
        cd.fit({'doc1': text1, 'doc2': text2})
        return cd.HCT(gamma=self._gamma, stbl=self._stbl)

    def documents_similarity(self, text1, text2):
        dfr = self.compare_docs(text1, text2)
        print(dfr)
        return dfr.HC.values[0]