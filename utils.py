#supporting functions for AuthAttrLib

import pandas as pd
import numpy as np
from tqdm import *
import re


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


def n_most_frequenct_words_balanced(
            df,
            n, 
            ngram_range = (1,1),
            words_to_ignore = []
            ) :
    """
        Returns n of the most frequent tokens by each author 
        in the corpus represented by the dataframe df.

        Takes approximately equals number of words from each author

        df has columns 'author', 'text', 'doc_id'

    """

    import random
    df1 = pd.DataFrame(df.groupby('author').text.sum()).reset_index()
    df1.loc[:,'len'] = df1.text.apply(lambda x : len(x.split()))
    df1.loc[:,'min'] = df1.len.min()
    df1.apply(lambda r : random.sample(
        population=r['text'].split(),
        k= r['min']),
        axis = 1
            )

    return n_most_frequent_words(
        df1.text,
        n=n,
        ngram_range=ngram_range,
        words_to_ignore=words_to_ignore
                    )


def n_most_frequent_words(texts, n, 
                words_to_ignore=[],
                ngram_range=(1, 1),
                balanced=False
                         ):
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


def frequent_words_tfidf(
    texts,
    no_words,
    ngram_range=(1,1),
    words_to_ignore=[]
                        ):
    """
        Returns no_words with LOWEST tf-idf score.
        Useful in removing proper names and rare words. 
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                       min_df=0,
                                       ngram_range = ngram_range,
                                       sublinear_tf=True,
                                       stop_words=words_to_ignore)
    tfidf = tfidf_vectorizer.fit_transform(list(texts))
    feature_names = tfidf_vectorizer.get_feature_names()

    idcs = np.argsort(tfidf.sum(0))
    vocab_tfidf = np.array(feature_names)[idcs][0]
    return vocab_tfidf[-no_words:]


def term_counts(text, vocab=[], symbols=[]):
    """return a dataframe of the form feature|n representing 
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

def to_docTermCounts(lo_texts, vocab=[],
                         max_features=500, ngram_range=(1, 1)):
    """
   convert list of strings to a doc-term matrix
   returns term-counts matrix (sparse) and a list of feature names

   Args:
        lo_texts -- each item in this list represents a different
                    document and is summarized by a row in the output
                    matrix
        vocab -- is the preset list of tokens to count. If empty, use...
        max_features -- ... number of words
        ngram_range -- is the ngram range for the vectorizer. 
                        Note: you must provide the ngram range even if a preset
                        vocabulary is used. This is due to the interface of
                        sklearn.CountVectorizer.
    Returns:
        tf -- term-frequency matrix (in sparse format)
        feature_names -- list of token names corresponding to rows
                        in tf. 
    """

    from sklearn.feature_extraction.text import CountVectorizer

    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"

    if vocab == []:
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
