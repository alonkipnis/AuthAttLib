# pipeline: data science
# project: bib-scripts

import pandas as pd
import logging
import sys
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import KFold

sys.path.append("../")
from MultiDoc import CompareDocs

pd.options.mode.chained_assignment = None


def arrange_data(data):
    # ds = data.rename(columns={'doc_no': 'doc_id'}).dropna()
    data.loc[:, 'doc_tested'] = data['doc_id']
    data.loc[:, 'len'] = data['text'].apply(lambda x: len(x.split()))
    return data


def n_most_frequent_words_per_author(df, n, words_to_ignore=[], ngram_range=(1, 1)):
    """
        Return 'n' of the most frequent tokens in the corpus represented by the
        list of strings 'texts'
    """

    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"

    tf_vectorizer = CountVectorizer(stop_words=words_to_ignore, token_pattern=pat, ngram_range=ngram_range)

    vocab = []
    for auth in df.author.unique():
        tf = tf_vectorizer.fit_transform(df[df.author == auth].text)
        feature_names = np.array(tf_vectorizer.get_feature_names_out())

        idcs = np.argsort(-tf.sum(0))
        vocab += list(np.array(feature_names)[idcs][0][:n])

    return list(set(vocab))


def n_most_frequent_words(texts, n, words_to_ignore=[], ngram_range=(1, 1)):
    """
        Returns the 'n' most frequent tokens in the corpus represented by the
        list of strings 'texts'
    """

    pat = r"\b\w\w+\b|[a\.!?%\(\);,:\-\"\`]"

    tf_vectorizer = CountVectorizer(stop_words=words_to_ignore,
                                    token_pattern=pat,
                                    ngram_range=ngram_range)
    tf = tf_vectorizer.fit_transform(list(texts))
    feature_names = np.array(tf_vectorizer.get_feature_names_out())

    idcs = np.argsort(-tf.sum(0))
    vocab_tf = np.array(feature_names)[idcs][0]
    return list(vocab_tf[:n])


def create_vocabulary(data, no_tokens, by_author):
    assert ('author' in data.columns)
    if by_author:
        return n_most_frequent_words_per_author(data, no_tokens)
    else:
        return n_most_frequent_words(data, no_tokens)


# pipeline: data science val
# project: bib-scripts


def build_model(data: pd.DataFrame, model_params) -> CompareDocs:
    """
    Returns a model object

    Args:
    -----
    data    doc_id|author|term

    """

    vocab = create_vocabulary(data, no_tokens=model_params['no_tokens'], by_author=model_params['by_author'])

    model = CompareDocs(vocabulary=vocab, **model_params)
    train_data = {}

    for auth in data.groupby('author'):
        train_data[auth[0]] = " ".join(auth[1].text)

    model.fit(train_data)
    return model


def main():
    data = arrange_data(pd.read_csv("../Data/PAN2018_probs_1_to_4.csv"))
    data = data[data.prob.isin(['problem00002', 'problem00003', 'problem00004'])]

    model_params = dict(no_tokens=250,  # most frequent tokens,
                        by_author=True,  # most frequent by each author of the known_authors list
                        feat_reduction_method="none",  # options are: div_persuit, one_vs_many, none
                        gamma=.2,
                        stbl=True,
                        min_cnt=1)

    for data_prob in data.groupby('prob'):
        print(f"Solving problem {data_prob[0]}. Number of candidates = {len(data_prob[1].author.unique())}")
        data_train = data_prob[1][~data_prob[1]['doc_no'].str.contains('test_')]
        data_test = data_prob[1][data_prob[1]['doc_no'].str.contains('test_')]

        model = build_model(data_train, model_params)

        results = pd.DataFrame()
        for doc in data_test.groupby('doc_id'):
            assert len(doc[1]) == 1
            r = model.predict(doc[1].text.values[0])
            r1 = model.predict_proba(doc[1].text.values[0]) # similarity with respect to each class
            rec = pd.DataFrame(dict(problem=data_prob[0],
                                    author=doc[1].author,
                                    doc_id=doc[0],
                                    predicted_author=r,
                                    ))
            results = pd.concat([results, rec])
        acc = np.mean(results['author'] == results['predicted_author'])
        print(f"\tAccuracy = {acc}")


if __name__ == '__main__':
    main()
