# pipeline: data science
# project: bib-scripts

import sys
import pandas as pd
import numpy as np

sys.path.append("../")
from MultiDoc import CompareDocs, ListFeatures

pd.options.mode.chained_assignment = None


def arrange_data(data):
    # ds = data.rename(columns={'doc_no': 'doc_id'}).dropna()
    data.loc[:, 'doc_tested'] = data['doc_id']
    data.loc[:, 'len'] = data['text'].apply(lambda x: len(x.split()))
    return data


def build_model(data: pd.DataFrame, model_params) -> CompareDocs:
    """
    Returns a model object

    Args:
    -----
    data    doc_id|author|term

    """

    vocab = ListFeatures(max_features=model_params['no_tokens'])(data.text, data.author)
    model = CompareDocs(vocabulary=vocab, **model_params)
    train_data = {}

    for auth in data.groupby('author'):
        train_data[auth[0]] = " ".join(auth[1].text)

    model.fit(train_data)
    return model


def main():
    data = arrange_data(pd.read_csv("../Data/PAN2018_probs_1_to_4.csv"))
    data = data[data.prob.isin(['problem00002', 'problem00003', 'problem00004'])]
    data = data[data.prob.isin(['problem00004'])]

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
            print(r1)
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
