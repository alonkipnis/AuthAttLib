import codecs
import numpy as np
import pandas as pd

Hamilton_known = [1,6,7,8,9,11,12,13,15,16,17,21,22,23,24,25,
                  26,27,28,29,30,31,32,33,34,35,36,59,60,61,
                  65,66,67,68,69,70,71,72,73,74,75,76,77]
Madison_known = [10,14,37,38,39,40,41,42,43,44,45,46,47,48]
disputed = [49,50,51,52,53,54,55,56,57,58,62,63]
Jay_known = [2,3,4,5,64]
joint = [18,19,20]

def Authorship(paper_no) :
    if (paper_no in Hamilton_known) :
        return 'Hamilton'
    elif (paper_no in Madison_known) : 
        return 'Madison'
    elif (paper_no in disputed) : 
        return 'disputed'
    elif (paper_no in Jay_known) : 
        return 'Jay'
    else :
        return 'ignore'

def load_Federalists_Papers(path) :
    f = codecs.open(path,'r',encoding='utf-8')
    by_line = f.read().split('\n')
    df = pd.DataFrame({'text' : by_line})
    f.close()

    df['paper_no'] = np.cumsum(df['text'].str.contains("FEDERALIST No."))
    df['tmp'] = np.cumsum(df['text'].str.contains("To the People of the State of New York"))
    df['header'] = df['paper_no']  - df['tmp']
    df['header'] += df['header'] + df['text'].str.contains("To the People of the State of New York:")
    df = df[(df['header'] == 0) & (df['paper_no'] > 0)]

    df = df.groupby('paper_no')['text'].apply(lambda x : " ".join(x)).reset_index()
    
    df['author'] = df['paper_no'].transform(Authorship)
    df = df[df['author'].isin(['Hamilton','Madison','disputed'])]
    
    
    print("Documents loaded:")
    lo_auth = pd.unique(df['author'])
    for auth in lo_auth :
        print("\t {} {} papers".format(len(pd.unique(df[df.author == auth].paper_no)), auth))
        
    return df

