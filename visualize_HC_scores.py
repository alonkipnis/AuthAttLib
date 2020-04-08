from plotnine import *
import numpy as np
import pandas as pd
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy

LIST_OF_COLORS = ['tab:red', 'tab:blue','tab:gray', "#00BA38", 
    'tab:olive', "#619CFF", 'tab:orange', "#F8766D",
    'tab:purple', 'tab:brown', 'tab:pink',
    'tab:green', 'tab:cyan', 'royalblue', 'darksaltgray', 'forestgreen',
    'cyan', 'navy'
    'magenta', '#595959', 'lightseagreen', 'orangered', 'crimson'
]

def plot_author_pair(df, value = 'HC', wrt_authors = [],
                     show_legend=True):
    
    df1 = df.filter(['doc_id', 'author', 'wrt_author', value])\
            .pivot_table(index = ['doc_id','author'],
                         columns = 'wrt_author',
                         values = [value])[value].reset_index()

    lo_authors = pd.unique(df.wrt_author)
    no_authors = len(lo_authors)
    
    if no_authors < 2 :
        raise ValueError
    
    if wrt_authors == [] :
        wrt_authors = (lo_authors[0],lo_authors[1])

    color_map = LIST_OF_COLORS

    df1.loc[:, 'x'] = df1.loc[:, wrt_authors[0]].astype('float')
    df1.loc[:, 'y'] = df1.loc[:, wrt_authors[1]].astype('float')
    p = (
        ggplot(aes(x='x', y='y', color='author', shape = 'author'), data=df1) +
        geom_point(show_legend=show_legend, size = 3) + geom_abline(alpha=0.5) +
        # geom_text(aes(label = 'doc_id', check_overlap = True)) +
        xlab(wrt_authors[0]) + ylab(wrt_authors[1]) +
        scale_color_manual(values=color_map) +  #+ xlim(0,35) + ylim(0,35)
        theme(legend_title=element_blank(), legend_position='top'))
    return p

def plot_author_pair_label(df, value = 'HC', wrt_authors = [],
                     show_legend=True):
    
    df1 = df.filter(['doc_id', 'author', 'wrt_author', value])\
            .pivot_table(index = ['doc_id','author'],
                         columns = 'wrt_author',
                         values = [value])[value].reset_index()

    lo_authors = pd.unique(df.wrt_author)
    no_authors = len(lo_authors)
    
    if no_authors < 2 :
        raise ValueError
    
    color_map = LIST_OF_COLORS

    df1.loc[:, 'x'] = df1.loc[:, wrt_authors[0]].astype('float')
    df1.loc[:, 'y'] = df1.loc[:, wrt_authors[1]].astype('float')
    p = (
        ggplot(aes(x='x', y='y', color='author', shape = 'author'), data=df1) +
        geom_point(show_legend=show_legend) + geom_abline(alpha=0.5) +
        geom_text(aes(label = 'doc_id', check_overlap = True)) +
        xlab(wrt_authors[0]) + ylab(wrt_authors[1]) +
        scale_color_manual(values=color_map) +  #+ xlim(0,35) + ylim(0,35)
        theme(legend_title=element_blank(), legend_position='top'))
    return p

def plot_author_pair_ROC(df, wrt_authors=('Author1', 'Author2')):
    df1 = df.filter(['doc_id', 'author', 'wrt_author', 'HC', 'rank'])\
            .pivot_table(index = ['doc_id','author'],
                         columns = 'wrt_author',
                         values = ['HC','rank']).HC.reset_index()

    #ROC curve
    def ROC(seq1, seq2):
        cc = np.concatenate([seq1, seq2])
        mx = np.max(cc)
        t = np.linspace(0, mx, 2 * len(cc))
        roc = pd.DataFrame({
            't':
            t,
            'FDR':
            np.mean(np.expand_dims(seq1, 0) < np.expand_dims(t, 1), 1),
            'TDR':
            np.mean(np.expand_dims(seq2, 0) < np.expand_dims(t, 1), 1)
        })
        return roc

    df1.loc[:, 'r'] = df1.loc[:, wrt_authors[1]] / df1.loc[:, wrt_authors[0]]
    FD_scores = df1[df1.author == wrt_authors[0]].r
    TD_scores = df1[df1.author == wrt_authors[1]].r
    row = ROC(FD_scores, TD_scores)

    p = (ggplot(row, aes(x='FDR', y='TDR')) + geom_line(color='red')
         )  # + ggtitle('ROC')
    return p


def plot_author_pair_col(df, value, wrt_authors, test_author):
    df1 = df.filter(['doc_id', 'author', 'wrt_author', value])\
            .pivot_table(index = ['doc_id','author'],
                         columns = 'wrt_author',
                         values = [value])[value].reset_index()


    df1.loc[:, 'x'] = df1.loc[:, wrt_authors[0]].astype('float')
    df1.loc[:, 'y'] = df1.loc[:, wrt_authors[1]].astype('float')

    
    df2 = df1.melt(['author', 'doc_id'], ['x', 'y'],
                   var_name='wrt_author')
    df2.wrt_author = df2.wrt_author.str.replace('x',
                                                wrt_authors[0]).replace(
                                                    'y', wrt_authors[1])
    df2.doc_id = df2.doc_id.astype(int).astype(str)

    p = (ggplot(aes(x='doc_id', y='value', fill='wrt_author'),
                data=df2[df2.author == test_author]) +
         geom_bar(position='dodge', stat="identity", show_legend=True) +
         xlab('Document ID') + ylab(value) +
         scale_fill_manual(values=LIST_OF_COLORS) +
         theme(legend_title=element_blank(), legend_position='top'))
    #ggtitle('Rank wrt each author ' + labels[0] + ' vs '+ labels[1])
    return p

def plot_LDA(df, value, wrt_authors, sym = False) : 

    df1 = df.filter(['doc_id', 'author', 'wrt_author', value])\
            .pivot_table(index = ['doc_id','author'],
                         columns = 'wrt_author',
                         values = [value])[value].reset_index()

    #project to discriminant component    
    if sym :  # project to the line perpendicular to y=x
        df1.loc[:,'t'] = np.dot(df1.filter(wrt_authors), [[1],[-1]]) 
    else : #LDA
        df_red = df1[df1.author.isin(wrt_authors)]
        X = np.array(np.vstack([df_red[wrt_authors[0]], df_red[wrt_authors[1]]]).T)
        y = np.array(df_red.author)
        clf = LinearDiscriminantAnalysis()
        clf.fit(X, y)  
        LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                      solver='lsqr', store_covariance=True, tol=0.0001)
        df1.loc[:,'t'] = clf.transform(df1.filter(wrt_authors))

    #compute means and pooled variance
    df_stat =  df1.groupby('author').t.agg(['var', 'count', 'mean']).loc[wrt_authors,:]
    num = 0
    den = 0
    for c in df_stat.iterrows() :
        num += c[1]['var'] * (c[1]['count'] - 1)
        den += c[1]['count']
    pooled_std = np.sqrt( num / (den - len(df_stat)))
    
    
    df1.loc[:,wrt_authors[0]] = scipy.stats.norm.pdf(df1.t,
                        loc = df_stat.loc[wrt_authors[0],'mean'],
                        scale = pooled_std
                        )
    
    df1.loc[:,wrt_authors[1]] = scipy.stats.norm.pdf(df1.t,
                        loc = df_stat.loc[wrt_authors[1],'mean'],
                        scale = pooled_std
                        )
    
    df_plot = df1.melt(['t','author','doc_id'], [wrt_authors[0],wrt_authors[1]])
 
    p = (
        ggplot(
                aes(
                x='t',
                y='value',
                color='author',
                fill='author',
                label='doc_id'
                ),
            data=df_plot
            ) + geom_rug(
            aes(x='t', y=0, color='author'),
            position=position_jitter(height=0),
            size=1
            ) + geom_bar(
                aes(x='t', y='value'),
                stat='identity',
                position='dodge',
                size=1) +
         #geom_label(aes(y = 'value'), color = 'black') + 
         scale_fill_manual(LIST_OF_COLORS) + scale_color_manual(LIST_OF_COLORS)+
         stat_function(fun = scipy.stats.norm.pdf,
                       args = {'loc' : df_stat.loc[wrt_authors[0],'mean'],
                               'scale' : pooled_std}, color = LIST_OF_COLORS[0]) +
         stat_function(fun = scipy.stats.norm.pdf,
                       args = {'loc' : df_stat.loc[wrt_authors[1],'mean'],
                               'scale' : pooled_std}, color = LIST_OF_COLORS[1])
         + ylab('prob') + xlab('projected score')
         )
    return p

def plot_col(df, value, sign, wrt_authors = []) :
    """ Bar plot of 'value' by term

    Parameters:
        df -- DataFrame with columns 'term'|value|'sign'
        value -- amplitude and order of columns
        sign -- real number represents which author use term more frequently
        wrt_authors -- name of two authors (only needed to display legend)

    """
    df = df.sort_values(value).reset_index()
    df['term'] = pd.Categorical(df['term'], categories=df['term'].values[::-1], ordered=True)
    
    if len(wrt_authors) == 0 :
        df.loc[:,'used more in'] = True
        show_legend = False
    else :
        df.loc[:,'used more in'] = df[sign].apply(lambda x : wrt_authors[0] if x>0 else wrt_authors[1])
        show_legend = True

    p = (ggplot(aes(x='term', y=value, fill='used more in'),
                data=df.dropna()) +
         geom_bar(position='dodge', stat="identity", show_legend=show_legend) +
         scale_fill_manual(values = LIST_OF_COLORS) +
         coord_flip() + 
         ylab(value) +
         xlab('') +
         scale_y_log10() +
         theme(legend_position='top')
        )        
    return p
