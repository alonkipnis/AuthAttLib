# HC-based test to discriminate word-frequency tables and attribute authorship. 

## Files:
- AuthAttrib.py -- 2 models for authorship attribution: 
                 - AuthorshipAttributionMulti -- comparision of disputed text to each author
                 - AuthorshipAttributionMultiBinary -- head to head comparison of each author against another
- DocTermHC -- model for constructing large-sacle word-frequency table and HC testing against it. 
- HC_aux.py -- auxiliary functions to evaluate Higher Criticism tests 

In order to use AuthorshipAttributionMulti and AuthorshipAttributionMultiBinary arrange input data in a pandas dataframe with columns 
> <author>, <doc_id>, <text> 
- author is the name of the class document is assoicated with.
- doc_id is a unique document identifyer.
- text contains the text (content) of the document. 

See AuthorshipAttribution_example.ipynb for usage in authorship attribution challenges. 

This code was used to get the resutls and figures reported in [this](https://web.stanford.edu/~kipnisal/authorship.html) page and in the paper:

Alon Kipnis, ``[Higher Criticism for Discriminating Word-Frequency Tables and Testing Authorship](https://arxiv.org/abs/1911.01208)'', 2019

