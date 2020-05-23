# HC-based test to discriminate word-frequency tables and attribute authorship. 

## Files:
- AuthAttrib.py -- 2 models for authorship attribution: 
                 - AuthorshipAttributionMulti -- comparision of disputed text to each author
                 - AuthorshipAttributionMultiBinary -- head to head comparison of each author against another
- DocTermHC -- model for constructing large-sacle word-frequency table and HC testing against it. 
- HC_aux.py -- auxiliary functions to evaluate Higher Criticism tests 

To use AuthorshipAttributionMulti and AuthorshipAttributionMultiBinary, arrange your datase in a pandas dataframe with columns *author*, *doc_id*, and *text*
- *author* is the name of the class the document is assoicated with.
- *doc_id* is a unique document identifyer.
- *text* is a string representing the content of the document. 

See AuthorshipAttribution_example.ipynb for a use case in authorship attribution challenges. Here is the Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alonkipnis/AuthorshipAttribution/master?filepath=Examples%2FAuthorshipAttribution_example.ipynb)

This code was used to get the resutls and figures reported in [this](https://web.stanford.edu/~kipnisal/authorship.html) and in the paper:

Alon Kipnis, ``[Higher Criticism for Discriminating Word-Frequency Tables and Testing Authorship](https://arxiv.org/abs/1911.01208)'', 2019

