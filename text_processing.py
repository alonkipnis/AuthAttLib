""" Various simple functions for parsing, processing, and cleaning text. 
"""

import re
import bs4 
from nltk.stem.snowball import SnowballStemmer

def find_propoer_names(text) : 
    #return all capitalized words preceeded by another word 
    #(does not capture two consecutive capitalized word)
    return re.findall(r"([A-Za-z])\s([A-Z][A-Za-z]+)", text)

def remove_proper_names(text) : 
    return re.sub(r"([A-Za-z])\s([A-Z][A-Za-z]+)","", text)

def find_capitalized_words(text) : 
    #return all capitalized words with preceeded by a space
    return " ".join([w for w in re.findall(r"\s([A-Z][A-Za-z]+)", text)])

def html_to_text(text_in_html) :
    soup = bs4.BeautifulSoup(text_in_html, "html.parser")
    return soup.get_text()

def stem_text(text, lang = 'english') :
    stemmer = SnowballStemmer(lang)
    return " ".join([stemmer.stem(w) for w in re.split('\W+', text)])

def collapse_terms(lo_terms, term, text) :
    #replaces every word in 'text' appearing in 'lo_terms' with 'term'
    for st in lo_terms :
        text = " ".join([w.replace(st,term) for w in text.split()])
    return text

def remove_digits(text) :
    return re.sub("[0-9]", "", text)

def remove_hexa_symbols(text) :
    #replace with a space
    return re.sub("\\\\x[0-9a-f]+"," ",text)

def remove_CR_LF(text) :
    #replace with a space
    return re.sub("\\\\(n|r)"," ",text)

def preprocess_text(text, stem = True, clean_names = True,
               clean_html_tags = True, clean_digits = True
               ) : 
    
    text_st = text
    
    if clean_html_tags : 
        text_st = html_to_text(text_st)
    
    if stem :
        text_st = stem_text(text_st)
        
    if clean_names :
        text_st = remove_proper_names(text_st)
        #lo_proper_names = find_capitalized_words(text_st)
        #text_st = " ".join(w for w in text_st.split() if w not in lo_proper_names)
        
    if clean_digits :
        text_st = remove_digits(text_st)
    
    return text_st

