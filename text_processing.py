""" Functions for parsing, processing, and cleaning text. 
"""
import re
import bs4 
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 




def remove_parts_of_speach(text, 
                        to_remove = ('NNP', 'NNPS', 'CD'),
                        lemmatize = True) :

    # 'NNP'-- proper noun, singluar
    # 'NNPS' -- proper noun, plural 
    # 'CD' -- cardinal digit
    # 'PRP' -- personal pronoun
    # 'PRP$' -- posessive pronoun
    # stem and remove numbers
    text_pos = nltk.pos_tag(nltk.word_tokenize(text))

    if lemmatize :
        lemmatizer = WordNetLemmatizer() 
        lemmas = [lemmatizer.lemmatize(w[0]) for w in text_pos if \
                  w[1] not in to_remove and 
                  (len(re.findall('[0-9]',w[0])) == 0)]
    else :
        lemmas = [w[0] for w in text_pos if \
                  w[1] not in to_remove and
                (len(re.findall('[0-9]',w[0])) == 0)
                  ]
    return " ".join(lemmas)

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

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

