import requests
import pandas as pd

Austin = [
('Jane Austin', 'Persuasion', "http://www.gutenberg.org/cache/epub/105/pg105.txt"),
('Jane Austin', 'Emma', "https://www.gutenberg.org/files/158/158-0.txt"),
('Jane Austin', 'Lady Susan', "http://www.gutenberg.org/cache/epub/946/pg946.txt"),
('Jane Austin', 'Mansfield Park', "https://www.gutenberg.org/files/141/141-0.txt"),
('Jane Austin', 'Northanger Abbey', "https://www.gutenberg.org/files/121/121-0.txt"),
('Jane Austin', 'Pride and Prejudice', "https://www.gutenberg.org/files/1342/1342-0.txt"),
('Jane Austin', 'Love and Friendship (Early Works)', "https://www.gutenberg.org/files/1212/1212-0.txt")
]
Dickens = [
('Charles Dickens', 'Barnaby Rudge', "https://www.gutenberg.org/files/917/917-0.txt"),
('Charles Dickens', 'Children Stories', "http://www.gutenberg.org/cache/epub/37121/pg37121.txt"),
('Charles Dickens', 'The Battle of Life', "http://www.gutenberg.org/cache/epub/676/pg676.txt"),
('Charles Dickens', 'Bleak House', "http://www.gutenberg.org/cache/epub/1023/pg1023.txt"),
('Charles Dickens', 'David Copperfield', "https://www.gutenberg.org/files/766/766-0.txt"),
('Charles Dickens', 'Doctor Marigold', "http://www.gutenberg.org/cache/epub/1415/pg1415.txt"),
('Charles Dickens', 'Great Expectations', "https://www.gutenberg.org/files/1400/1400-0.txt"),
('Charles Dickens', 'Oliver Twist', "http://www.gutenberg.org/cache/epub/730/pg730.txt"),
('Charles Dickens', 'A Tale of Two Cities', 'https://www.gutenberg.org/files/98/98-0.txt'),
('Charles Dickens', 'A Chrismas Carol', 'https://www.gutenberg.org/files/46/46-0.txt'),
('Charles Dickens', 'Little Dorit', 'https://www.gutenberg.org/files/963/963-0.txt'),
('Charles Dickens', 'Hard Times', 'https://www.gutenberg.org/files/786/786-0.txt'),
('Charles Dickens', 'The Pickwick Papers', 'https://www.gutenberg.org/files/580/580-0.txt'),
('Charles Dickens', 'Our Mutual Friend', 'https://www.gutenberg.org/files/883/883-0.txt'),
('Charles Dickens', 'The Mufdog and Other Sketches', 'https://www.gutenberg.org/files/912/912-0.txt'),
('Charles Dickens', 'Nicholas Nickleby', 'https://www.gutenberg.org/files/967/967-0.txt'),
('Charles Dickens', 'The Old Curiosity Shop', 'https://www.gutenberg.org/files/700/700-0.txt'),
('Charles Dickens', 'A Child\'s History of England' , 'http://www.gutenberg.org/cache/epub/699/pg699.txt'),
('Charles Dickens', 'The Mystery of Edwin Drood', 'https://www.gutenberg.org/files/564/564-0.txt'),
('Charles Dickens', 'Martin Chuzzlewit', 'https://www.gutenberg.org/files/968/968-0.txt'),
('Charles Dickens', 'Sketches by Boz', 'https://www.gutenberg.org/files/882/882-0.txt'),
('Charles Dickens', 'Dombey and son', 'https://www.gutenberg.org/files/821/821-0.txt'),
('Charles Dickens', 'The Lamplighter', 'https://www.gutenberg.org/files/927/927-0.txt')
]

def download_text(url) :    
    r = requests.get(url)
    if r.status_code == requests.codes.ok :
        txt = r.text.encode('UTF-8')
        return txt
    else:
        print('Failed')
        return ""

lo_lists = {'Austin' :  Austin, 
            'Dickens' : Dickens}
    
for ls in lo_lists :
    data = pd.DataFrame(lo_lists[ls], columns=['author', 'title', 'url'])
    data.loc[:,'text'] = data.url.apply(download_text)
    for r in data.iterrows() :
        print(" Downloaded {} by {} of length {} words".format(r[1].title,r[1].author, len(r[1].text.split())) )

    data.to_csv('../Data/' + str(ls) + '.csv')