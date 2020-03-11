import re
import string
import nltk
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import time
import gensim
import pyLDAvis.gensim


def clean(text):
    '''
    Remove punctuation, capital letters and stopwords from text.
    Apply stemming to words.
    '''
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1]  # no single letter words
    except IndexError:
        pass
    return text


# Read csv and prepare dataframe
df = pd.read_csv('/home/andrew/CS5560-KDM/ICP6/reviews.csv', error_bad_lines=False)
df['content'] = df['content'].astype(str)

# Stopwords to extract from text
stop_words = stopwords.words('english')
stop_words.extend(
    ['news', 'say', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'took', 'time',
     'year', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack',
     'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come', 'new',
     'said', 'like', 'people'])

# Apply Porter Stemming algorithm
stemmer = PorterStemmer()

# Measure the time preparing words
startTime = time.time()
df['tokenized_content'] = df['content'].apply(clean)
stopTime = time.time()
print('Cleaning & tokenizing', len(df), 'reviews:', (stopTime - startTime) / 60, 'min\n')
print("Reviews and their tokenized version:")
print(df.head(5))

# Create a Gensim dictionary from the tokenized data
tokenized = df['tokenized_content']

# Create a term dictionary of corpus, where each unique term is assigned an index
dictionary = corpora.Dictionary(tokenized)

# Filter terms by frequency
# no_below is min number of doc appearances
# no_above is max percentage of doc appearances
dictionary.filter_extremes(no_below=20, no_above=0.25)

# Convert the dictionary to a bag-of-words corpus
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]

print(corpus[:1])
print([[(dictionary[idx], freq) for idx, freq in cp] for cp in corpus[:1]])

# LDA
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=10)
# saving the model
ldamodel.save('model_combined.gensim')
topics = ldamodel.print_topics(num_words=3)
print('\n')
print("Now printing the topics and their composition")
print("This output shows the Topic-Words matrix for the 7 topics created and the 4 words within each topic")
for topic in topics:
    print(topic)

# finding the similarity of the first review with topics
print('\n')
print("first review is:")
print(df.content[0])
get_document_topics = ldamodel.get_document_topics(corpus[0])
print('\n')
print("The similarity of this review with the topics and respective similarity score are ")
print(get_document_topics)

# visualizing topics
lda_viz = gensim.models.ldamodel.LdaModel.load('model_combined.gensim')
lda_display = pyLDAvis.gensim.prepare(lda_viz, corpus, dictionary, sort_topics=True)
pyLDAvis.show(lda_display)
