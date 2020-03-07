# text processing
import re
import string
import nltk
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np

# read the csv file with amazon reviews
reviews_df = pd.read_csv('C:/Users/shs6g/Desktop/KDM/code/ICP6/reviews.csv', error_bad_lines=False)
reviews_df['Reviews'] = reviews_df['Reviews'].astype(str)
print(reviews_df.head(6))


def initial_clean(text):
    """
    Function to clean text-remove punctuations, lowercase text etc.
    """
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower()  # lower case text
    text = nltk.word_tokenize(text)
    return (text)


stop_words = stopwords.words('english')
stop_words.extend(
    ['news', 'say', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'took', 'time',
     'year',
     'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make',
     'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come', 'new', 'said',
     'like', 'people'])


def remove_stop_words(text):
    return [word for word in text if word not in stop_words]


stemmer = PorterStemmer()


def stem_words(text):
    """
    Function to stem words
    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1]  # no single letter words
    except IndexError:
        pass

    return text


def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))


# clean reviews and create new column "tokenized"
import time

t1 = time.time()
reviews_df['tokenized_reviews'] = reviews_df['Reviews'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(reviews_df), "reviews:", (t2 - t1) / 60,
      "min")  # Time to clean and tokenize 3209 reviews: 0.21254388093948365 min

print('\n')
print("reviews with their respective tokenize version:")
print(reviews_df.head(5))
# LDA
import gensim
import pyLDAvis.gensim

# Create a Gensim dictionary from the tokenized data
tokenized = reviews_df['tokenized_reviews']
# Creating term dictionary of corpus, where each unique term is assigned an index.
dictionary = corpora.Dictionary(tokenized)
# Filter terms which occurs in less than 1 review and more than 80% of the reviews.
dictionary.filter_extremes(no_below=1, no_above=0.8)
# convert the dictionary to a bag of words corpus
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
print(corpus[:1])

print([[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]])

# LDA
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=7, id2word=dictionary, passes=15)
# saving the model
ldamodel.save('model_combined.gensim')
topics = ldamodel.print_topics(num_words=4)
print('\n')
print("Now printing the topics and their composition")
print("This output shows the Topic-Words matrix for the 7 topics created and the 4 words within each topic")
for topic in topics:
    print(topic)

# finding the similarity of the first review with topics
print('\n')
print("first review is:")
print(reviews_df.Reviews[0])
get_document_topics = ldamodel.get_document_topics(corpus[0])
print('\n')
print("The similarity of this review with the topics and respective similarity score are ")
print(get_document_topics)

# visualizing topics
lda_viz = gensim.models.ldamodel.LdaModel.load('model_combined.gensim')
lda_display = pyLDAvis.gensim.prepare(lda_viz, corpus, dictionary, sort_topics=True)
pyLDAvis.show(lda_display)
