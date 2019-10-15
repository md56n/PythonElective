import requests
from bs4 import BeautifulSoup
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk

#extrat URL using BeautifulSoup
html = requests.get("https://umkc.box.com/s/7by0f4540cdbdp3pm60h5fxxffefsvrw")

bsObj = BeautifulSoup(html.content, "html.parser")
words=bsObj.text
import re, collections

#tokenize (break up text into words)
def tokens(text):
    """
    Get all words from the corpus
    """
    return re.findall('[a-z]+', text.lower())


WORDS = tokens(open('big.txt').read())
WORD_COUNTS = collections.Counter(WORDS)
print(WORDS)
# top 10 words in corpus
print(WORD_COUNTS.most_common(10))

#part of speech (POS) tagging (assigns tag to each word)
stokens = nltk.sent_tokenize(words)
wtokens = nltk.word_tokenize(words)

for s in stokens:
    print(s)
for t in wtokens:
    print(t)

print(nltk.pos_tag(wtokens))

#stemming, extract root from each word
pStemmer = PorterStemmer()
print(pStemmer.stem(words))

#lemmatization (helps give root version of word) normalizing data
lemmatizer = WordNetLemmetizer()
print(lemmatizer.lemmatize(words))

#chunk into groups of three
from nltk.util import ngrams

trigrams = ngrams (wtokens, 3)
for i in trigrams:
    print(i)
print(ne_chunk(pos_tag(wordpunct_tokenize(words))))


#after normalization, feed new data to model to test
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

#reading new data
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

#feeds to model
tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

# print(tfidf_Vect.vocabulary_)/train classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

#test data
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)
