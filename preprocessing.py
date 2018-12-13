import re
from nltk.corpus import stopwords
import string
from string import punctuation

##################### CORPUS ########################
# Replace word with mistakes (replace char, insert char, delete char) and slang words
# using corpus

# Use dictionary from http://luululu.com/tweet/typo-corpus-r1.txt
# http://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz
# to handle abbreviations, mistakes...etc. (IN )

# LULU-CORPUS
# (1) INSERT (IN): a character is added to the original word.
# (2) REMOVE (RM): a character is removed from the original word.
# (3) REPLACE1 (R1): the order of character is different from the original word (the number of differences is one).
# (4) REPLACE2 (R2): a character is different from the original word

final_corpus = {}

def corpusReplace(corpus):
    for word in corpus:
        word = word.decode('utf8')
        word = word.split()
        final_corpus[word[0]] = word[1]

def applyCorpus(tweet):
    new_tweet = ''
    for w in tweet.split(' '):
        if w in final_corpus.keys():
            #Replace with correct value
            new_word = final_corpus[w]
            new_tweet = new_tweet + ' ' + new_word
        else:
            new_tweet = new_tweet + ' ' + w
    return new_tweet


# remove <user>, <url>, punctuation..etc.
def cleanTweet(tweet):
    tweet = re.sub('<url>','',tweet)
    tweet = re.sub('<user>','',tweet)
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    tweet = re.sub(r'#\w*', '', tweet) #hashtag
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet) # puntuaction
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = tweet.lstrip(' ')
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet


def applyPreProcessingCorpus():
    corpus_lulu = open('corpus/lulu-corpus.txt', 'rb')
    corpusReplace(corpus_lulu)
    corpus_lulu.close()

    corpus_emnlp = open('corpus/emnlp-corpus.txt', 'rb')
    corpusReplace(corpus_emnlp)
    corpus_emnlp.close()

def applyPreProcessing(text):
    t1 = applyCorpus(text)
    t2 = cleanTweet(t1)
    return t2