import re
from nltk.corpus import stopwords
import string
from string import punctuation
import itertools
import enchant
from split_hashtags import *



##################### CORPUS ########################
# Replace word with mistakes (replace char, insert char, delete char) and slang words
# using corpus

# Use dictionary from:
# http://luululu.com/tweet/typo-corpus-r1.txt
# http://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz
# http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/
# to handle abbreviations, mistakes...etc. (IN )

# LULU-CORPUS
# (1) INSERT (IN): a character is added to the original word.
# (2) REMOVE (RM): a character is removed from the original word.
# (3) REPLACE1 (R1): the order of character is different from the original word (the number of differences is one).
# (4) REPLACE2 (R2): a character is different from the original word

d = enchant.Dict('en_US')
final_corpus = {}

def remove_repetitions(tweet):
    """
    Functions that remove noisy character repetition like for instance :
    llloooooooovvvvvve ====> love
    This function reduce the number of character repetition to 2 and checks if the word belong the english
    vocabulary by use of pyEnchant and reduce the number of character repetition to 1 otherwise
    Arguments: tweet (the tweet)
    """
    tweet=tweet.split()
    for i in range(len(tweet)):
        tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i])).replace('#', '')
        if len(tweet[i])>0:
            if not d.check(tweet[i]):
                tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet=' '.join(tweet)
    return tweet

def corpusReplace(corpus):
    for word in corpus:
        word = word.decode('utf8')
        word = word.split()
        final_corpus[word[0]] = word[1]

corpus_lulu = open('../corpus/lulu-corpus.txt', 'rb')
corpusReplace(corpus_lulu)
corpus_lulu.close()

corpus_emnlp = open('../corpus/emnlp-corpus.txt', 'rb')
corpusReplace(corpus_emnlp)
corpus_emnlp.close()

corpus_feiliu = open('../corpus/fei-liu.txt', 'rb')

for word in corpus_feiliu:
    word = word.decode('utf8')
    word = word.split()
    final_corpus[word[1]] = word[3]

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
    splitted_tweet = tweet.split(' ')
    for (i, word) in enumerate(splitted_tweet):
        if re.match(r"#[A-Za-z0-9]+", word):
            splitted_tweet[i] = infer_spaces(word[1:])
            
    tweet = ' '.join(splitted_tweet)
    
    tweet = re.sub('<url>','',tweet)
    tweet = re.sub('<user>','',tweet)
    
    tweet = re.sub(r"\'s", " \'s", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet)
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)
    
    
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'#', ' ', tweet) #hashtag
    
    tweet = tweet.strip(' ')
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet.lower()


def applyPreProcessing(text):
    
    text = cleanTweet(text)
    text = remove_repetitions(text)
    text = applyCorpus(text)
    return text
