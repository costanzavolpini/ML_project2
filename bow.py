from sklearn.feature_extraction.text import CountVectorizer
import string

# tokenize helper function
def text_process(raw_text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in list(raw_text) if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.lower().split()]

def generateBow(tweets):
    "Convert each message which is represented by a list of tokens into a vector that a machine learning model can understand."
    # vectorize
    vecBOW = CountVectorizer(analyzer=text_process).fit(tweets)
    return vecBOW

def getVectorRepresentation(vecBOW, tweet):
    return vecBOW.transform([tweet])

def vectorizeDataSet(vecBOW, tweets):
    return vecBOW.transform(tweets)
