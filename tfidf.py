from sklearn.feature_extraction.text import TfidfTransformer

def getTFIDFTransformer(bowTeewts):
    "Map all bag of words tweets to a tfidf value"
    return TfidfTransformer().fit(bowTeewts)

def getTFIDFBowSingle(tfidfTransformer, bowTweet):
    "Get corresponding value tfidf given a tweet"
    "Example: tfidf_transformer.idf_[vecBOW.vocabulary_['hello']] => 6.6840798423603545"
    "tfidf_transformer is getTFIDFTransformer(bowTeewts)"
    return tfidfTransformer.transform(bowTweet)