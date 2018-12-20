posfile = open('train_pos_full.txt', 'rb')
pos_word_counts = []

for tweet in posfile.readlines():
    tweet = tweet.decode('utf-8')
    w_len = len(tweet.split(' '))
    if w_len > 40:
        print(tweet, w_len)
    pos_word_counts.append(w_len)
    
posfile.close()


negfile = open('train_neg_full.txt', 'rb')
neg_word_counts = []

for tweet in negfile.readlines():
    tweet = tweet.decode('utf-8')
    w_len = len(tweet.split(' '))
    if w_len > 40:
        print(tweet, w_len)
    neg_word_counts.append(w_len)

negfile.close()


testfile = open('test_data.txt', 'rb')
test_word_counts = []

for tweet in testfile.readlines():
    tweet = tweet.decode('utf-8')
    w_len = len(tweet.split(' '))
    if w_len > 40:
        print(tweet, w_len)
    test_word_counts.append(w_len)

testfile.close()



import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.xlim(0,45)
plt.xlabel('words count')
plt.ylabel('frequency')
g = plt.hist([neg_word_counts, pos_word_counts], bins=100, color=['r','g'], alpha=0.38, label=['positive tweet','negative tweet'])
plt.legend(loc='upper right')
