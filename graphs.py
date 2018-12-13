def zipfslaw(tweets):
    words = []
    for t in tweets:
        ws = t.split()
        for w in ws:
            words.append(w.lower())

    # plot word frequency distribution of first few words
    plt.figure(figsize=(12,5))
    plt.title('Top 25 most common words')
    plt.xticks(fontsize=13, rotation=90)
    fd = nltk.FreqDist(words)
    fd.plot(25,cumulative=False)

    # log-log plot
    word_counts = sorted(Counter(words).values(), reverse=True)
    plt.figure(figsize=(12,5))
    plt.loglog(word_counts, linestyle='-', linewidth=1.5)
    plt.ylabel("Freq")
    plt.xlabel("Word Rank")
    plt.title('log-log plot of words frequency')


def bow(all_words):
    # create a word frequency dictionary
    wordfreq = Counter(all_words)
    # draw a Word Cloud with word frequencies
    wordcloud = WordCloud(width=900,
                        height=500,
                        max_words=500,
                        max_font_size=100,
                        relative_scaling=0.5,
                        colormap='Blues',
                        normalize_plurals=True).generate_from_frequencies(wordfreq)
    plt.figure(figsize=(17,14))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()