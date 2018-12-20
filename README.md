# Tweets Sentiment Classification - PROJECT 2

##################################
## AUTHORS: Riccardo Succa, Costanza Volpini, Marco Zoveralli
##################################

### DESCRIPTION:
The aim of this project is to improve the accuracy of sentiment classification of tweets.

### RESULTS:
By using cross validation the best solution that we got is a model that is composed by a convolutional layer, followed by a max pooling, an LSTM and 3 fully connected layers. This led to an accuracy of 87.4%. We used 2-grams as embedding, that was trained along with the rest of the model.

### STRUCTURE:
    - corpus/ : folder containing the dictionaries used in order to fix the grammar typos in the tweets
    - embedding_hashtag_split/ : folder containing the saved embeddings and model
    - train_pos_full.txt : text file containing the positive tweets of the whole train dataset
    - train_neg_full.txt : text file containing the negative tweets of the whole train dataset
    - test_data.txt: text file containing all the tweets of the test dataset
    - words-by-frequency.txt : text file containing the words sorted by frequency. Used in order to separate concatenated words.

### PYTHON SCRIPTS:
    - feature_extraction.py : functions to extract the feature representation of the tweets and the embedding matrix. It is also possible to save them on the hard disk. It uses the train_pos_full.txt and train_neg_full.txt files. It generates the embedding saved in the embedding_hashtag_split folder
    - models_full_pretrained : functions to generate the models that we tried during this project. The model discussed in our report (and that gave us the claimed accuracy) is model3
    - ngram.py : functions to generate and add n-grams
    - plot.py : script to generate plot in order to see the frequency of words in the training dataset
    - preprocessing : functions to process the tweets before extracting the features and generating the embeddings. It uses the files in the corpus folder
    - run.py : script to generate the model and the csv file that we used in order to obtain the claimed score. The csv is generated in this folder, the model is saved in the embedding_hashtag_split folder
    - split_hashtags.py : function to split the hashtag composed by multiple concatenated words. It uses the words-by-frequency.txt file

### TO RUN THE CODE:
    1. Install the requirements: pip install -r requirements.txt
    2. python3 run.py (in this folder)
