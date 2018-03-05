# Abstract based sentiment classifier with LSTM. 

A Classifier to predict aspect-based polarities of opinions in sentences, that assigns a polarity label to every triple &lt;aspect categories, aspect_term, sentence>. The polarity labels are positive, negative and neutral. Note that a sentence may have several opinions.

# Model short description:

I used a Recurrent Neural Network with the spacy pre-trained word embdedding vectors from the 'en_core_web_lg' library.

The model consists in the main following steps:

- text preprocessing from the raw data of the csv files. We 'center' the sentences on the target word to get a more precise context than just the whole sentence

- this centering process use a window of 4 words (before and after the target word).

- word embedding loading from the spacy lib vocabulary

- definition of the features of the train samples based on the word embedding

- creation of the 5 layers neural network mainly based on BiDirectional LSTM.

- fiting with some hyperparameters: 15 epochs and a validation split of 0.1

- test preprocessing

- test prediction


# Accuracy

The final accuracy on the devdata is 81.38 with an execution time of roughly 40 sec# ABSA_sentiment_classifier
