README:

1. STUDENTS:
- Louis Veillon, mail: b00727589@essec.edu
- Quentin_Boutoille-Blois, mail: quentin.boutoille---blois@student.ecp.fr

2. MODEL SHORT DESCRIPTION:

We used a Recurrent Neural Network with the spacy pre-trained word embdedding vectors from the 'en_core_web_lg' library.
The model consists in the main following steps:
- text preprocessing from the raw data of the csv files. We 'center' the sentences on the target word to get a more precise context than just the whole sentence
- this centering process use a window of 4 words (before and after the target word).
- word embedding loading from the spacy lib vocabulary
- definition of the features of the train samples based on the word embedding
- creation of the 5 layers neural network mainly based on BiDirectional LSTM.
- fiting with some hyperparameters: 15 epochs and a validation split of 0.1
- test preprocessing
- test prediction

3. ACCURACY

Our final accuracy on the devdata is 81.38 with an execution time of roughly 40 sec
