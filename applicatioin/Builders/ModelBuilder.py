import pandas as pd
import numpy as np

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from gensim.models import Word2Vec

# from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
# from keras.models import Sequential, load_model, model_from_config
# import keras.backend as K


# from sklearn.model_selection import KFold,train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import cohen_kappa_score

# dataset=pd.read_excel("training_set.xlsx")
# X=dataset[['essay_set','essay']]
# X.drop(6973,axis=0,inplace=True)
# Y=dataset['domain1_score']
# Y.dropna(inplace=True)

def essay_to_wordlist(essay_v, remove_stopwords):
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split() # converting to lower case
    if remove_stopwords:
        stops = set(stopwords.words("english")) # removing stopwords
        words = [w for w in words if not w in stops]
    return (words)

def lemmatize(words):
    lemmatizer=WordNetLemmatizer()
    lemmatized_words=[]
    for e in words:
        lemmatized_words.append(lemmatizer.lemmatize(e))
    return lemmatized_words

"""Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
def essay_to_sentences(essay_v, remove_stopwords):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

# """Make Feature Vector from the words list of an Essay."""
# def makeFeatureVec(words, model, num_features):
#     featureVec = np.zeros((num_features,),dtype="float32")
#     num_words = 0.
#     index2word_set = set(model.wv.index2word)
#     for word in words:
#         if word in index2word_set:
#             num_words += 1
#             featureVec = np.add(featureVec,model[word])        
#     featureVec = np.divide(featureVec,num_words)
#     return featureVec

# """Main function to generate the word vectors for word2vec model."""
# def getAvgFeatureVecs(essays, model, num_features):
#     counter = 0
#     essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
#     for essay in essays:
#         essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
#         counter = counter + 1
#     return essayFeatureVecs

# def get_model():
#     """Define the model."""
#     model = Sequential()
#     model.add(LSTM(300, dropout=0.25, recurrent_dropout=0.2, input_shape=[1, 300], return_sequences=True))
#     model.add(LSTM(64, recurrent_dropout=0.2))
#     model.add(Dropout(0.25))
#     model.add(Dense(1, activation='relu'))

#     model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
#     model.summary()

#     return model


# ### Training 

# cv = KFold(len(X), n_folds=5, shuffle=True)
# results = []
# y_pred_list = []

# count = 1
# for traincv, testcv in cv:
#     print("\n--------Fold {}--------\n".format(count))
#     X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]
    
#     train_essays = X_train['essay']
#     test_essays = X_test['essay']
    
#     sentences = []
    
#     for essay in train_essays:
#             # Obtaining all sentences from the training essays.
#             sentences += essay_to_sentences(essay, remove_stopwords = True)
            
#     # Initializing variables for word2vec model.
#     num_features = 300 
#     min_word_count = 40
#     num_workers = 4
#     context = 10
#     downsampling = 1e-3

#     print("Training Word2Vec Model...")
#     model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

#     model.init_sims(replace=True)
#     model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)

#     clean_train_essays = []
    
#     # Generate training and testing data word vectors.
#     for essay_v in train_essays:
#         clean_train_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
#     trainDataVecs = getAvgFeatureVecs(clean_train_essays, model, num_features)
    
#     clean_test_essays = []
#     for essay_v in test_essays:
#         clean_test_essays.append(essay_to_wordlist( essay_v, remove_stopwords=True ))
#     testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )
    
#     trainDataVecs = np.array(trainDataVecs)
#     testDataVecs = np.array(testDataVecs)
#     # Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
#     trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
#     testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
    
#     lstm_model = get_model()
#     lstm_model.fit(trainDataVecs, y_train, batch_size=64, epochs=50)
#     #lstm_model.load_weights('./model_weights/final_lstm.h5')
#     y_pred = lstm_model.predict(testDataVecs)
    
#     # Save any one of the 8 models.
#     if count == 5:
#          lstm_model.save('./model_weights/final_lstm.h5')
    
#     # Round y_pred to the nearest integer.
#     y_pred = np.around(y_pred)
    
#     # Evaluate the model on the evaluation metric. "Quadratic mean averaged Kappa"
#     result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')
#     print("Kappa Score: {}".format(result))
#     results.append(result)

#     count += 1


# X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
# train_essays = X_train['essay']
# test_essays = X_test['essay']

# sentences = []
# for essay in train_essays:
#     # Obtaining all sentences from the training essays.
#     sentences += essay_to_sentences(essay, remove_stopwords = True)

# num_features = 300 
# min_word_count = 40
# num_workers = 4
# context = 10
# downsampling = 1e-3

# print("Training Word2Vec Model.")
# model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)


# clean_train_essays = []
    
# # Generate training and testing data word vectors.
# for essay_v in train_essays:
    
#     clean_train_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
# trainDataVecs = getAvgFeatureVecs(clean_train_essays, model, num_features)
    
# clean_test_essays = []
# for essay_v in test_essays:
    
#     clean_test_essays.append(essay_to_wordlist( essay_v, remove_stopwords=True ))
# testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )

# trainDataVecs = np.array(trainDataVecs)
# testDataVecs = np.array(testDataVecs)
# # Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
# trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
# testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

# lstm_model = get_model()
# lstm_model.fit(trainDataVecs, y_train, batch_size=75, epochs=50)
# y_pred = lstm_model.predict(testDataVecs)

# # Round y_pred to the nearest integer.
# y_pred = np.around(y_pred)
    
# # Evaluate the model on the evaluation metric. "Quadratic mean averaged Kappa"
# result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')
# print("Kappa Score: {}".format(result))