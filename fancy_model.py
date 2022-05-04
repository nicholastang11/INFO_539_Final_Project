from tokenize import TokenInfo
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import csv
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def read_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    df = df[~df.duplicated()] #remove duplicate tweets
    text_len = [] # to find length of each tweet
    for text in df.tweet_text:
        tweet_len = len(text.split())
        text_len.append(tweet_len)
    df['text_len'] = text_len #setting new column in df
    df = df[df['text_len'] > 3] #keeping tweets greater than 3 words
    df = df[df['text_len'] <100] #removing tweets greater than 100 words
    return df

def preprocess_data(df):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    features=df['tweet_text'].values
    for feature in range(len(features)):
        features[feature]=re.sub('@[^\s]+','',features[feature])
        features[feature]=re.sub('http[^\s]+','',features[feature])
        features[feature]=re.sub('rt  [^\s]+','',features[feature])
        features[feature]=re.sub('rt [^\s]+','',features[feature])
        features[feature]=re.sub('____[^\s]+','',features[feature])
        features[feature]=re.sub('pictwitter[^\s]+','',features[feature])
        features[feature]=re.sub('pictwittercomm[^\s]+','',features[feature])
        features[feature]=features[feature].lower()
        features[feature]=re.sub('[^0-9a-z #+_]','',features[feature])
        features[feature]=re.sub(r'[0-9]','',features[feature])

    data_list_s = [] #list containing lemmatized words
    for words in features:
        words = word_tokenize(words)
        words_s = ''
        for w in words:
            if len(w)>3 and len(w)<14: #keeping words >3 and <14 characters long
                if w not in stop_words:
                    w_s = lemmatizer.lemmatize(w)
                    words_s+=w_s+' '
        data_list_s.append(words_s)

    df['text_clean'] = data_list_s #adding new column with cleaned text
    df.drop_duplicates("text_clean", inplace=True) #removing anymore duplicates that may have come up during preprocessing
    new_text_len = [] #creating column for cleaned text length
    for text in df.text_clean:
        tweet_len = len(text.split())
        new_text_len.append(tweet_len)
    #print(new_text_len)
    df['new_text_len'] = new_text_len #setting new column
    df = df[df['new_text_len'] > 3] #keeping tweets greater than 3 words
    df = df[df['new_text_len'] <100] #removing tweets greater than 100 words
    longest_tweet_len =len(max(data_list_s))
    return df, longest_tweet_len

def get_class_counts(df):
    return df.cyberbullying_type.value_counts()

def tokenize(df,sequence_len):
    print(f"longest tweet len {sequence_len}")
    tokenizer = Tokenizer(num_words=4000, filters='!"$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    #tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df['text_clean'])
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    X = tokenizer.texts_to_sequences(df['text_clean'])
    X = pad_sequences(X, maxlen=sequence_len) #maxlen = longest tweet length
    Y = pd.get_dummies(df['cyberbullying_type']).values
    return X,Y
def train_test_split_data(X,Y):
    X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,stratify=y_train, random_state=1)
    X_train_shape,y_train_shape = X_train.shape, y_train.shape
    X_dev_test_shape,y_dev_test_shape=X_val.shape, y_val.shape
    write_files=False
    if write_files == True: #will rewrite train/dev/test files to cwd if any paramaters are tuned
        a_file = open("X_train.txt", "w")
        for row in X_train:
            np.savetxt(a_file, row)
        a_file.close()

        a_file = open("X_test.txt", "w")
        for row in X_test:
            np.savetxt(a_file, row)
        a_file.close()

        a_file = open("X_val.txt", "w")
        for row in X_val:
            np.savetxt(a_file, row)
        a_file.close()

        a_file = open("y_val.txt", "w")
        for row in y_val:
            np.savetxt(a_file, row)
        a_file.close()

        a_file = open("y_train.txt", "w")
        for row in y_train:
            np.savetxt(a_file, row)
        a_file.close()

        a_file = open("y_test.txt", "w")
        for row in y_test:
            np.savetxt(a_file, row)
        a_file.close()

        X_train = np.loadtxt("X_train.txt").reshape(X_train_shape)
        y_train = np.loadtxt("y_train.txt").reshape(y_train_shape)
        #X_val = np.loadtxt("X_val.txt").reshape(X_dev_test_shape) #test set is same shape
        #y_val = np.loadtxt("y_val.txt").reshape(y_dev_test_shape)
        #X_test = np.loadtxt("X_test.txt").reshape(X_dev_test_shape) #test set is same shape
        #y_test = np.loadtxt("y_test.txt").reshape(y_dev_test_shape)
    return X_train,y_train

def create_model(X,X_train,y_train):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(4000, 100, input_length=X.shape[1],mask_zero=True),
    tf.keras.layers.LSTM(75,return_sequences=True), #75 did the best
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30)),
    tf.keras.layers.Dense(6, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3, batch_size=35,validation_split=0.1)
    model.save("C:\\Users\\nickj\\OneDrive\\Documents\\INFO539\\INFO_539_Final_Project")


def main():
    df = read_data("./cyberbullying_tweets.csv")
    df,longest_tweet_len = preprocess_data(df)
    print(get_class_counts(df))
    X,Y=tokenize(df,longest_tweet_len)
    print(X)
    X_train,y_train=train_test_split_data(X,Y)
    model=create_model(X,X_train,y_train)
    print("model created and saved")

if __name__ =="__main__":
    main()