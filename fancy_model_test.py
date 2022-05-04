#test model
from tokenize import TokenInfo
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
model_dir="C:\\Users\\nickj\\OneDrive\\Documents\\INFO539\\INFO_539_Final_Project"


df = pd.read_csv("./cleaned_data.csv")


tokenizer = Tokenizer(num_words=4000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text_clean'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(df['text_clean'].values)
my_model=tf.keras.models.load_model(model_dir)
X_val = np.loadtxt("X_val.txt").reshape(7965, 39) #test set is same shape
y_val = np.loadtxt("y_val.txt").reshape(7965, 6)
score = my_model.evaluate(X_val, y_val, verbose=0)
print("%s: %.2f%%" % (my_model.metrics_names[1], score[1]*100))
print(my_model.summary())


print("*************************")
guesses= np.argmax(my_model.predict(X_val), axis=-1)

#print(guesses)

print(np.unique(guesses))
classes=['age',  'ethnicity',  'gender',  'not_cyberbullying','other_cyberbullying', 'religion']

Xval=tokenizer.sequences_to_texts(X_val)
for row in range(2000):
    print()
    val=guesses[row]
    print(Xval[row]+"\nPredicted index:", guesses[row],"\nPredicted class:",classes[val],"\nActual Class:",y_val[row])
    print()
print("%s: %.2f%%" % (my_model.metrics_names[1], score[1]*100))

print("*************************")


guesses= np.argmax(my_model.predict(X_val), axis=-1)
#print(guesses)
print(np.unique(guesses))
cm=confusion_matrix(np.argmax(y_val,axis=-1), guesses)
print(cm)
print(classification_report(np.argmax(y_val,axis=-1), guesses,target_names=classes))

ax = sns.heatmap(cm, annot=True, cmap='Blues',fmt='g')

ax.set_title('Model B Dev Set Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Cyberbullying Category')
ax.set_ylabel('Actual Cyberbullying Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Age',  'Ethnicity',  'Gender',  'None','Other', 'Religion'],fontsize=9)
ax.yaxis.set_ticklabels(['Age',  'Ethnicity',  'Gender',  'None','Other', 'Religion'],fontsize=9)

## Display the visualization of the Confusion Matrix.
plt.savefig('C:\\Users\\nickj\\OneDrive\\Documents\\INFO539\\INFO_539_Final_Project\\Model_B_dev.png',bbox_inches='tight')
print("saved")
plt.show()