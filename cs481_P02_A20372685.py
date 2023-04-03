# %%
import sys
import pandas as pd
import numpy as np
import nltk
import re
from math import log
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



# %%
lowercase = None
if len(sys.argv) == 2 and sys.argv[1].upper() == 'YES':
    ignore_step = 'lowercase'
else:
    ignore_step = 'None'

# %%
print("Vera, Jose, A20372685 solution:")
print(f"Ignored pre-processing step: {ignore_step.upper()}")


# %%
fake_news = pd.read_csv('Fake.csv')
real_news = pd.read_csv('True.csv')

# %%
fake_news['class'] = 0  
real_news['class'] = 1  

# Create a list of DataFrames to concatenate
dfs = [fake_news[['title', 'text', 'class']], real_news[['title', 'text', 'class']]]
data = pd.concat(dfs, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True) ##shuffle data
data['text'] = data['title'] + ' ' + data['text']  ## Combine title and text

# %%
print("Pre-processing text...")
ps = PorterStemmer() #Stemmer that will be used for stemming
stop_words = set(stopwords.words('english'))
##If argument YES given, ingore_step will be 'lowercase', and lowercasing step will be skipped.
if ignore_step != 'lowercase':
    data['text'] = data['text'].apply(lambda x: x.lower())  # Lowercase
##Remove Stopwords
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))  # Remove stop words
##Perfrom Stemming
data['text'] = list(map(lambda x: ' '.join(ps.stem(word) for word in x.split()), data['text']))
##Remove non alphatical characters
data['text'] = data['text'].apply(lambda text: re.sub(r'[^a-zA-Z\s]', ' ', text))

# %%
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.2, random_state=42)

# %%
def train_naive_bayes(X_train):
    ##Binary count vectorizer object
    vectorizer = CountVectorizer(binary=True)

    vectorizer.fit(X_train)

    X_train_bow_matrix = vectorizer.transform(X_train).toarray()
    ##Separate BOW into different matrices
    
    X_fake = X_train_bow_matrix[y_train == 0, :]
    X_real = X_train_bow_matrix[y_train == 1, :]

    log_prior = {}

    # Calculate P(c) term
    numb_doc = len(X_train_bow_matrix)
    numb_classes = 2
    class_counts = np.bincount(y_train)
    for label in range(numb_classes):
        log_prior[label] = np.log(class_counts[label]/numb_doc) 

    # Create Vocabulary of D
    V = vectorizer.get_feature_names_out()
    ##Get necessary counts to calcualte probability
    real_word_counts = np.sum(X_real, axis=0)
    fake_word_counts = np.sum(X_fake, axis=0)
    real_words_total = np.sum(real_word_counts)
    fake_words_total = np.sum(fake_word_counts)
    real_doc_count = len(X_real)
    fake_doc_count = len(X_fake)

    #Calculate probabilites using lapace smoothing of 1
    fake_probs = {}
    real_probs = {}
    for word in range(len(V)):
        fake_count = fake_word_counts[word]
        real_count = real_word_counts[word]
        fake_probs[V[word]] = np.log((fake_count + 1) / (fake_words_total + len(V)))
        real_probs[V[word]] = np.log((real_count + 1) / (real_words_total + len(V)))
    # Create log_likelihood dictionary
    log_likelihood = {}
    log_likelihood[0] = fake_probs
    log_likelihood[1] = real_probs

    V_list = V.tolist()
    
    return log_prior,log_likelihood,V_list,




# %%
def test_naive_bayes(X_test, log_prior, log_likelihood, C, V):
    
    vectorizer = CountVectorizer(vocabulary=V, binary=True)
    testdoc = vectorizer.transform(X_test).toarray()

    # Create a matrix of log likelihoods for all words in the vocabulary for each class
    log_likelihood_matrix = np.array([list(log_likelihood[c].values()) for c in C]).T

    # Calculate the sum of log likelihoods for each document and class using broadcasting
    sum_c = (testdoc @ log_likelihood_matrix) + list(log_prior.values())

    # Choose the class with the highest sum
    best_c = np.argmax(sum_c, axis=1)

    return best_c, sum_c


# %%
print('Training classifier…')

# %%
log_prior, log_likelihood, V = train_naive_bayes(X_train)

# %%
print('Testing classifier…')
y_pred, sum_c = test_naive_bayes(X_test, log_prior, log_likelihood, [0,1], V)

# %%
print("Test results / metrics:\n")

# %%
## create confusion matrix and calculate metrics
conf_mat = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf_mat.ravel()

sensitivity = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
precision = precision_score(y_test, y_pred)
npv = tn / (tn + fn)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("Number of true positives:", tp)
print("Number of true negatives:", tn)
print("Number of false positives:", fp)
print("Number of false negatives:", fn)
print("Sensitivity (recall):", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)
print("Negative predictive value:", npv)
print("Accuracy:", accuracy)
print("F-score:", f1)


# %%
##Ask user for input sentence then apply classifier
while True:
    sentence = input("Enter your sentence: ")
    ##If argument YES given, ingore_step will be 'lowercase', and lowercasing step will be skipped.
    if ignore_step != 'lowercase':
        sentence = sentence.lower()
    #Remove stop words and rejoin the remaining words back into a string
    filtered_sentence = ' '.join([word for word in sentence.split() if word.lower() not in stop_words]) 
    ##Perfrom Stemming
    text = ' '.join([ps.stem(word) for word in filtered_sentence.split()])
    text = [re.sub(r'[^a-zA-Z\s]', ' ', text)]

    class_label, class_probabilities = test_naive_bayes(text,log_prior, log_likelihood, [0,1], V)
    if class_label[0] == 0:
        other_label = 1
        class_name = 'Fake'
        other_name = 'Real'
    else:
        other_label = 0
        class_name = 'Real'
        other_name = 'Fake'

    print(f"Sentence '{sentence}' was classified as '{class_name}'.")
    print(f"P({class_name} | S) = {class_probabilities[0][class_label[0]]:.4f}")
    print(f"P({other_name} | S) = {class_probabilities[0][other_label]:.4f}")
    
    answer = input("Do you want to enter another sentence [Y/N]? ")
    if answer.lower() != 'y':
        break


# %%



