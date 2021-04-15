#!/usr/bin/env python
# coding: utf-8

# # Creating a chatbot using tensorflow and tflearn

# ## transform conversational intent definitions to Tensorflow model

# In[10]:


# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random


# ### Importing intents file

# In[11]:


import json
with open('intents.json') as intents_data:
    intents = json.load(intents_data)


# 
# 
# ### Organizing words, documents and classification classes

# In[12]:




nltk.download('punkt')


# In[13]:


words = []
classes = []
documents = []
ignore_words = ['?']

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)

        # add to our words list
        words.extend(w)

        # add to documents in our corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)


# ### Stemming

# The stem `tak` will match `take`, `taking`, `takers`, etc. We could clean the words list and remove useless entries but this will suffice for now.
# 

# 
# 
# 
# 
# 
# 
# 
# This data structure won’t work with Tensorflow, we need to transform it further: *from documents of words into tensors of numbers.*

# In[14]:


# create our training data
training = []
output = []

# create empty array for our output
output_array = [0] * len(classes)

# training set, bag of words for each sentence
for document in documents:
    # init bag of words
    bag = []

    # list of tokenized words for the pattern
    pattern_words = document[0]

    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    # create bog of words array
    for word in words:
        bag.append(1 if word in pattern_words else 0)

    # output is 0 for each tag and 1 for current tag
    output_row = list(output_array)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = (np.array(training))

# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])


# In[15]:


from tensorflow.python.framework import ops
ops.reset_default_graph()


# In[16]:


# reset underlying graph data
ops.reset_default_graph()

# build a neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


"""""input = keras.input(shape=[len(intent_train_x[0])]) 
intent_net = input
intent_net = layers.Dense(8)(intent_net) 
intent_net = layers.Dense(8)(intent_net) 
intent_net = layers.Dense(len(intent_train_y[0], activation='softmax')(intent_net) 
intent_model = keras.Model(inputs=[input], outputs=[output])
intent_model.load('model1')"""""


# ### Saving data structures using `Pickle`

# In[17]:


# save all of our data structures
import pickle
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x,
             'train_y': train_y}, open("training_data", "wb"))


# 
# 
# ### Doing some testing on the model

# In[18]:


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


# In[19]:


'''p = bow("is your shop open today?", words)
print (p)
print (classes)'''


# In[20]:


p1 = bow("what is EV charging station", words)
print (p1)
print (classes)


# In[21]:


print(model.predict([p1]))


# Intent that is closest to our sentence:

# In[22]:


def get_predicted_intent(predictions):
    return classes[np.argmax(predictions)]

print(get_predicted_intent(model.predict([p1])))


# In[ ]:




