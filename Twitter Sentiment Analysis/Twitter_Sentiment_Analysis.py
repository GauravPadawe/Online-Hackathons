#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement
# 
# - The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.
# 
# - Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.
# 
# ### Motivation
# 
# - Hate  speech  is  an  unfortunately  common  occurrence  on  the  Internet.  Often social media sites like Facebook and Twitter face the problem of identifying and censoring  problematic  posts  while weighing the right to freedom of speech. The  importance  of  detecting  and  moderating hate  speech  is  evident  from  the  strong  connection between hate speech and actual hate crimes. Early identification of users promoting  hate  speech  could  enable  outreach  programs that attempt to prevent an escalation from speech to action. Sites such as Twitter and Facebook have been seeking  to  actively  combat  hate  speech. In spite of these reasons, NLP research on hate speech has been very limited, primarily due to the lack of a general definition of hate speech, an analysis of its demographic influences, and an investigation of the most effective features.
# 
# ### Data
# 
# - Our overall collection of tweets was split in the ratio of 65:35 into training and testing data. Out of the testing data, 30% is public and the rest is private.
# 
# - Data Files :
#     - train.csv - For training the models, we provide a labelled dataset of 31,962 tweets. The dataset is provided in the form of a csv file with each line storing a tweet id, its label and the tweet.
#     - There is 1 test file (public)
#     - test_tweets.csv - The test data file contains only tweet ids and the tweet text with each tweet in a new line.
#  
# 
# ### Submission Details
# 
# ##### The following 3 files are to be uploaded.
# 
# - test_predictions.csv - This should contain the 0/1 label for the tweets in test_tweets.csv, in the same order corresponding to the tweets in test_tweets.csv. Each 0/1 label should be in a new line.
#  
# 
# - A .zip file of source code - The code should produce the output file submitted and must be properly commented.
#  
# 
# ### Evaluation Metric:
# 
# - The metric used for evaluating the performance of classification model would be F1-Score.
# 
# - The metric can be understood as :
# 
#     - True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.
#     - True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.
#     - False Positives (FP) – When actual class is no and predicted class is yes.
#     - False Negatives (FN) – When actual class is yes but predicted class in no.
#     - Precision = TP/TP+FP
#     - Recall = TP/TP+FN 
# 
# - F1 Score = 2*(Recall * Precision) / (Recall + Precision)
# 
# - F1 is usually more useful than accuracy, especially if for an uneven class distribution.

# In[1]:


#importing Required packages
#regex, numpy, pandas , tensorflow, nltk, etc
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import string
import nltk
import warnings
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[2]:


#setting a path
path = 'D:/Data Science/DS Prac/Datasets/PRACTICE DATASETS/Twitter Sentiment Analysis/'


# In[3]:


#reading training data
train = pd.read_csv(path+'train.csv', header=0, encoding='utf-8')
train.head()


# In[5]:


#reading training data
test = pd.read_csv(path+'test.csv', header=0, encoding='utf-8')
test.head()


# ### Data Cleaning

# In[6]:


#eliminating unwanted characters with the help of regex
train['tweet_clean'] = train['tweet'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
test['tweet_clean'] = test['tweet'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)


# In[7]:


#stop words removal

stop_words = set(stopwords.words('english'))
#print (stop_words)

train_words = np.array([w for w in train['tweet_clean'] if not w in stop_words])
#train_words

test_words = np.array([w for w in test['tweet_clean'] if not w in stop_words])
#test_words


# In[8]:


test_words = np.array([x for x in test['tweet_clean'] if not isinstance(x, int)])
test_words

train_words = np.array([x for x in train['tweet_clean'] if not isinstance(x, int)])
train_words


# In[9]:


#punctuation remover
table = str.maketrans('','', string.punctuation)

stripped_train = np.array([w.translate(table) for w in train_words])
#stripped_train = [w.translate(table) for w in train_words]
#stripped_train = train_words.str.replace("[^a-zA-Z#]", " ")
#stripped_train

stripped_test = np.array([w.translate(table) for w in test_words])
#stripped_test = test['tweet'].str.replace("[^a-zA-Z#]", " ")
stripped_train.shape


# In[10]:


#short words removal

'''new_train = []
new_test = []

shortword = re.compile(r'\W*\b\w{1,3}\b')

for i in stripped_train:
    line_train = shortword.sub('', i)
    new_train.append(line_train)
    
for j in stripped_test:
    line_test = shortword.sub('', j)
    new_test.append(line_test)'''


# In[11]:


#new_train = np.array(new_train)
#new_test = np.array(new_test)


# In[13]:


stripped_train


# In[15]:


'''def duplicate_removal(data):
    temp = []
    new_list =[]
    new_str = []
    
    for i in data:
        temp.append(i.split(' '))
    
    for j in temp:
        mylist = list(dict.fromkeys(j))
        new_list.append(mylist)
    for k in new_list:
        join_str = ' '.join(k)
        new_str.append(join_str)
    
    return new_str
    

x = duplicate_removal(stripped_train)

y = duplicate_removal(stripped_test)'''


# ### Removal of Emojis and Binary text

# - After a fair bit of Research I was able to finf out user-defined function to eliminate emojis, symbols, etc, if any. 
# 
# 
# - Source : https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

# In[18]:


#emoji, symbols elimination

def remove_emoji(string):
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f""]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# In[19]:


#applying function on data

train_emoji_rem = []
for i in x:
    train_emoji_rem.append(remove_emoji(i))
  
  
test_emoji_rem = []
for i in y:
    test_emoji_rem.append(remove_emoji(i))


# In[20]:


'''new_list=[]
for word in emoji_rem:
    if word.encode('utf-8').decode('ascii','ignore') !='':
        new_list.append(word)'''


# In[28]:


'''import unicodedata
from unidecode import unidecode

def deEmojify(inputString):
  returnString = ""
  for character in inputString:
    try:
      character.encode("ascii")
      returnString += character
    except UnicodeEncodeError:
      returnString += ''
    return returnString'''


# In[29]:


#tokenize the data and fit on data
token = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
token.fit_on_texts(train_emoji_rem)


# In[30]:


#text to matrix using tfidf
text_to_mat = token.texts_to_matrix(train_emoji_rem, mode='tfidf')
text_to_mat.shape


# In[31]:


#token.word_index


# In[32]:


#from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Reshape, Flatten, LSTM, GRU, SimpleRNN

#xtrain, xtest, ytrain, ytest = train_test_split(text_to_mat, train['label'], test_size=0.4, random_state=128)


# In[33]:


#early stopping, Learning Rate optimization and model checkpoint implementation

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_mod.h5',
                                                      monitor='val_loss',
                                                      save_best_only=True)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=10
                                                 )

'''LR_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.1, 
                                                    patience=10,
                                                    mode='auto', 
                                                    cooldown=0,
                                                    min_lr=0
                                                   )'''


# In[40]:


#Multi-layred ANN 

model = tf.keras.models.Sequential()

model.add(BatchNormalization(input_shape=(5000,), axis=1))

#model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(200, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(50, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(25, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))


#opt = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#sgd = tf.keras.optimizers.SGD(lr=0.03, decay=1e-6, momentum=0.8, nesterov=True)
#rms = tf.keras.optimizers.RMSprop(lr=0.3, rho=0.5, epsilon=0, decay=0.9)

model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])


# In[175]:


#fit on data

model.fit(text_to_mat, train['label'],
          validation_split=0.47,
          epochs=100,
          batch_size=128,
          shuffle=True, 
          callbacks=[model_checkpoint, early_stopping, 
                     #LR_reduction
                    ]
         )


# In[176]:


#tokenizing test data
token_test = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
token_test.fit_on_texts(test_emoji_rem)

#text to matrix
test_text_to_mat = token.texts_to_matrix(test_emoji_rem, mode='tfidf')
test_text_to_mat.shape


# In[177]:


#prediction on test data

pred = model.predict(test_text_to_mat)
pred


# In[178]:


#probability rounding off
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
print (pred)


# In[179]:


#submission file
final = pd.DataFrame({'id':test['id'].values, 'label':pred.flatten().astype('int')})
final = final.set_index('id')
final.head()


# In[180]:


#wrining dataframe to csv
final.to_csv('C:/Users/Gaurav/Desktop/sub_464.csv')

