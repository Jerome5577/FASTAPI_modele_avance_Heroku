
from fastapi import FastAPI
from pydantic import BaseModel

from uvicorn import run
import uvicorn
import pickle
#from joblib import load

import contractions
import re

from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# import libraries
#import p7_nlp_preprocessing_local

# =========================================================================================== 
# Remove URL
def remove_urls(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', str(data))
    return data
# Remove USERNAME
def remove_username(data):
    username_pattern = re.compile(r'@\S+')
    data = username_pattern.sub(r'', str(data))
    return data
# Replaciong emojis with their corresponding sentiments
def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' positiveemoji ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negetiveemoji ', tweet)
    return tweet
# +++++++++++++++++++++++++++++++++++++++++++++
# Processing the tweet
def process_tweet_phase1(tweet):
    tweet = remove_username(tweet)                                    # Removes usernames
    tweet = remove_urls(tweet)                                        # Remove URLs
    tweet = emoji(tweet)                                               # Replaces Emojis
    return tweet
# +++++++++++++++++++++++++++++++++++++++++++++
from chat_words_local import chat_words_list, chat_words_map_dict
# Conversion of chat words
def convert_chat_words(data):
    tokens = word_tokenize(str(data))
    new_text = []
    for w in tokens:
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)
# EXPAND CONTRACTIONS
def expend_contractions(data):
    new_text = ""
    for word in str(data).split():
        # using contractions.fix to expand the shortened words
        #expanded_words.append(contractions.fix(word))      
        new_text = new_text + " " + contractions.fix(word)
    return new_text 
# REMOVE MAXIMUM    
def remove_maximum(data):
    data = re.sub(r'[^a-zA-z]', r' ', data)
    data = re.sub(r"\s+", " ", str(data))
    return data
# +++++++++++++++++++++++++++++++++++++++++++++
def process_tweet_phase2(tweet):    
    #tweet = convert_numbers(tweet)    
    tweet = convert_chat_words(tweet)
    tweet = expend_contractions(tweet)                                           
    tweet = tweet.lower()                                             # Lowercases the string
    tweet = re.sub(r"\d+", " ", str(tweet))                           # Removes all digits
    tweet = re.sub('"'," ", str(tweet))                               # Remove (") 
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))                       # Removes all punctuations
    tweet = re.sub(r'(.)\1+', r'\1\1', str(tweet))                    # Convert more than 2 letter repetitions to 2 letter
    tweet = re.sub(r"\s+", " ", str(tweet))                           # Replaces double spaces with single space    
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
    tweet = remove_maximum(tweet)
    return tweet


from tensorflow import keras
from keras.utils import pad_sequences

# load the model
model_file_name = 'best_model_3A_dataset2_length50.h5'
model = keras.models.load_model( model_file_name )

# load the tokenizer
tokenizer_file_name = 'saved_tokenizer_pickle_3A.pkl'
tokenizer = pickle.load(open(tokenizer_file_name,'rb'))

# create the input schema using pydantic basemodel
# Pydantic models are structures that ingest the data, parse it and make sure it conforms 
# to the fields’ constraints defined in it
class Input(BaseModel):
    Tweet : str
    
# create FastAPI instance
app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of a tweet"
            )

# create routes
# home route(/) which will just return “Car Price Predictor” message 
@app.get("/")
def read_root():
    return {"msg":'''WELCOME to the test API for tweet sentiments'''}


def decode_scores(score):
    return 1 if score>0.5 else 0
def decode_proba(score):
    return str(score[0])

# predict route(/predict)
#Input = "i\'m not happy"
@app.post("/predict_tweet")
def predict_tweet(input:Input):
    max_length = 50
    # PRE PROCESSING
    tw1 = process_tweet_phase1(input)
    tw2 = process_tweet_phase2(tw1)
    #tw3 = p7_nlp_preprocessing_local.process_tweet_wordnet_lemmatizer(tw2)
    # SEQUENCE
    tweet_tokenized_sequence = tokenizer.texts_to_sequences([tw2])

    # PADDING
    tweet_seq_pad = pad_sequences(tweet_tokenized_sequence, 
                                    maxlen=max_length, 
                                    padding='post')
    # PREDICTION
    y_pred_test = model.predict(tweet_seq_pad)
    y_pred_target = decode_scores(y_pred_test)
    prob = decode_proba(y_pred_test)

    if y_pred_target == 1:
        output = 'Positif'
    elif y_pred_target == 0 :
        output = 'Negatif'
    else :
        output = 'An error as occured'
     
    return {
        'Prediction': output,
        'Probabilité associée :' : prob
        }

if __name__=="__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

