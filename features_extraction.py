# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:11:14 2021

@author: IT Doctor
"""

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import re 
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def generate_embeddings(dataframe):
    ## load hub module
    embed_type = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    dataset_embeddings = []
    for index, row in dataframe[dataframe.columns[:-6]].iterrows():
        session_embed = []
        for text in row.to_list():
          if (text is not None) and (text != "empety"):
            emb = embed_type(tf.constant([text], dtype=tf.string)).numpy()[0]
            session_embed.append(emb)
          else:
            session_embed.append(np.zeros(512))
        dataset_embeddings.append(session_embed)
    return np.asarray(dataset_embeddings)

def generate_vine_embeddings(dataframe):
    ## load hub module
    embed_type = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    dataset_embeddings = []
    for index, row in dataframe[dataframe.columns[:-5]].iterrows():
        session_embed = []
        for text in row.to_list():
          if (text is not None) and (text != "empty"):
            emb = embed_type(tf.constant([text], dtype=tf.string)).numpy()[0]
            session_embed.append(emb)
          else:
            session_embed.append(np.zeros(512))
        dataset_embeddings.append(session_embed)
    return np.asarray(dataset_embeddings)

def get_publication_time_list(df):
  publication_time_list = []
  for index, row in df.iterrows():
    time_list_per_session = []
    for r in row[:-6]:
      # print(re.search('((.*))', r).group(1))
      if r == "empety":
        time_list_per_session.append(0)
      else:
        time_list_per_session.append(r[-18:-1])
    publication_time_list.append(time_list_per_session)
  return publication_time_list
  
def convert_string_to_datetime(string_list):
  datetime_list = []
  for string_datetime in string_list:
    if string_datetime == 0:
      datetime_list.append(0)
    else:
      date_time_obj = datetime.strptime(string_datetime, '%y-%m-%d %H:%M:%S')
      datetime_list.append(date_time_obj)
  return datetime_list

def convert_vine_string_to_datetime(string_list):
  datetime_list = []
  for string_datetime in string_list:
    if string_datetime == 0:
      datetime_list.append(0)
    else:
      date_time_obj = datetime.strptime(string_datetime, '%Y-%m-%d %H:%M:%S')
      datetime_list.append(date_time_obj)
  return datetime_list
  
def convert_publication_time_to_datetime(publication_time_list):
  converted_list = []
  cpt = 0
  for list_elem in publication_time_list:
    datetime_per_list = []
    for elem in list_elem:
      try:
        date_time_obj = datetime.strptime(elem, '%y-%m-%d %H:%M:%S')
        datetime_per_list.append(date_time_obj)
        cpt += 1 
      except (TypeError, ValueError):
        datetime_per_list.append(0)
    converted_list.append(datetime_per_list)
  return converted_list

def convert_vine_publication_time_to_datetime(publication_time_list):
  converted_list = []
  cpt = 0
  for list_elem in publication_time_list:
    datetime_per_list = []
    for elem in list_elem:
      try:
        date_time_obj = datetime.strptime(elem, '%Y-%m-%d %H:%M:%S')
        datetime_per_list.append(date_time_obj)
        cpt += 1 
      except (TypeError, ValueError):
        datetime_per_list.append(0)
    converted_list.append(datetime_per_list)
  return converted_list

def get_vine_publication_time_list(df):
  publication_time_list = []
  for index, row in df.iterrows():
    time_list_per_session = []
    for r in row[:-5]:
      try:
        if r == "empety" or r =='empty':
          time_list_per_session.append(0)
        elif r.startswith('Media'):
          time_list_per_session.append(re.search('Media posted at:(.*)\.', re.sub('T',' ', r)).group(1))
        else:
          time_list_per_session.append(re.search('\(created_at:(.*)\.', re.sub('T',' ', r)).group(1))
      except AttributeError:
        time_list_per_session.append(0)
    publication_time_list.append(time_list_per_session)
  return publication_time_list


def extract_time_features(dataframe):    
    publication_time_list = get_publication_time_list(dataframe)
    
    cptn_time_list = dataframe['cptn_time'].tolist()
    
    for index in range(len(cptn_time_list)):
      if cptn_time_list[index].startswith('<font'):
        cptn_time_list[index] = cptn_time_list[index][-18:-1]
      elif cptn_time_list[index].startswith('Media'):
        # cptn_time_list[index] = cptn_time_list[index][18:-2]
        cptn_time_list[index] = re.search('0(.*)', cptn_time_list[index]).group(1)[:-1]
      elif cptn_time_list[index].startswith('empety'):
        cptn_time_list[index] = 0
        
    cptn_datetime_list = convert_string_to_datetime(cptn_time_list)
    publication_datetime_list = convert_publication_time_to_datetime(publication_time_list)
    
    subtract_time_list = []
    for x, y in zip(publication_datetime_list, cptn_datetime_list):
      subtract_time_list.append([(datetime_elem - y).total_seconds() if datetime_elem !=0 else 0 for datetime_elem in x])

    time_features = np.array(subtract_time_list)
    
    ### Scaling features
    scaler = MinMaxScaler()
    scaler.fit(time_features)
    scaled_time_features = scaler.transform(time_features)
    
    return scaled_time_features

def extract_vine_time_features(dataframe):
    cptn_time_list = dataframe['cptn_time'].tolist()
    
    publication_time_list = get_vine_publication_time_list(dataframe)
    
    for index in range(len(cptn_time_list)):
      if cptn_time_list[index].startswith('Media'):
        cptn_time_list[index] = re.search('Media posted at:(.*)\.', re.sub('T', ' ', cptn_time_list[index])).group(1)
      else: #cptn_time_list[index].startswith('empety'):
        list_index = 0
        while publication_time_list[index][list_index] == 0:
          list_index += 1
        date_time_obj = datetime.strptime(publication_time_list[index][list_index][-8:], '%H:%M:%S')
        date_time_obj = date_time_obj + timedelta(minutes=list_index)
        date_time_obj = date_time_obj.strftime('%H:%M:%S')
        # print(publication_time_list[index][list_index][-8:], date_time_obj)
        cptn_time_list[index] =  publication_time_list[index][list_index][:-8] + date_time_obj  
        
    cptn_datetime_list = convert_vine_string_to_datetime(cptn_time_list)
    
    publication_datetime_list = convert_vine_publication_time_to_datetime(publication_time_list)
    
    subtract_time_list = []
    for x, y in zip(publication_datetime_list, cptn_datetime_list):
      try:
        subtract_time_list.append([(datetime_elem - y).total_seconds() if datetime_elem !=0 else 0 for datetime_elem in x])
      except TypeError:
        print(type(y))
    time_features = np.array(subtract_time_list)
    scaler = MinMaxScaler()
    
    scaler.fit(time_features)
    scaled_time_features = scaler.transform(time_features)
        
    return scaled_time_features


def extract_likes(dataframe):
    likes_features = []
    for elem in dataframe['likes'].tolist():
      txt = [int(s) for s in elem.split() if s.isdigit()]
      try:
        likes_features.append(txt[0])
      except IndexError:
        likes_features.append(0)   
    likes_features = np.asarray(likes_features).reshape(-1,1)
    
    ### Scaling features
    scaler = MinMaxScaler()
    scaler.fit(likes_features)
    scaled_likes_features = scaler.transform(likes_features)
    
    return scaled_likes_features

def extract_vine_likes(dataframe):
    likes_features = np.asarray(dataframe['likes']).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(likes_features)
    scaled_likes_features = scaler.transform(likes_features)
    return scaled_likes_features
    

def extract_sentiment_features(dataframe):
    def strip_creation_time(text):
      if text == 'empety':
        return ''
      else:
        # print(text)
        try:
          string_to_strip = re.search('(created_at:(.*))', text).group(0)
          stripped_text = text.replace(string_to_strip, '')[:-1]
        except AttributeError:
          try:
            string_to_strip = re.search('(created at:(.*))', text).group(0)
            stripped_text = text.replace(string_to_strip, '')[:-1]
          except AttributeError:
            stripped_text = text
      return stripped_text
    
    sentiment_features = []
    analyzer = SentimentIntensityAnalyzer()
    
    df_stripped_time = dataframe[dataframe.columns[:-6]].applymap(strip_creation_time)
    for index, row in df_stripped_time.iterrows():
      all_sentiments_list = []
      for cmnt in row:
        sentiment_list = []
        blob_cmnt = TextBlob(cmnt)
        sentiment_list.append(blob_cmnt.sentiment.polarity)
        sentiment_list.append(blob_cmnt.sentiment.subjectivity)
    
        vader_sentiments = analyzer.polarity_scores(cmnt)
        sentiment_list.append(vader_sentiments['compound'])
        sentiment_list.append(vader_sentiments['pos'])
        sentiment_list.append(vader_sentiments['neu'])
        sentiment_list.append(vader_sentiments['neg'])
    
        all_sentiments_list.append(sentiment_list)
      sentiment_features.append(all_sentiments_list)
      
    sentiment_features = np.asarray(sentiment_features)
    return sentiment_features

def extract_vine_sentiment_features(dataframe):
    sentiment_features = []
    analyzer = SentimentIntensityAnalyzer()
    for index, row in dataframe[dataframe.columns[:-5]].iterrows():
      all_sentiments_list = []
      for cmnt in row:
        sentiment_list = []
        blob_cmnt = TextBlob(cmnt)
        sentiment_list.append(blob_cmnt.sentiment.polarity)
        sentiment_list.append(blob_cmnt.sentiment.subjectivity)
    
        vader_sentiments = analyzer.polarity_scores(cmnt)
        sentiment_list.append(vader_sentiments['compound'])
        sentiment_list.append(vader_sentiments['pos'])
        sentiment_list.append(vader_sentiments['neu'])
        sentiment_list.append(vader_sentiments['neg'])
    
        all_sentiments_list.append(sentiment_list)
      sentiment_features.append(all_sentiments_list)
      
    sentiment_features = np.asarray(sentiment_features)
    return sentiment_features

def extract_owner_cmnt_embedding(dataframe):
    owner_cmnt_embeddings = []
    embed_type = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    for text in dataframe['owner_cmnt'].tolist():
      try:
        if (text is not None) and (text != "empety"): 
          emb = embed_type(tf.constant([text], dtype=tf.string)).numpy()[0]
          owner_cmnt_embeddings.append(emb)
        else:
          owner_cmnt_embeddings.append(np.zeros(512))
      except TypeError:
        owner_cmnt_embeddings.append(np.zeros(512))
    
    return np.asarray(owner_cmnt_embeddings)

def extract_media_cap_embeddings(dataframe):
    media_caption_embeddings = []
    embed_type = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    for text in dataframe['mediacaption'].tolist():
      try:
        if (text is not None) and (text != "empety"): 
          emb = embed_type(tf.constant([text], dtype=tf.string)).numpy()[0]
          # emb = model.encode(text)
          media_caption_embeddings.append(emb)
        else:
          media_caption_embeddings.append(np.zeros(512))
      except TypeError:
        media_caption_embeddings.append(np.zeros(512))
    
    return np.asarray(media_caption_embeddings)


def extract_all_features(dataframe):
    return generate_embeddings(dataframe), extract_owner_cmnt_embedding(dataframe), extract_likes(dataframe), extract_sentiment_features(dataframe)    

def extract_vine_features(dataframe):
    return generate_vine_embeddings(dataframe), extract_media_cap_embeddings(dataframe), extract_vine_time_features(dataframe), extract_vine_likes(dataframe), extract_vine_sentiment_features(dataframe)
    
    
    
    
    
    
  