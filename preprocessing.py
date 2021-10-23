# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 21:14:27 2021

@author: IT Doctor
"""
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
import re
import itertools
import html
from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()
    
contractions_dict = {
"u" : "you",         
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "iit will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

emo_list = [ ':D',
 ':)',
 ';)',
 ';]',
 ';D',
 ':[',
 ':P',
 ':/',
 'XP',
 ';;',
 '=)',
 ':O',
 ':]',
 ':(',
 'D:',
 ':3',
 'D;',
 ':o',
 '=]',
 ':}',
 ':-)',
 'XD',
 '=D',
 ':|',
 '^_^',
 ':&',
 '=p',
 ':{',
 'D:',
 '>:/',
 'd:',
 '(-_-)',
 ':S',
 'o.O',
 ':L',
 '0:)',
 ':-*',
 '3:)',
 ':-]',
 '*)',
 '=/',
 'o_0',
 '=3',
 'oO',
 ':(',
 ':[',
 ':c',
 ';_;',
 ':x',
 ';-;',
 ":')",
 ":'(",
 ':-(',
 'DX',
 '0:3',
 ':*']

def expand_contractions(text, contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def decoding_html(text):
  # new_words = []
  # for word in words:
  #   new_word = html.unescape(word)
  #   new_words.append(new_word)
  return html.unescape(text)
# def denoise_text(text):
#     text = strip_html(text)
#     text = remove_between_square_brackets(text)
#     return text


def denoise_text(text):
    text = decoding_html(text)
    text = expand_contractions(text, contractions_dict)
    text = standardization(text)
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    # text = autospell(text)
    return text

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
      if word not in emo_list:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
      if word not in emo_list:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def standardization(text):
  text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
  return text

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def rename_dataframe(df, to_sort):
  if to_sort == True:
    for column in df.columns:
      df.rename(columns={column : int(column[4:])}, inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)
  else:
    for column in df.columns:
      df.rename(columns={column : 'clmn'+ str(column)}, inplace=True) 
  return df

def process_dataframe(df):
  list_series = []
  for _, j in df.iterrows():
    list_series.append(j.apply(strip_tags))
  list_columns = df.columns.to_list()
  new_df = pd.DataFrame(data=list_series, columns=list_columns)
  new_df_copy = new_df.copy()
  new_df_copy = new_df_copy[new_df_copy.columns[1:]]
  new_df_copy = rename_dataframe(new_df_copy, to_sort=True)
  new_df_copy = rename_dataframe(new_df_copy, to_sort=False)
  new_df_copy['label'] = df['question2']
  return new_df_copy

### Preprocess owner_id column
def preprocess_onwer_id(x):
  return x[22:-9]

### Preprocessing text comments
def initial_preprocessing(text):
  text = re.sub(r"(?:\@|https?\://)\S+", "", text)
  if text != "empety":
    text = text.split(' ', 1)[1][:-33]
  return text

### Preprocessing a session
def preprocessing_session(text):
  b = '_ÙÕª_ÙÓÔ'
  for char in b: text=text.replace(char, '')
  text = ''.join(remove_punctuation(''.join(remove_non_ascii(''.join(to_lowercase(denoise_text(initial_preprocessing(text))))))))
  if text == '  ' or text == ' ' or text == '':
    text = 'empety'
  return text

### Function to relabel the dataset
def relabeling_dataset(df):
  relabeled_dataset = pd.DataFrame(columns = df.columns)
  list_ids = df['_unit_id'].unique()
  for id in list_ids:
    noneBll = 0
    simple_df = df.loc[df['_unit_id'] == id]
    # duplicates = simple_df.pivot_table(index=['label'], aggfunc='size')
    for _, row in simple_df.iterrows():
      if row['label'] == 'noneBll':
        noneBll += 1
    simple_row = simple_df.iloc[0].copy()
    if noneBll >=3:
      simple_row['label'] = 'noneBll'
    else:
      simple_row['label'] = 'bullying'
    relabeled_dataset = relabeled_dataset.append(simple_row, ignore_index=True)
  return relabeled_dataset

### Applying session preprocessing to a dataframe
def preprocess_data(dataframe):
  for column in dataframe.columns[:-5]:
    dataframe[column] = dataframe[column].apply(preprocessing_session)