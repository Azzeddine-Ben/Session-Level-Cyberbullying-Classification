# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:25:17 2021

@author: IT Doctor
"""
import pickle


path = "/content/drive/MyDrive"
def load_labels(dataset_name):
    pickle_in = open(path + '/' + dataset_name + '_data/train/y_train', 'rb' )
    y_train   = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(path + '/' + dataset_name + '_data/test/y_test', 'rb')
    y_test    = pickle.load(pickle_in)
    pickle_in.close()
    return y_train, y_test

def load_train_features(dataset_name):
    
    pickle_in = open(path + '/' + dataset_name + '_data/train/X_train', 'rb')
    X_train = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(path + '/' + dataset_name + '_data/train/X_train_cmnt_emb', 'rb')
    X_train_cmnt_emb = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(path + '/' + dataset_name + '_data/train/X_train_time', 'rb')
    X_train_time = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(path + '/' + dataset_name + '_data/train/X_train_likes', 'rb')
    X_train_likes = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(path + '/' + dataset_name + '_data/train/X_train_sntms', 'rb')
    X_train_sntms = pickle.load(pickle_in)
    pickle_in.close()    
  
    return [X_train, X_train_cmnt_emb, X_train_time, X_train_likes, X_train_sntms]


def load_test_features(dataset_name):
    
    pickle_in = open(path + '/' + dataset_name + '_data/test/X_test', 'rb')
    X_test = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(path + '/' + dataset_name + '_data/test/X_test_cmnt_emb', 'rb')
    X_test_cmnt_emb = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(path + '/' + dataset_name + '_data/test/X_test_time', 'rb')
    X_test_time = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(path + '/' + dataset_name + '_data/test/X_test_likes', 'rb')
    X_test_likes = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(path + '/' + dataset_name + '_data/test/X_test_sntms', 'rb')
    X_test_sntms = pickle.load(pickle_in)
    pickle_in.close()    
  
    return [X_test, X_test_cmnt_emb, X_test_time, X_test_likes, X_test_sntms]