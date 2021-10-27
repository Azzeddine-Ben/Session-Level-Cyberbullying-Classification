# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:20:40 2021

@author: BENAISSA-Azzeddine
"""

import numpy as np
import pandas as pd
import preprocessing
from features_extraction import extract_all_features, extract_time_features
from sklearn.model_selection import train_test_split
import os 
import pickle

if __name__ == '__main__':
    ### Reading the data
    path ='/content/drive/MyDrive/'
    path_first_slice  = path + "labeled_0plus_to_10__full.csv"
    path_second_slice = path + "labeled_10plus_to_40_full.csv"
    path_third_slice  = path + "labeled_40plus_full.csv"
    
    paths_list, dataframes_list, original_dataframes_list = [path_first_slice, path_second_slice, path_third_slice], [], []
       
    for path in paths_list:
      df = pd.read_csv(path, encoding = 'ISO-8859-1', header = 0)
      original_dataframes_list.append(df)
      dataset = df.iloc[:,15:211]
      dataset = preprocessing.process_dataframe(dataset)
      dataset['_unit_id'] = df['_unit_id']
      dataset['cptn_time']  = df['cptn_time']
      dataset['likes']      = df['likes']
      dataset['owner_id']   = df['owner_id']
      dataset['owner_cmnt'] = df['owner_cmnt']
      dataframes_list.append(dataset)
      
    csv_dataframe = pd.concat([original_dataframes_list[0], original_dataframes_list[1], original_dataframes_list[2]])
    
    print("Preprocessing dataframe ... \n")
    ### Reordering the columns and dropping unused ones
    relabeled_dataset = pd.concat([dataframes_list[0], dataframes_list[1], dataframes_list[2]])
    
    ### Preprocessing owner_id column
    relabeled_dataset['owner_id'] = relabeled_dataset['owner_id'].apply(preprocessing.preprocess_onwer_id)
    
    ### Making one row per session
    relabeled_dataset = preprocessing.relabeling_dataset(relabeled_dataset)
    relabeled_dataset_copy = relabeled_dataset.copy()
    
    ### Preprocessing text comments
    preprocessing.preprocess_data(relabeled_dataset)
    
    print("Preprocessing Done. \n")
    
    #### define labels
    labels = []
    for x in relabeled_dataset["label"]:  
      if x == "noneBll":
        labels.append(0)
      else:
        labels.append(1)
    labels = np.asarray(labels)
    
    
    ### Here you can choose the number of comments to keep in a session 
    for num in range(101,196):
      relabeled_dataset.drop('clmn'+ str(num), inplace=True, axis=1)
      
    ### Extract features
    print("Extracting features ... \n")
    dataset_embeddings, cmnt_owner_embeddings, like_features, sentiment_features = extract_all_features(relabeled_dataset)
    time_features = extract_time_features(relabeled_dataset_copy)
    print("Extraction Done. \n")
    
    ### Train test split
    print("Splitting data ... \n")
    X_train, X_test, y_train, y_test  = train_test_split(dataset_embeddings, labels, test_size = 0.2, random_state = 42)
    X_train_cmnt_emb, X_test_cmnt_emb = train_test_split(cmnt_owner_embeddings, test_size = 0.2, random_state = 42)
    X_train_time, X_test_time         = train_test_split(time_features, test_size = 0.2, random_state = 42)
    X_train_likes, X_test_likes       = train_test_split(like_features, test_size = 0.2, random_state = 42)
    X_train_sntms, X_test_sntms       = train_test_split(sentiment_features, test_size = 0.2, random_state = 42)
    
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    print("Spliiting Done. \n")
    
    print("Saving train & test data ... \n")
    directory = 'instagram_data'
    if (os.path.exists(directory)) == False:  
        os.mkdir(directory)
        # train_dir, valid_dir, test_dir = directory + '/train', directory + '/valid', directory + '/test'
        train_dir, test_dir = directory + '/train', directory + '/test'
        os.mkdir(train_dir)
        # os.mkdir(valid_dir)
        os.mkdir(test_dir)
        
        pickle.dump( X_train, open( train_dir + "/X_train", "wb" ) )
        pickle.dump( y_train, open( train_dir + "/y_train", "wb" ) )        
        pickle.dump( X_train_cmnt_emb, open( train_dir + "/X_train_cmnt_emb", "wb" ) )        
        pickle.dump( X_train_time, open( train_dir + "/X_train_time", "wb" ) )       
        pickle.dump( X_train_likes, open( train_dir + "/X_train_likes", "wb" ) )         
        pickle.dump( X_train_sntms, open( train_dir + "/X_train_sntms", "wb" ) )      
    
        pickle.dump( X_test, open( test_dir + "/X_test", "wb"))
        pickle.dump( y_test, open( test_dir + "/y_test", "wb" ) )
        pickle.dump( X_test_cmnt_emb, open( test_dir + "/X_test_cmnt_emb", "wb" ))
        pickle.dump( X_test_time, open( test_dir + "/X_test_time", "wb" ) )
        pickle.dump( X_test_likes, open( test_dir + "/X_test_likes", "wb" ))
        pickle.dump( X_test_sntms, open( test_dir + "/X_test_sntms", "wb" ) )     
        
    
    # =============================================================================
    #     pickle.dump( X_valid, open( valid_dir + "/X_valid", "wb" ) )
    #     pickle.dump( y_valid, open( valid_dir + "/y_valid", "wb" ) )
    #     pickle.dump( X_valid_stylometric, open( valid_dir + "/X_valid_stylometric", "wb" ))
    #     pickle.dump( X_valid_lexical, open( valid_dir + "/X_valid_lexical", "wb" ) )
    #     pickle.dump( X_valid_readability, open( valid_dir + "/X_valid_readability", "wb" ))
    #     pickle.dump( X_valid_liwc, open( valid_dir + "/X_valid_liwc", "wb" ) )     
    #     pickle.dump( X_valid_sentiments, open( valid_dir + "/X_valid_sentiments", "wb" ) )
    # =============================================================================
        print("Terminated.")



  
  
  
  