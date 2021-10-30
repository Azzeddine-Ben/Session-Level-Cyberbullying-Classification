# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 22:45:19 2021

@author: IT Doctor
"""
import numpy as np
import pandas as pd
import preprocessing
from features_extraction import extract_vine_features
from sklearn.model_selection import train_test_split
import os
import pickle

if __name__ == '__main__':
    ### Reading the data
    path ='/content/drive/MyDrive/'
    vine_dataset = pd.read_csv(path +'vine_labeled_cyberbullying_data.csv', encoding="ISO-8859-1", header = 0)
    
    vine_dataframe_org = pd.DataFrame()
# =============================================================================
#     clmn_nbr = 1
# =============================================================================
    
    for clmn_nbr in range(1, 661):
      vine_dataframe_org['clmn'+str(clmn_nbr)] = vine_dataset['column'+ str(clmn_nbr)]
    
    vine_dataframe_org['_unit_id']  = vine_dataset['_unit_id']
    vine_dataframe_org['label']     = vine_dataset['question2']
    vine_dataframe_org['cptn_time'] = vine_dataset['creationtime']
    vine_dataframe_org['likes']     = vine_dataset['likecount']
    
    #### Preprocessing 
    print('Preprocessing data ... \n')
    vine_dataframe_org['label'].replace('noneBll', 0, inplace = True)
    vine_dataframe_org['label'].replace('bullying', 1, inplace = True)
    
    vine_dataframe_pr = pd.DataFrame()
    for column in vine_dataframe_org.columns[:-4]:
      vine_dataframe_pr[column] = vine_dataframe_org[column].apply(preprocessing.preprocessing_vine_session)
    
    for column in vine_dataframe_org.columns[-4:]:
      vine_dataframe_pr[column] = vine_dataframe_org[column]
      
    for num in range(101,661):
      vine_dataframe_org.drop('clmn'+ str(num), inplace=True, axis=1)  
      
    print('Preprocessing done. \n')
    
    print("Extracting Features ... \n")
    
    dataset_embeddings, media_cap_embeddings, time_features, likes_features = extract_vine_features(vine_dataframe_org)
    
    print("Features extraction done. \n")
    
    #### define labels
    labels = vine_dataframe_pr['label'].tolist()
    
    print("Splitting data ... \n")
    X_train, X_test, y_train, y_test  = train_test_split(dataset_embeddings, labels, test_size=0.2, random_state=42)
    X_train_time, X_test_time         = train_test_split(time_features, test_size = 0.2, random_state = 42)
    X_train_likes, X_test_likes       = train_test_split(likes_features, test_size = 0.2, random_state = 42)
    X_train_media_cpt, X_test_media_cpt = train_test_split(media_cap_embeddings, test_size = 0.2, random_state = 42)
    
    X_train, X_test, y_train, y_test = np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)
    print("Splitting done.")  

    print("Saving train & test data ... \n")
    directory = 'vine_data'
    if (os.path.exists(directory)) == False:  
        os.mkdir(directory)
        # train_dir, valid_dir, test_dir = directory + '/train', directory + '/valid', directory + '/test'
        train_dir, test_dir = directory + '/train', directory + '/test'
        os.mkdir(train_dir)
        # os.mkdir(valid_dir)
        os.mkdir(test_dir)
        
        pickle.dump( X_train, open( train_dir + "/X_train", "wb" ) )
        pickle.dump( y_train, open( train_dir + "/y_train", "wb" ) )        
        pickle.dump( X_train_media_cpt, open( train_dir + "/X_train_mediacap_emb", "wb" ) )        
        pickle.dump( X_train_time, open( train_dir + "/X_train_time", "wb" ) )       
        pickle.dump( X_train_likes, open( train_dir + "/X_train_likes", "wb" ) )         
# =============================================================================
#         pickle.dump( X_train_sntms, open( train_dir + "/X_train_sntms", "wb" ) )      
# =============================================================================
    
        pickle.dump( X_test, open( test_dir + "/X_test", "wb"))
        pickle.dump( y_test, open( test_dir + "/y_test", "wb" ) )
        pickle.dump( X_test_media_cpt, open( test_dir + "/X_test_mediacap_emb", "wb" ))
        pickle.dump( X_test_time, open( test_dir + "/X_test_time", "wb" ) )
        pickle.dump( X_test_likes, open( test_dir + "/X_test_likes", "wb" ))
# =============================================================================
#         pickle.dump( X_test_sntms, open( test_dir + "/X_test_sntms", "wb" ) )     
# =============================================================================
        
    
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
      
      
      
      