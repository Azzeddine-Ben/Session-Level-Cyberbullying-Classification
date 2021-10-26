# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 16:43:31 2021

@author: IT Doctor
"""
import numpy as np
import pandas as pd
import random as python_random
import tensorflow as tf
import warnings
import argparse
import os
from load_data_module import load_labels, load_train_features, load_test_features
from slcbc_framework import slcbc_framework
from tensorflow import keras
from sklearn.utils import class_weight
from results_prediction_module import print_learning_curves, predict_and_visualize

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

path = '/content/Drive/MyDrive/'
if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    
    # Add the arguments
    my_parser.add_argument('dataset_name',
                           metavar='dataset_name',
                           type=str,
                           help='Name of the dataset to extract features')
    my_parser.add_argument('cost_sensitive',
                           metavar='cost_sensitive',
                           type=str,
                           help='Cost sensitive method (cw: class weights / focal: focal loss)',
                           )
    args = my_parser.parse_args()
    dataset_name = args.dataset_name
    cs_method = args.cost_sensitive
    
    #### Load data 
    X_train, X_train_cmnt_emb, X_train_time, X_train_likes, X_train_sntms = load_train_features(dataset_name)
    X_test, X_test_cmnt_emb, X_test_time, X_test_likes, X_test_sntms      = load_test_features(dataset_name)
    y_train, y_test = load_labels(dataset_name)
    
    model = slcbc_framework()
    model.summary()
    chkp_path = dataset_name + '_model_classification.h5'
    mchkp = tf.keras.callbacks.ModelCheckpoint(chkp_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])     
    keras.utils.plot_model(model)
     
    # callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)]
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=5, factor=0.125)]
    # callbacks = [tf.keras.callbacks.ModelCheckpoint(chkp_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True),
    #              tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True),
    #              tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=5, factor=0.125)
    #               ]
    
    cw = class_weight.compute_class_weight('balanced',  [0, 1], y_train)
    cw_dict = {0: cw[0], 1: cw[1]}
    
    if cs_method == 'cw':
        history = model.fit(
            [X_train, X_train_time, X_train_likes, X_train_cmnt_emb],# X_train_sntms],, 
            y_train, 
            epochs=10, 
            batch_size=32, 
            validation_split=0.2,
            class_weight=cw_dict,
            callbacks=callbacks
            )        
        results_dir = dataset_name + '_classification_results_cw'
    
    ### Saving results
    os.mkdir(results_dir)
    print_learning_curves(history, results_dir)
    # model.load_weights(chkp_path)
    clf_report, confusion_matrix_fig = predict_and_visualize(model, [X_test, X_test_time, X_test_likes, X_test_cmnt_emb], y_test)# X_test_sntms], y_test)
    
    ### Saving the classification report as a CSV file
    clf_report_df = pd.DataFrame(clf_report).transpose()
    clf_report_df.to_csv(results_dir + '/classification_report.csv') 
    ### Saving the confusion matrix figure
    confusion_matrix_fig.savefig(results_dir + '/confusion_matrix')
    ### Saving the model
    model.save_weights(results_dir + '/saved_weights')
    