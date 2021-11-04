# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:44:44 2021

@author: IT Doctor
"""
import keras
# import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, GRU, Dropout, Bidirectional, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, Attention, MultiHeadAttention, BatchNormalization
# from keras.utils import plot_model
import tensorflow as tf
import random as python_random
import numpy as np 
from tcn import TCN

class MultiHeadAttention_(keras.layers.Layer):
  def __init__(self, maxlen, num_heads, embed_dim):
    super().__init__()
    self.maxlen = maxlen
    self.num_heads = num_heads
    self.projection_dim = embed_dim // num_heads
    self.query_dense = Dense(embed_dim)
    self.key_dense = Dense(embed_dim)
    self.value_dense = Dense(embed_dim)
    self.combine_heads = Dense(embed_dim)

  def split_heads(self, x):
    x = tf.reshape(x, (-1, self.maxlen, 
                       self.num_heads, 
                       self.projection_dim)) 
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    return x
  
  def calc_attention(self, Q, K, V):
    dimension_k = tf.cast(K.shape[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True)/tf.math.sqrt(dimension_k)
    weights = keras.layers.Activation('softmax', name='self-attention')(scores)
    attention = tf.matmul(weights, V)
    attention = tf.transpose(attention, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(attention, (-1, self.maxlen, 
                                              self.num_heads*self.projection_dim))
    return concat_attention

  def call(self, inputs):
    Q = self.query_dense(inputs)
    K = self.key_dense(inputs)
    V = self.value_dense(inputs)

    Q = self.split_heads(Q)
    K = self.split_heads(K)
    V = self.split_heads(V)

    concat_attention = self.calc_attention(Q, K, V)
    outputs = self.combine_heads(concat_attention)
    return outputs

class PositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
    def call(self, x, maxlen):
# =============================================================================
#         maxlen = maxlen
# =============================================================================
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        # emb = embed_type(x)
        # emb = embed_type(tf.constant([x], dtype='float32'))
        return x + positions
        
embed_dim = 512 # X_train.shape[-1]  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
# =============================================================================
# maxlen = 100
# =============================================================================

def slcbc_framework(maxlen, model_type):
    ##### Embeddings inputs
    np.random.seed(123)
    python_random.seed(123)
    tf.random.set_seed(1234)
    
    embeddings_input_layer = Input(shape=(maxlen, embed_dim), dtype='float32')
    
    #### Sentiment features
    sentiments_input_layer = Input(shape=(maxlen, 6), dtype='float32')
    
    if model_type == 'gru':
        itrm_layer = tf.keras.layers.Bidirectional(GRU(5, activation='relu', return_sequences=True))
    elif model_type == 'tcn':
        itrm_layer = TCN(kernel_size=6, dilations=[1, 2, 4, 8, 16], return_sequences=True)
    # mha = MultiHeadAttention(num_heads=2, key_dim=2)
    mha = MultiHeadAttention_(maxlen, 2, 512)
    pos_embed_layer = PositionEmbedding(maxlen, embed_dim)
    
    query_seq_encoding = itrm_layer(embeddings_input_layer)
    # query_value_attention_seq = mha(embeddings_input_layer, embeddings_input_layer)
    query_value_attention_seq = mha(embeddings_input_layer)
    sentence_postion_encoding = pos_embed_layer(embeddings_input_layer, maxlen)
    
    concatLayer = Concatenate()(
        [sentence_postion_encoding, query_value_attention_seq, query_seq_encoding, sentiments_input_layer])
    # pooling = GlobalAveragePooling1D()(concatLayer)
    ####################################
    
    ###### Time features inputs
    time_input_layer = Input(shape=(maxlen,))
    time_input_reshape = keras.layers.Reshape((maxlen, 1))(time_input_layer)
    time_embed = Dense(20, use_bias=False)(time_input_reshape)
    norm_time = BatchNormalization()(time_embed)
    
    ###### Likes features inputs
    likes_input_layer = Input(shape=(1,))
    # likes_input_reshape = keras.layers.Reshape((1,))(likes_input_layer)
    likes_post = Dense(20, use_bias=False)(likes_input_layer)
    norm_likes = BatchNormalization()(likes_post)
    
    ###### Owner comment embeddings
    cmnt_input_layer = Input(shape=(512,))
    cmnt_post = Dense(20, use_bias=False)(cmnt_input_layer)
    
    ###### Concatenated 
    all_features_concat = Concatenate(axis = -1)(
        [concatLayer, norm_time]
        # [pooling, norm_time]
    )
    
    query_value_attention = GlobalAveragePooling1D()(
        all_features_concat)
    
    concat_likes = Concatenate()(
        [query_value_attention, norm_likes, cmnt_post]
    )
    
    dropout = keras.layers.GaussianNoise(stddev=0.4)(concat_likes)
    # dropout = keras.layers.Dropout(0.2)(concat_likes)
    dense = Dense(10, activation='relu')(dropout)
    # dense = Dense(16, activation='relu')(dense)
    dense = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs = [embeddings_input_layer, time_input_layer, likes_input_layer, cmnt_input_layer, sentiments_input_layer], outputs = dense)
    return model

# =============================================================================
#     model.summary()
#     model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#     
#     keras.utils.plot_model(model)
#     
#     # callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)]
#     # callbacks = [tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=5, factor=0.125)]
#     callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True),
#                   tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=5, factor=0.125)
#                   ]
# =============================================================================
