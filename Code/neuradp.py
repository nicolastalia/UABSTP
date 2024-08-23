# importing libraries
from geopy.geocoders import Nominatim
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from collections import Counter
import copy
import networkx as nx
import random
import math
import operator

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model, clone_model
from keras import layers
from keras import ops
from keras import initializers
from tensorflow.keras.models import Sequential

import states as st


class ConstInitializer(initializers.Initializer):
    def __init__(self, value=0.3):
        self.value = value
    def __call__(self, shape, dtype=None):
        return tf.constant(self.value, shape=shape, dtype=dtype)
        # return tf.ones(shape, dtype=dtype

class ZeroInitializer(initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.zeros(shape, dtype=dtype)

class NeurADP:
    def __init__(self,centres_len):
        self.shape = 3
        self.model = self.model_NN()
        self.target = clone_model(self.model)
        self.target_update_tau = 0.1
    
    def model_NN(self):
        modelnn = Sequential()
        # Linear Regression
        modelnn.add(Dense(1, input_shape=(self.shape,), activation='linear'))

        # Uncomment this for NN with hidden layers
        # modelnn.add(Dense(200,activation='relu',input_shape=(self.shape,)))
        # modelnn.add(Dense(300, activation='elu'))
        # modelnn.add(Dense(300, activation='elu'))
        # modelnn.add(Dense(1, activation='linear'))  # Output layer for regression (no activation))


        modelnn.compile(optimizer='adam', loss='mean_squared_error')
        return modelnn
    
    def train_model(self,buf):
        exp_sample = random.sample(buf,100)
        x_first_train = np.empty((0, 6))
        y_first_train = np.array([])
        for exp in exp_sample:
            for veh_ind, veh_val in exp.state.vehicles.items():
                y_first_train = np.append(y_first_train,exp.state.vehicles[veh_ind].value_function)
                x_first_train = np.vstack((x_first_train, exp.state.feature_nn()))
                # x_first_train = np.append(x_first_train,exp.state.feature_nn(veh_ind).reshape(1,-1))
        print(f'x shape is {x_first_train.shape} and y shape is {y_first_train.shape}')
        self.model.fit(x_first_train, y_first_train, epochs=200, batch_size=32)
        

    def balanced_sample(self,buffer,sample_size):
        sampled_objects = random.sample(buffer.experiences,sample_size)
        return sampled_objects

    def update_parameters(self,M):
        experience_sample = random.sample(M,30)
        ypredzero = 0
        for exp in experience_sample:
            post_m = copy.deepcopy(exp)
            pre_m = copy.deepcopy(exp)
            actions = pre_m.matching_ADP(individual_objective=True)
            if len(actions) > 0:
                for action in actions:
                    x_train = pre_m.feature_nn().reshape(1, -1)
                    y_train = np.array(action[2])
                    ypredzero += 1
                    with tf.GradientTape() as tape:
                        # Forward pass
                        predictions = self.model(x_train, training=True)
                        # print(f'prediction is {predictions}')
                        
                        # Compute the loss
                        loss = tf.keras.losses.MSE(y_train, predictions)

                    # # Compute gradients
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    # print(f'gradients: {gradients}')

                    # Update the weights manually
                    # optimizer = SGD(learning_rate=0.01)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    def _soft_update_function(self, target_model, source_model):
        target_weights = target_model.trainable_weights
        source_weights = source_model.trainable_weights

        updates = []
        for target_weight, source_weight in zip(target_weights, source_weights):
            new_weights = self.target_update_tau * source_weight + (1. - self.target_update_tau) * target_weight
            target_weight.assign(new_weights)

    def soft_update(self,buffer):
        feat_0 = buffer.experiences[0].pos_state.feature_nn('vehicle_1')
        experience_sample = self.balanced_sample(buffer,32)
        x_train = np.empty((0, feat_0.shape[0]))
        y_train = np.array([])
        for exp in experience_sample:
            index = buffer.pre_str.index(exp.pre_state_str)
            buffer.trained[index] += 1
            pre_m = copy.deepcopy(exp.pre_state)
            actions = pre_m.matching_ADP(individual_objective=True)
            for action in actions:
                x_train = np.vstack((x_train, exp.pre_state.feature_nn(action[0])))
                y_train = np.append(y_train,action[2])
        self.model.fit(x_train, y_train, epochs=1, batch_size=32) # this update the weights of the model
        self._soft_update_function(self.target,self.model)


    def update_parameters2(self,buf):
        print(f'training paramv2')
        items_buf = list(buf.values())
        experience_sample = random.sample(items_buf,30)
        min_time = 480
        max_time = 960

        for exp in experience_sample:
            
            pre_m = copy.deepcopy(exp.statepre)
            post_m = copy.deepcopy(exp.statepos)
            act = copy.deepcopy(exp.statepos.action)
            if pre_m.time > 490:
                for v in pre_m.vehicles.keys():
                    feature = post_m.feature_nn()
                    value_hat = st.state.nnmodel.predict(feature)[0][0]
                    reward = 0
                    if len(act) > 0:
                        for a in act:
                            if a[0] == v:
                                reward = a[1]['lab']
                    y_train = np.array(reward + ((max_time - pre_m.time)/min_time)*value_hat)
                    post_m_tminus1 = copy.deepcopy(buf[post_m.time - 10,exp.realization].statepos)
                    feature_post_past = pre_m.feature_nn().reshape(1, -1)
                    x_train = feature_post_past
                
                    with tf.GradientTape() as tape:
                            # Forward pass
                            predictions = self.model(x_train, training=True)
                            
                            # Compute the loss
                            loss = tf.keras.losses.MSE(y_train, predictions)

                    # # Compute gradients
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    # print(f'gradients: {gradients}')

                    # Update the weights manually
                    # optimizer = SGD(learning_rate=0.01)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                        
       