# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:20:03 2022

@author: eshan
"""
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential, Input

#%%
class FeatureSelection:
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
 
class ModelDevelopment:
    def dl_model(self, X_train, y_train, nb_node=128, dropout_rate=0.3):
        model = Sequential()
        model.add(Input(shape=np.shape(X_train)[1:]))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(len(np.unique(y_train)),activation='softmax'))
        model.summary()
        return model
    
class ModelEvaluation:
    def plot_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history[list(hist.history.keys())[0]])
        plt.plot(hist.history[list(hist.history.keys())[2]])
        plt.xlabel('epoch')
        plt.legend([list(hist.history.keys())[0],list(hist.history.keys())[2]])
        plt.show()

        plt.figure()
        plt.plot(hist.history[list(hist.history.keys())[1]])
        plt.plot(hist.history[list(hist.history.keys())[3]])
        plt.xlabel('epoch')
        plt.legend([list(hist.history.keys())[1],list(hist.history.keys())[3]])
        plt.show()
        pass
