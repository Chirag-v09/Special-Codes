import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import models
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras import metrics
from itertools import combinations_with_replacement, product


class BestModel:
    
    def __init__(self):
        
        self.loss_list = []
        self.val_loss_list = []
        self.acc_list = []
        self.val_acc_list = []
        self.layer_name = []

    
    def findmodel(self, X, y, input_shape, output_shape, activation_first, activations, activation_output, optimizer, loss_function, metric, epochs_num, apply_art = 'combination', val_X = 0, val_y = 0, layers = [2, 3], neurons = [64, 128, 256], batch_size = [64, 128]):
        
        self.__init__()
        self.epochs = epochs_num
        print("Testing on Architectures which are get by appyling : " + apply_art)
        for i in layers:
            for k in batch_size:
                
                temp = []
                if(apply_art == "permutation"):
                    oc = product(neurons, i)
                else:
                    oc = combinations_with_replacement(neurons, i)
                # op = permutations(n, i)
                for o in oc:
                    temp.append(list(o))
                
                for a in range(len(temp)):
                    
                    model = models.Sequential()
                    model.add(Dense(temp[a][0], activation = activations, input_shape = input_shape))
                    
                    for b in range(1, i):
                        model.add(Dense(temp[a][b], activation = 'relu'))
                    
                    model.add(Dense(output_shape, activation = activation_output))
                    
                    model.compile(optimizer = optimizer, loss = loss_function, metrics = metric)
                    
                    history = model.fit(X, y, batch_size = k, epochs = epochs_num, validation_data = (val_X, val_y), verbose = 1)
                
                    self.history_dict = history.history
                    
                    loss = self.history_dict['loss']
                    val_loss = self.history_dict['val_loss']
                    self.loss_list.append(loss)
                    self.val_loss_list.append(val_loss)
                    
                    acc = self.history_dict['acc']
                    val_acc = self.history_dict['val_acc']
                    self.acc_list.append(acc)
                    self.val_acc_list.append(val_acc)
                    
                    neurons_num = [str(value) for value in temp[a]]
                    neurons_num = ' '.join(neurons_num)
                    self.layer_name.append("Layers:- " + str(i) + " : Neurons:- " + str(neurons_num) + " : Batch Size:- " + str(k))
                    print("layer Complete")
                    
        
        self.createdataframe()
        return self.history_dict
    
    
    def createdataframe(self):
        
        data = {"loss":self.loss_list, "val_loss":self.val_loss_list, "acc":self.acc_list, "val_acc":self.val_acc_list, "layers_name":self.layer_name}
        self.dataframe = pd.DataFrame(data)
     
    
    def savemodelhistory(self):
        
        self.dataframe.to_csv("History.csv")
        return self.dataframe
    
   
    def savefullhistory(self):
        
        dataframe = pd.DataFrame(self.history_dict)
        dataframe.to_csv("Full History.csv")
        return dataframe
    
    
    def getmodel(self, basis = 1):
        
        if(basis == 1):
            factor = len(self.val_loss_list[0]) - 1
        elif(basis == 2):
            factor = len(self.val_loss_list[0]) - 2
        else:
            factor = len(self.val_loss_list[0]) - 3
        
        val_loss_avg = []
        for i in range(len(self.val_loss_list)):
            val = 0
            for j in range(len(self.val_loss_list[0]) - factor):
                val += self.val_loss_list[i][j + factor]
            val_loss_avg.append(val)
        
        # self.best_hyper_parameter_index = val_loss_avg.index(min(val_loss_avg))
        self.best_hyper_parameter_index = np.argmin(val_loss_avg)
        self.best_hyper_parameter = self.dataframe.iloc[self.best_hyper_parameter_index, :]
        
        return self.best_hyper_parameter
    
    
    def loss_plot(self):
        
        epoch = np.arange(self.epochs)
        plt.plot(epoch, self.loss_list[self.best_hyper_parameter_index], 'bo', label = 'Training Loss')
        plt.plot(epoch, self.val_loss_list[self.best_hyper_parameter_index], 'b', label = 'validation loss')
        plt.title("Training & Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


    def acc_plot(self):
        
        epoch = np.arange(self.epochs)
        plt.plot(epoch, self.acc_list[self.best_hyper_parameter_index], 'bo', label = 'Training Accuracy')
        plt.plot(epoch, self.val_acc_list[self.best_hyper_parameter_index], 'b', label = 'Validation Accuracy')
        plt.title("Training & Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()




