import torch
import torch.nn as nn
import torch.autograd as autograd

import numpy as np

from utils import partition, accuracy, Globals

class Trainer:

    def __init__(self, model, optimizer, loss):
        self.model = model

        self.optimizer = optimizer
        self.loss = loss


    def run(self, X, Y, num_epochs, save_path):
        
        X_train, Y_train, X_val, Y_val = partition(X,Y)
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_val = torch.from_numpy(X_val)
        Y_val = torch.from_numpy(Y_val)

        self.model.train()
    
        best = 10e18


        for epoch in range(num_epochs):

            print("EPOCH #", epoch)

            self.optimizer.zero_grad()
            train_pred = self.model(X_train)
            train_loss = self.loss(train_pred, Y_train)
            train_loss.backward()
            self.optimizer.step(lambda: -train_loss)
            
            self.model.eval() ## torch.no_grad nie działa na mojej wersji pytrocha

            test_loss = self.loss(self.model(X_val), Y_val)

            val_pred =self.model(X_val)
            #print(val_pred.shape)

            print( "\tTrain loss: {}, \tValidation loss: {}".format( train_loss.item(), test_loss.item() ))
        
            if Globals.debug:
                print("\tTrain acc: {} \tValidation acc: {}".format(accuracy(train_pred.data.numpy(), Y_train.data.numpy()), accuracy(val_pred.data.numpy(), Y_val.data.numpy())))

            if best > test_loss.item():
                best = test_loss.item()
                torch.save(self.model.state_dict(), save_path+"model")

            self.model.train()








