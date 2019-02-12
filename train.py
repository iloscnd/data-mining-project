import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils import data

import numpy as np

import utils

from utils import partition, accuracy, Globals
from sklearn import neural_network

class Trainer:

    def __init__(self, model, optimizer, scheduler, loss):
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

    def reevLoss(self, Xtr, Ytr):
        self.optimizer.zero_grad()
        loss = self.loss(self.model(Xtr), Ytr)
        #print("!!!", loss)
        loss.backward()
        return loss

    def run(self, X, Y, num_epochs,print_every, save_path, prev_best):
        
        X_train, Y_train, X_val, Y_val = partition(X,Y, test_size=0.5)
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_val = torch.from_numpy(X_val)
        Y_val = torch.from_numpy(Y_val)

        y0 = np.sum(np.array(Y_train)==0)
        y1 = np.sum(np.array(Y_train)==1)
        ###print("!!!",y0,y1)

        minNr = min(y0,y1)

        ny0, ny1 = 0, 0

        resX = []
        resY = []
        ###print(X_train.shape, Y_train.shape)
        for i in range(len(Y_train)):
            if Y_train[i] == 0 and ny0 < minNr:
                #print(X_train[i].shape)
                resX.append(np.array(X_train[i]))
                resY.append(np.array(Y_train[i]))
                ny0 += 1
            if Y_train[i] == 1 and ny1 < minNr:
                resX.append(np.array(X_train[i]))
                resY.append(np.array(Y_train[i]))
                ny1 += 1

        ###print(np.array(resX).shape)
        X_train = torch.from_numpy(np.array(resX))
        Y_train = torch.from_numpy(np.array(resY))

        y0 = np.sum(np.array(Y_train)==0)
        y1 = np.sum(np.array(Y_train)==1)
        ###print("!!!",y0,y1)
        print(X_train.shape, Y_train.shape)

        #classifier = neural_network.MLPClassifier(max_iter = 200000).fit(X_train, Y_train)
        #print(classifier.score(X_val, Y_val))
        #return

        self.model.train()
    
        best = prev_best

        for epoch in range(num_epochs):

            self.scheduler.step()

            self.optimizer.zero_grad()
            train_pred = self.model.forward(X_train)
            train_loss = self.loss(train_pred, Y_train)
            train_loss.backward()

            self.optimizer.step()
            
            with torch.no_grad():


                val_pred =self.model.forward(X_val)
                test_loss = self.loss(val_pred, Y_val)

                if Globals.log_file:
                    print(train_loss.item(),
                          test_loss.item(),
                          accuracy(train_pred.data.numpy(), Y_train.data.numpy()), 
                          accuracy(val_pred.data.numpy(), Y_val.data.numpy()),
                          sep=", ", 
                          file=Globals.log_file)

                if not (epoch % print_every) or epoch + 1 == num_epochs:

                    print("EPOCH #", epoch)

                    print( "\tTrain loss: {}, \tValidation loss: {}".format( train_loss.item(), test_loss.item() ))
            
                    if Globals.debug:
                        print("\tTrain acc: {} \tValidation acc: {}".format(accuracy(train_pred.data.numpy(), Y_train.data.numpy()), accuracy(val_pred.data.numpy(), Y_val.data.numpy())))
                        print("\tTrain confusion matrix:")
                        print(utils.confusion_matrix(np.argmax(train_pred.data.numpy(), axis=1), Y_train.data.numpy()))
                        print("\tValidation confusion matrix:")
                        print(utils.confusion_matrix(np.argmax(val_pred.data.numpy(), axis=1), Y_val.data.numpy()))

                if best > test_loss.item() and epoch > num_epochs*4./5.:
                    best = test_loss.item()
                    print("\tNEW best val loss: {}".format(best))
                    torch.save(self.model.state_dict(), save_path)

        return best

            








