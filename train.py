import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils import data

import numpy as np

from utils import partition, accuracy, Globals
from sklearn import neural_network

class Trainer:

    def __init__(self, model, optimizer, loss):
        self.model = model

        self.optimizer = optimizer
        self.loss = loss

    def reevLoss(self, Xtr, Ytr):
        self.optimizer.zero_grad()
        loss = self.loss(self.model(Xtr), Ytr)
        #print("!!!", loss)
        loss.backward()
        return loss

    def run(self, X, Y, num_epochs, print_every):
        
        X_train, Y_train, X_val, Y_val = partition(X,Y)
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
        ###print(X_train.shape, Y_train.shape)

        #classifier = neural_network.MLPClassifier(max_iter = 200000).fit(X_train, Y_train)
        #print(classifier.score(X_val, Y_val))
        #return

        self.model.train()

        print(self.model)

        #ds = data.Dataset(X_train, Y_train)
        #dl = data.DataLoader(ds, shuffle = True)

        for epoch in range(num_epochs):

            self.optimizer.zero_grad()
            train_pred = self.model(X_train)
            train_loss = self.loss(train_pred, Y_train)
            train_loss.backward()
            self.optimizer.step()#lambda: ((lambda obj, xs, ys: obj.reevLoss(xs, ys))(self, X_train, Y_train)))  #train_loss)
            
            self.model.eval() ## torch.no_grad nie działa na mojej wersji pytrocha

            test_loss = self.loss(self.model(X_val), Y_val)

            val_pred =self.model(X_val)
            #print(val_pred.shape)

            if not (epoch % print_every) or epoch + 1 == num_epochs:

                print("EPOCH #", epoch)

                print( "\tTrain loss: {}, \tValidation loss: {}".format( train_loss.item(), test_loss.item() ))
        
                if Globals.debug:
                    print("\tTrain acc: {} \tValidation acc: {}".format(accuracy(train_pred.data.numpy(), Y_train.data.numpy()), accuracy(val_pred.data.numpy(), Y_val.data.numpy())))


            self.model.train()








