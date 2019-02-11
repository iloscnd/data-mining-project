import torch
import torch.nn as nn
import torch.autograd as autograd

import numpy as np

from utils import partition, accuracy, Globals
from sklearn import neural_network

class Trainer:

    def __init__(self, model, optimizer, loss):
        self.model = model

        self.optimizer = optimizer
        self.loss = loss

    def reevLoss(self, Xtr, Ytr):
        loss = self.loss(self.model(Xtr), Ytr)
        #print("!!!", loss)
        #loss.backward()
        return loss

    def run(self, X, Y, num_epochs, print_every):
        
        X_train, Y_train, X_val, Y_val = partition(X,Y)
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_val = torch.from_numpy(X_val)
        Y_val = torch.from_numpy(Y_val)

        #classifier = neural_network.MLPClassifier(max_iter = 200000).fit(X_train, Y_train)
        #print(classifier.score(X_val, Y_val))
        #return

        self.model.train()

        print(self.model)

        for epoch in range(num_epochs):

            self.optimizer.zero_grad()
            train_pred = self.model(X_train)
            train_loss = self.loss(train_pred, Y_train)
            train_loss.backward()
            self.optimizer.step(lambda: ((lambda obj, xs, ys: obj.reevLoss(xs, ys))(self, X_train, Y_train)))  #train_loss)
            
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








