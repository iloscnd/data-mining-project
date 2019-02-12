import numpy as np
import sys
import random
import argparse


from parser import parse
from preprocess import get_XY
from utils import confusion_matrix, Globals
from train import Trainer

import torch
import torch.nn as nn
import torch.optim as optim



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, help="path to input data")
    parser.add_argument('--save_dir', help="Save pre-trained models", default="models/", nargs="?")
    parser.add_argument('--model', help="unused", default=None, nargs="?")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=10 )
    parser.add_argument('--buckets', type=int, default=10)
    parser.add_argument('--bucket_size', type=float, default=0.04)
    parser.add_argument('--hidden', type=int, default=50 )


    return parser.parse_args()




def main():

    args = get_args()
    Globals.debug = args.debug


    input_size = args.buckets * 2
    hidden_size = args.hidden
    epochs = args.epochs
    print_every = args.print_every

    X, Y = get_XY(parse(args.data), n_buckets=input_size//2, bucket_size=args.bucket_size)


    if Globals.debug:
        print(X)
        print(Y)

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(hidden_size, hidden_size//2),
        nn.ReLU(),
        nn.Linear(hidden_size//2, hidden_size//4),
        nn.Tanh(),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(hidden_size//4, 2),
        nn.LogSoftmax()
    )

    ### zrownowazyc dane
    #optimizer = optim.Adam(model.parameters(),lr=0.01, weight_decay=.3)
    optimizer = optim.LBFGS(model.parameters(), max_iter=50)
    #optimizer = optim.Rprop(model.parameters(), lr=0.1)
    loss = nn.NLLLoss()


    trainer = Trainer(model, optimizer, loss)

    trainer.run(X, Y, epochs, print_every)

    

    ## save of sth


if __name__ == "__main__":
    main()
