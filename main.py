import numpy as np
import sys
import random
import argparse


from parser import parse
from preprocess import get_XY
from utils import confusion_matrix, Globals, accuracy
from train import Trainer

import torch
import torch.nn as nn
import torch.optim as optim



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, help="path to input data")
    parser.add_argument('--save_path', help="Save pre-trained models", default=None, nargs="?")
    parser.add_argument('--log_path', help="Where to save logs. No logs if unset", default=None, nargs="?")
    parser.add_argument('--model', help="unused", default=None, nargs="?")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=10 )
    parser.add_argument('--buckets', type=int, default=10)
    parser.add_argument('--bucket_size', type=float, default=0.04)
    parser.add_argument('--hidden', type=int, default=50 )


    return parser.parse_args()




def main():

    args = get_args()
    Globals.debug = args.debug
    if args.log_path:
        Globals.log_file = open(args.log_path, "w")
        print("Train_loss, Validation_loss, Train_acc, Validation_acc", file=Globals.log_file)

    input_size = args.buckets * 2
    hidden_size = args.hidden
    epochs = args.epochs
    print_every = args.print_every

    (X, Y), _ = get_XY(parse(args.data), n_buckets=input_size//2, bucket_size=args.bucket_size)

    if args.save_path is None:
        set_number = args.data[args.data.find('.')+1:args.data.rfind('.')]
        args.save_path = "models/model"+ set_number


    if Globals.debug:
        print(X)
        print(Y)
        print(args.save_path)

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Dropout(p=0.6, inplace=False),
        nn.Linear(hidden_size, hidden_size//4),
        nn.ReLU(),
        nn.Linear(hidden_size//4, 2)
    )

    optimizer = optim.Adam(model.parameters(),lr=1., weight_decay=1.)
    #optimizer = optim.LBFGS(model.parameters(), lr=1000., max_iter=50)
    #optimizer = optim.Rprop(model.parameters(), lr=0.1)
    #optimizer = optim.Adam(model.parameters(), weight_decay=0.1, lr=10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.4)

    loss = nn.CrossEntropyLoss()

    if args.eval:  # dokładność na całym zbiorze, nie tylko części testowej, nie użyte w raporcie
        model.load_state_dict(torch.load(args.save_dir + "model"))
        model.eval()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        pred = model(X)
        loss = loss(pred, Y)
        print("Loss: {}".format(loss.item()))
        print("Acc: {}".format(accuracy(pred.data.numpy(), Y.data.numpy())))

    else:
        #optimizer = optim.Adam(model.parameters(),lr=0.001, weight_decay=0.2)
#        optimizer = optim.LBFGS(model.parameters(), lr=1)

        best = 10e18

        for _ in range(10):

            model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(p=0.6, inplace=False),
                nn.Linear(hidden_size, hidden_size//4),
                nn.ReLU(),
                nn.Linear(hidden_size//4, 2)
            )

            optimizer = optim.Adam(model.parameters(),lr=1., weight_decay=1.)
            #optimizer = optim.LBFGS(model.parameters(), lr=1000., max_iter=50)
            #optimizer = optim.Rprop(model.parameters(), lr=0.1)
            #optimizer = optim.Adam(model.parameters(), weight_decay=0.1, lr=10)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)


            trainer = Trainer(model, optimizer, scheduler, loss)
            best = trainer.run(X, Y, epochs,print_every, args.save_path, best)



if __name__ == "__main__":
    main()
