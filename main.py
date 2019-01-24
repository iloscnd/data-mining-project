import numpy as np
import sys
import random

from parser import parse
from preprocess import get_XY
from utils import confusion_matrix, Globals
from train import Trainer

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    if(len(sys.argv) < 2):
        return

    Globals.debug = True

    file_name = sys.argv[1]

    input_size = 5 * 2
    hidden_size = 20

    X, Y = get_XY(parse(file_name), input_size//2)


    if Globals.debug:
        print(X)
        print(Y)

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 2),
        nn.LogSoftmax()
    )

    optimizer = optim.Adam(model.parameters(), weight_decay=0.1)
    loss = nn.NLLLoss()


    trainer = Trainer(model, optimizer, loss)

    trainer.run(X, Y, 100)

    ## save of sth


if __name__ == "__main__":
    main()
