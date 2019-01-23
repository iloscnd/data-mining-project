import numpy as np
import sys
import random

from parser import parse
from preprocess import get_XY
from utils import confusion_matrix


from sklearn.neural_network import MLPClassifier

def main():
    if(len(sys.argv) < 2):
        return

    file_name = sys.argv[1]

    X, Y = get_XY(parse(file_name))

    print(X)
    print(Y)
    classifier = MLPClassifier(solver='lbfgs', activation="relu", alpha=1e-4, hidden_layer_sizes=(800, 800, 800), max_iter=200)

    n = len(Y)
    k = int(0.8 * n)
    perm = np.random.permutation(n)


    X = X[perm]
    Y = Y[perm]
    print(X.shape)

    classifier.fit(X[:k], Y[:k])

    Y_check = classifier.predict(X[k:])

    print("test: ", classifier.score(X[k:],Y[k:]), ", train: ", classifier.score(X[:k],Y[:k]))
    print(confusion_matrix(Y_check, Y[k:]))
    

#    print(classifier.predict_proba(np.array([-1]).reshape(1,1)))


if __name__ == "__main__":
    main()
