import numpy as np
import sys

from parser import parse
from utils import confusion_matrix

from sklearn import linear_model

#import random


def prep_data(data, train_size=0.8):
    
    prev = list(data.keys())[0]
    imbalances = []
    growths = []
    for key in list(data.keys())[1:]:

        if data[prev][2] != data[key][2]: #only if mid-price changes
            growths.append(data[prev][2] < data[key][2])
            imbalances.append(data[prev][3])
        prev = key

    
    k = int( len(imbalances) * train_size)


    imbalances = np.array(imbalances).reshape(-1,1)
    growths = np.array(growths)

    perm = np.random.permutation(len(growths))

    imbalances = imbalances[perm]
    growths = growths[perm]


    #print(imbalances, growths)
    return imbalances[:k], growths[:k], imbalances[k:], growths[k:]



def main():
    if(len(sys.argv) < 2):
        return

    file_name = sys.argv[1]

    X_train, Y_train, X_test, Y_test = prep_data(parse(file_name))

    classifier = linear_model.SGDClassifier(loss="log", alpha=0.1, max_iter=3000, tol=0, shuffle=False)

    classifier.fit(X_train, Y_train)

    Y_check = classifier.predict(X_test)

    print(classifier.score(X_test,Y_test))
    print(confusion_matrix(Y_check, Y_test))
    

    print(classifier.predict_proba(np.array([-1]).reshape(1,1)))





if __name__ == "__main__":
    main()











