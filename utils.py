import numpy as np




def confusion_matrix(prediction, true_vals, classes=None):
    if classes is None:
        classes = np.unique(true_vals, axis=0)
        

    res = []

    for clss1 in classes:
        row = []
        for clss2 in classes:
            row.append( np.logical_and((true_vals == clss2), (prediction == clss1)).sum()  )
        
        res.append(np.array(row))
    
    return np.array(res)

def partition(X, Y, test_size=0.8):
    n = len(Y)
    k = int(0.8 * n)
    perm = np.random.permutation(n)


    X = X[perm]
    Y = Y[perm]
    if Globals.debug:
        print(X.shape)

    return X[k:],Y[k:], X[:k], Y[:k]


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


class Globals(object):
    debug = False
