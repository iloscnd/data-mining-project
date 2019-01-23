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

