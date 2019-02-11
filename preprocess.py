import numpy as np 

from utils import Globals

def fits(index, arr):
    return index >= 0 and index < len(arr)


def get_XY(data, n_buckets=5, bucket_size=0.05 , omit_no_change=True):
    
    keys = list(data.keys())
    keys.sort()

    growths = []
    X = []

    for i, curr in enumerate(keys[:-1]):
        currKey = curr
        nextKey = keys[i+1]

        #print("!", curr, nextKey)

        currday, currh = currKey.split()
        nextday, nexth = nextKey.split()

        currh = int(currh[:4])
        nexth = int(nexth[:4])

        

        if currh < 900 or currh > 1600 or currh + 1 != nexth:
            continue

        #print(currh, nexth)

        #norm = 0.

        if not omit_no_change or data[nextKey][2] != data[currKey][2]:
            rows = np.zeros(2*n_buckets)
            
            mid_price = data[currKey][2]
            centers = (np.arange(2*n_buckets) - n_buckets + 0.5)*bucket_size*mid_price + mid_price
            #print(mid_price, centers)

            #print(mid_price)
            for bid_price, bid_size in reversed(data[currKey][0]):
                ###bucket = int(( n_buckets*bucket_size -  (mid_price-bid_price)/mid_price)/bucket_size)
                #print(bucket)
                ###if fits(bucket, rows):
                ###    rows[bucket] += bid_size
                #norm += bid_size
                for i in range(len(rows)//2):
                    rows[i] += bid_size / max(abs(centers[i]-bid_price)/(mid_price*bucket_size), 0.5)

            for ask_price, ask_size in data[currKey][1]:
                ###bucket = int(((ask_price - mid_price)/mid_price )/bucket_size)
                #print(bucket + n_buckets)
                ###if fits(bucket + n_buckets, rows):
                ###    rows[bucket + n_buckets] += ask_size
                #norm += bid_size
                for i in range(len(rows)//2, len(rows)):
                    rows[i] += bid_size / max(abs(centers[i]-bid_price)/(mid_price*bucket_size), 0.5)

            ###print("!!!", currKey, nextKey)

            # poprawne dane - min ask > max bid
            if data[currKey][0][-1][0] <  data[currKey][1][0][0] and data[nextKey][0][-1][0] <  data[nextKey][1][0][0]:
                growths.append(data[currKey][2] < data[nextKey][2])
                rows /= rows.sum() #norm
                X.append(rows)

                #print(rows)
            #print(rows.sum())
            #print(data[keys[i]])
    
    if Globals.debug:
        print("SIZE", np.array(X).shape, np.array(growths).shape)
        print(np.array(X)[0,:])

    return np.array(X, dtype=np.float32), np.array(growths, dtype=np.int)







