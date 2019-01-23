import numpy as np

def cut(index, arr):
    l,r = 0, len(arr) - 1
    return min(r, max(index,l))


def get_XY(data, n_buckets=5, bucket_size=0.05 , omit_no_change=True):
    
    keys = list(data.keys())
    keys.sort()

    growths = []
    X = []

    

    for i, curr in enumerate(keys[:-1]):
        currKey = curr
        nextKey = keys[i+1]

        if not omit_no_change or data[nextKey][2] != data[currKey][2]:
            rows = np.zeros(2*n_buckets)
            mid_price = data[currKey][2]
            #print(mid_price)
            for bid_price, bid_size in reversed(data[currKey][0]):
                bucket = int(( n_buckets*bucket_size -  (mid_price-bid_price)/mid_price)/bucket_size)
                #print(bucket)
                if bucket in range(len(rows)):
                    rows[bucket] += bid_size

            for ask_price, ask_size in data[currKey][1]:
                bucket = int(((ask_price - mid_price)/mid_price )/bucket_size)
                #print(bucket + n_buckets)
                if bucket + n_buckets in range(len(rows)):
                    rows[bucket + n_buckets] += ask_size

            ###print("!!!", currKey, nextKey)

            # poprawne dane - min ask > max bid
            if data[currKey][0][-1][0] <  data[currKey][1][1][0] and data[nextKey][0][-1][0] <  data[nextKey][1][1][0]:
                #print "A"
                growths.append(data[currKey][2] < data[nextKey][2])
                rows /= rows.sum()
                X.append(rows)

            #print(rows)
            #print(rows.sum())
            #print(data[keys[i]])
    
    print "SIZE", np.array(X).shape, np.array(growths).shape
    #print np.array(X)[0,:]
    return np.array(X), np.array(growths)







