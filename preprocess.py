import numpy as np




def get_XY(data, n_buckets=5, bucket_size=0.05 , omit_no_change=True):
    
    keys = list(data.keys())
    keys.sort()

    growths = []
    X = []

    

    for i, curr in enumerate(keys[1:]):
        if not omit_no_change or data[keys[i]][2] != data[curr][2]:
            rows = np.zeros(2*n_buckets)
            mid_price = data[keys[i]][2]
            #print(mid_price)
            for bid_price, bid_size in reversed(data[keys[i]][0]):
                if (mid_price-bid_price)/mid_price >= n_buckets*bucket_size:
                    break
                #print( (mid_price-bid_price)/mid_price )
                bucket = int(( n_buckets*bucket_size -  (mid_price-bid_price)/mid_price)/bucket_size)
                #print(bucket)
                rows[bucket] += bid_size

            for ask_price, ask_size in data[keys[i]][1]:
                if (ask_price - mid_price)/mid_price >= n_buckets*bucket_size:
                    break
                
                #print( (ask_price - mid_price)/mid_price )
                bucket = int(((ask_price - mid_price)/mid_price )/bucket_size)
                #print(bucket + n_buckets)
                rows[bucket + n_buckets] += ask_size

            growths.append(data[keys[i]][2] < data[curr][2])
            
            rows /= rows.sum()
            X.append(rows)

            #print(rows)
            #print(rows.sum())
            #print(data[keys[i]])
    
    return np.array(X), np.array(growths)







