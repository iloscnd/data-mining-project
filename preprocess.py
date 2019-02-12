import numpy as np 

from utils import Globals

def fits(index, arr):
    return index >= 0 and index < len(arr)


def get_XY(data, n_buckets=5, bucket_size=0.05 , omit_no_change=True):
    
    keys = list(data.keys())
    keys.sort()

    
    bad0 = 0
    bad1 = 0

    growths = []
    X = []

    for i, curr in enumerate(keys[:-1]):
        currKey = curr
        nextKey = keys[i+1]

        currday, currh = currKey.split()
        nextday, nexth = nextKey.split()

        currh = int(currh[:4])
        nexth = int(nexth[:4])

        
        # odrzucamy dane z początku i końca dnia jako mało wiarygodne
        if currh < 900 or currh > 1600 or currh + 1 != nexth:
            bad0 += 1
            continue

        if not omit_no_change or data[nextKey][2] != data[currKey][2]:
            rows0 = np.zeros(2*n_buckets)
            rows = np.zeros(2*n_buckets)
            
            mid_price = data[currKey][2]

            # środki przedziałów
            centers = (np.arange(2*n_buckets) - n_buckets + 0.5)*bucket_size*mid_price + mid_price
            
            for bid_price, bid_size in reversed(data[currKey][0]):
                
                # do którego przedziału/kubełka trafia słupek
                bucket = int(( n_buckets*bucket_size -  (mid_price-bid_price)/mid_price)/bucket_size)
                
                if fits(bucket, rows):  # słupek jest w jednym z przedziałów które bierzemy
                    rows0[bucket] += bid_size  # alternatywny pomysł - wliczanie po prostu wysokości słupków w kubełku
                    
                    norm = 0.  # normalizacja wag z jakimi wliczamy słupek do różnych przedziałów
                    for i in range(len(rows)//2):
                        norm += 1. / max(1.*(abs(centers[i]-bid_price)/(mid_price*bucket_size))**2., 0.1)
                        # na początku liczymy sumę wag w celu późniejszej normalizacji
                    for i in range(len(rows)//2):
                        rows[i] += bid_size* ( (1./max((abs(centers[i]-bid_price)/(mid_price*bucket_size))**2., 0.1)) / norm )
                        # wliczamy do każdego przedziału wysokość słupka pomnożoną przez wagę 
                        # proporcjonalną do kwadratu odwrotności odległości do środka tego przedziału 
                        # (ograniczoną dla słupków bardzo bliskich centrum przedziału)
                        
            for ask_price, ask_size in data[currKey][1]:

                # do którego przedziału/kubełka trafia słupek
                bucket = int(((ask_price - mid_price)/mid_price )/bucket_size)
                
                if fits(bucket + n_buckets, rows):  # słupek jest w jednym z przedziałów które bierzemy
                    rows0[bucket + n_buckets] += ask_size  # alternatywny pomysł - wliczanie po prostu wysokości słupków w kubełku
                    norm = 0.
                    for i in range(len(rows)//2, len(rows)):
                        norm += 1. / max(1.*(abs(centers[i]-ask_price)/(mid_price*bucket_size))**2., 0.1)
                        # na początku liczymy sumę wag w celu późniejszej normalizacji
                    for i in range(len(rows)//2, len(rows)):
                        rows[i] += ask_size* ( (1./max(1.*(abs(centers[i]-ask_price)/(mid_price*bucket_size))**2., 0.1)) / norm )
                        # wliczamy do każdego przedziału wysokość słupka pomnożoną przez wagę 
                        # proporcjonalną do kwadratu odwrotności odległości do środka tego przedziału 
                        # (ograniczoną dla słupków bardzo bliskich centrum przedziału)

            
            # poprawne dane - min ask > max bid  -- czasem zdarzało się inaczej
            if data[currKey][0][-1][0] <  data[currKey][1][0][0] and data[nextKey][0][-1][0] <  data[nextKey][1][0][0]:
                growths.append(data[currKey][2] < data[nextKey][2])


                rows /= rows.sum() # normalizacja - wektor wliczonych ważonych słupków dla wszystkich
                                   # przedziałów sumuje się łącznie do 1ki 
                
                rows0 /= rows0.sum()
                ###print("!", rows, rows0)
                X.append(rows)
            else:
                bad1 += 1

            
    
    if Globals.debug:
        print("DATA LEN: ", len(data.keys()))
        print("BAD: ", bad0, bad1)
        print("SIZE", np.array(X).shape, np.array(growths).shape)
        print(np.array(X)[0,:])

    X = np.array(X, dtype=np.float32)
    Y = np.array(growths, dtype=np.int)

    test_size =  len(Y)//8

    return (X[test_size:], Y[test_size:]), (X[:test_size], Y[:test_size])







