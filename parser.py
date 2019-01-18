

def parse(file_name):
    

    data = {}
    for line in open(file_name):
        line = line.split()

        key = line[0] + " " + line[1]

        bid_prices = []
        ask_prices = []

        for i in range(3, len(line), 2):
            if line[i] == "ASK":
                i = i+1
                break
            bid_prices.append((float(line[i]), float(line[i+1])))
        while i < len(line):
            ask_prices.append((float(line[i]), float(line[i+1])))
            i += 2

        bid_prices.sort()
        ask_prices.sort()

        mid_price = (ask_prices[0][0] - bid_prices[-1][0])/2 + bid_prices[-1][0]
        inbalance = (ask_prices[0][1] - bid_prices[-1][1])/(bid_prices[-1][1] + ask_prices[0][1])

        data[key] = (bid_prices, ask_prices, mid_price, inbalance)

    return data

