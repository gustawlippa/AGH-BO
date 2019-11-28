import pickle


def save(data, filename='log'):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()


def load(filename='log'):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data
