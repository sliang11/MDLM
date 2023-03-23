import pickle


def load_data(fn):
    with open(fn + '.pickle', "rb") as f:
        x = pickle.load(f)
    return x


def save_data(data, fn):
    with open(fn + '.pickle', "wb") as f:
        pickle.dump(data, f)