

def lprint(iterable, header=None):
    if header:
        print("\t".join([str(elem) for elem in header]))
    for elem in iterable:
        if type(elem) == list:
            print("\t".join([str(e) for e in elem]))
        else:
            print(str(elem))

def read_config():
    with open("config.txt", "r") as f:
        config_dict = eval(f.readline())
    f.close()
    return config_dict
