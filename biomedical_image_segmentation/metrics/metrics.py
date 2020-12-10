
def accuracy(x, y):

    compare = (x.float() == y.float())
    acc = compare.sum().item()

    ##########################
    # import numpy as np

    # x = x.numpy().astype(np.float16)
    # y = y.numpy().astype(np.float16)

    # compare = np.equal(x, y)
    # acc = np.sum(compare)

    return acc/len(x.flatten())
    