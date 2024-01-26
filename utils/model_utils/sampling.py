import numpy as np


def sampling_class(X, y, target_class, n, positive_sample=True):
    sample_x = []
    sample_y = []

    cnt = 0
    for i in range(0, len(y)):
        if (positive_sample and y[i] == target_class) or (not positive_sample and y[i] != target_class):
            sample_x.append(X[i])
            sample_y.append(y[i])
            cnt += 1
            if cnt > n:
                break
    return np.asarray(sample_x), np.asarray(sample_x)