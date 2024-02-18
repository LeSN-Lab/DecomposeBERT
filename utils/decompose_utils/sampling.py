import numpy as np


def sampling_class(x, y, target_class, num_label, positive_sample=True):
    sample_x = []
    sample_y = []

    count = 0
    for i in range(len(y)):
        if (positive_sample and y[i] == target_class) or (
            not positive_sample and y[i] != target_class
        ):
            sample_x.append(x[i])
            sample_y.append(y[i])
            count += 1
            if count > num_label:
                break
    return np.asarray(sample_x), np.asarray(sample_x)
