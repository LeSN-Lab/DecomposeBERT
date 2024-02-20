import numpy as np


def sampling_class(x, y, target_class, num_samples, positive_sample=True):
    sample_x = []
    sample_y = []
    count = 0
    if positive_sample:
        target_class_keys = [i for i in y.keys() if target_class in y[i]]
    else:
        target_class_keys = [i for i in y.keys() if target_class not in y[i]]

    for i in range(num_samples):
        for j in target_class_keys:
            sample_x.append(x[j][i])
            sample_y.append(y[j][i])
    return sample_y, sample_y
