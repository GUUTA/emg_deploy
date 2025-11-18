import numpy as np


def representative_dataset():
    for i in range(200):
        data = np.load(f"../test_data/calib_{i}.npy").astype(np.float32)
        yield [data]
