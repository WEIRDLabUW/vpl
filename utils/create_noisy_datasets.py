import numpy as np
import pickle
original_dataset = "data/test_maze_dataset"


for noise_ratio in [0.0, 0.1, 0.25, 0.75]:
    with open(original_dataset, "rb") as f:
        dataset = pickle.load(f)
    updated_path = original_dataset + f"_noise{noise_ratio}"
    labels = dataset["labels"]
    mask = np.argwhere(np.random.rand(*labels.shape) < noise_ratio)
    labels[mask] = 1 - labels[mask]
    dataset["labels"] = labels
    with open(updated_path, "wb") as f:
        pickle.dump(dataset, f)