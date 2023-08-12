import random

def train_test_split(data, test_size=0.2, random_state=None):
    """
    Splits the input data into training and testing sets.

    Parameters:
        data (list or array-like): The dataset to be split.
        test_size (float): The proportion of the dataset to include in the test set (default is 0.2).
        random_state (int): Random seed for reproducibility (default is None).

    Returns:
        tuple: A tuple containing the training set and the testing set.
    """

    if not 0 <= test_size <= 1:
        raise ValueError("test_size should be a float between 0 and 1.")

    if random_state is not None:
        random.seed(random_state)

    data_size = len(data)
    test_data_size = int(data_size * test_size)
    test_indices = random.sample(range(data_size), test_data_size)

    train_data = [data[i] for i in range(data_size) if i not in test_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, test_data

# Sample dataset
dataset = list(range(100))

# Split the dataset into training and testing sets with 80% for training and 20% for testing
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

print("Training set:", train_set)
print("Testing set:", test_set)