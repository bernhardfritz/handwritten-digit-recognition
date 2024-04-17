import gzip
import numpy as np

from neural_network import NeuralNetwork

training_set_images = "train-images-idx3-ubyte.gz"
training_set_labels = "train-labels-idx1-ubyte.gz"
test_set_images = "t10k-images-idx3-ubyte.gz"
test_set_labels = "t10k-labels-idx1-ubyte.gz"


def read_images_file(filename):
    with gzip.open(filename, "rb") as f:
        magic_number = int.from_bytes(f.read(4))
        assert magic_number == 2051, "magic_number should be 2051"
        number_of_images = int.from_bytes(f.read(4))
        number_of_rows = int.from_bytes(f.read(4))
        number_of_columns = int.from_bytes(f.read(4))
        return [
            np.reshape(
                np.frombuffer(
                    f.read(number_of_rows * number_of_columns), dtype=np.uint8
                ).astype(np.float64)
                / 255,
                (number_of_rows * number_of_columns, 1),
            )
            for _ in range(number_of_images)
        ]


def label_to_one_hot(i):
    one_hot = np.zeros((10, 1))
    one_hot[i] = 1
    return one_hot


def read_labels_file(filename):
    with gzip.open(filename, "rb") as f:
        magic_number = int.from_bytes(f.read(4))
        assert magic_number == 2049, "magic_number should be 2049"
        number_of_items = int.from_bytes(f.read(4))
        return np.frombuffer(f.read(number_of_items), dtype=np.uint8)


training_images = read_images_file(training_set_images)
training_labels = read_labels_file(training_set_labels)
test_images = read_images_file(test_set_images)
test_labels = read_labels_file(test_set_labels)
neural_network = NeuralNetwork([training_images[0].size, 16, 16, 10])
training_data = [
    (image, label_to_one_hot(label))
    for image, label in zip(training_images, training_labels)
]
test_data = list(zip(test_images, test_labels))
neural_network.train(training_data, 10, 100, 0.25, test_data)
