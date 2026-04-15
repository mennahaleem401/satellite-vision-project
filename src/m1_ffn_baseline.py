import numpy as np
import os
from PIL import Image


def importImageData(dataset_path, max_images=500):
    images = []
    labels = []

    count = 0
    classes = os.listdir(dataset_path)

    for class_folder in classes:
        class_path = os.path.join(dataset_path, class_folder)

        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                if count >= max_images:
                    break

                image_path = os.path.join(class_path, image_file)
                image = Image.open(image_path).resize((64, 64))

                images.append(image)
                labels.append(class_folder)

                count += 1

        if count >= max_images:
            break

    return images, labels

def preprocessData(images, labels):
    import numpy as np

    # normalize + flatten
    X = np.array([np.array(img) / 255.0 for img in images])
    X = X.reshape(X.shape[0], -1)

    # label encoding manually
    classes = list(set(labels))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels])

    # one-hot
    num_classes = len(classes)
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1

    return X, y_onehot, num_classes

def initialize_parameters(input_size, h1, h2, output_size):
    np.random.seed(42)

    params = {
        "W1": np.random.randn(input_size, h1) * 0.01,
        "b1": np.zeros((1, h1)),
        "W2": np.random.randn(h1, h2) * 0.01,
        "b2": np.zeros((1, h2)),
        "W3": np.random.randn(h2, output_size) * 0.01,
        "b3": np.zeros((1, output_size)),
    }

    return params

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)

    Z3 = A2 @ params["W3"] + params["b3"]
    A3 = softmax(Z3)

    return A3

def categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

def demo_activation():
    sample = np.array([[1.0, -2.0, 3.0]])

    print("ReLU:", relu(sample))
    print("Sigmoid:", sigmoid(sample))

def main():
    dataset_path = r"satellite-vision-project\data"
    images, labels = importImageData(dataset_path)
    X, y, num_classes = preprocessData(images, labels)
    input_size = X.shape[1]
    params = initialize_parameters(input_size, 128, 64, num_classes)
    y_pred = forward(X, params)
    loss = categorical_cross_entropy(y, y_pred)
    print("Loss:", loss)
    demo_activation()


if __name__ == "__main__":
    main()