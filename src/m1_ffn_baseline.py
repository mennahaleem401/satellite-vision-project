import numpy as np
import os
from PIL import Image

def importImageData(dataset_path, max_images=500):
    """
    Loads images from dataset directory.

    Args:
        dataset_path (str): Path to dataset root folder.
        max_images (int): Maximum number of images to load.

    Returns:
        images (list): List of PIL images.
        labels (list): Corresponding class labels.
    """
    images = []
    labels = []

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    count = 0
    classes = os.listdir(dataset_path)

    for class_folder in classes:
        class_path = os.path.join(dataset_path, class_folder)

        if not os.path.isdir(class_path):
            continue

        for image_file in os.listdir(class_path):
            if count >= max_images:
                break

            image_path = os.path.join(class_path, image_file)

            try:
                image = Image.open(image_path).convert("RGB").resize((64, 64))
                images.append(image)
                labels.append(class_folder)
                count += 1

            except Exception as e:
                print(f"Skipping corrupted image: {image_path} | Error: {e}")

        if count >= max_images:
            break

    return images, labels

def preprocessData(images, labels):
    """
    Preprocess images:
    - Normalize
    - Flatten
    - Encode labels (one-hot)

    Returns:
        X (np.array): Feature matrix
        y (np.array): One-hot labels
        num_classes (int)
    """
    # Normalize + flatten
    X = np.array([np.array(img) / 255.0 for img in images])
    X = X.reshape(X.shape[0], -1)

    # Stable class ordering
    classes = sorted(list(set(labels)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    y_indices = np.array([class_to_idx[label] for label in labels])

    # One-hot encoding
    num_classes = len(classes)
    y = np.zeros((len(y_indices), num_classes))
    y[np.arange(len(y_indices)), y_indices] = 1

    return X, y, num_classes

def initialize_parameters(input_size, h1, h2, output_size):
    """
    Initialize weights and biases.
    """
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
    """Applies ReLU activation"""
    return np.maximum(0, Z)


def sigmoid(Z):
    """Applies Sigmoid activation"""
    return 1 / (1 + np.exp(-Z))


def softmax(Z):
    """Applies Softmax activation"""
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward(X, params):
    """
    Performs forward propagation.

    Returns:
        A3 (np.array): Output probabilities
    """
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)

    Z3 = A2 @ params["W3"] + params["b3"]
    A3 = softmax(Z3)

    return A3


def categorical_cross_entropy(y_true, y_pred):
    """
    Computes categorical cross-entropy loss.
    """
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss


def demo_activation():
    """
    Demonstrates ReLU and Sigmoid outputs.
    """
    sample = np.array([[1.0, -2.0, 3.0]])

    print("\nActivation Demo:")
    print("Input:", sample)
    print("ReLU:", relu(sample))
    print("Sigmoid:", sigmoid(sample))


def main():
    dataset_path = r"data"

    print("Loading data...")
    images, labels = importImageData(dataset_path)

    print("Preprocessing data...")
    X, y, num_classes = preprocessData(images, labels)

    print("X shape:", X.shape)
    print("Number of classes:", num_classes)

    print("Initializing model...")
    input_size = X.shape[1]
    params = initialize_parameters(input_size, 128, 64, num_classes)

    print("Running forward pass...")
    y_pred = forward(X, params)

    print("Calculating loss...")
    loss = categorical_cross_entropy(y, y_pred)

    print("\nFinal Loss:", loss)

    demo_activation()


if __name__ == "__main__":
    main()