from sklearn.datasets import fetch_openml
import numpy as np
import os

def download_mnist(target_folder='data'):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    print("MNIST downloading...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    # X: images (70000, 784), y: labels (0-9)
    X, y = mnist["data"], mnist["target"].astype(int)
    
    # Saving in numpy format
    np.save(f'{target_folder}/X_mnist.npy', X)
    np.save(f'{target_folder}/y_mnist.npy', y)
    print("Data saved in /data!")

download_mnist()