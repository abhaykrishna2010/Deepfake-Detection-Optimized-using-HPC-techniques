import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical #type:ignore
from image_processing import convert_to_ela_image

def prepare_image(image_path, image_size=(128, 128)):
    return np.array(convert_to_ela_image(image_path, 91).resize(image_size)).flatten() / 255.0

def load_dataset(path_real, path_fake):
    X = []  # ELA converted images
    Y = []  # 0 for fake, 1 for real
    for dirname, _, filenames in os.walk(path_real):
        for filename in filenames:
            if filename.endswith(('jpg', 'png')):
                full_path = os.path.join(dirname, filename)
                X.append(prepare_image(full_path))
                Y.append(1)

    for dirname, _, filenames in os.walk(path_fake):
        for filename in filenames:
            if filename.endswith(('jpg', 'png')):
                full_path = os.path.join(dirname, filename)
                X.append(prepare_image(full_path))
                Y.append(0)
    
    X = np.array(X).reshape(-1, 128, 128, 3)
    Y = to_categorical(Y, 2)
    return train_test_split(X, Y, test_size=0.2, random_state=5)
