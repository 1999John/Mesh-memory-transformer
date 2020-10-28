import numpy as np
import tensorflow as tf

def create_datasets_1000(batch_size):
    train_img,train_cap = np.load("/home/mist/Mesh_memory/data/resnet101_img_train_2.npy"),np.load("/home/mist/Mesh_memory/data/resnet101_cap_train_2.npy")
    # /home/mist/Mesh_memory/data/resnet101_cap_train_2.npy
    dataset = tf.data.Dataset.from_tensor_slices((train_img,train_cap))
    dataset = dataset.shuffle(1000).batch(batch_size)

    return dataset

def get_train(batch_size):
    train_img, train_cap = np.load("/home/mist/Mesh_memory/data/resnet101_img_train_2.npy"), np.load(
        "/home/mist/Mesh_memory/data/resnet101_cap_train_2.npy")
    return train_img,train_cap