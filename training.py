import os
import sys
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import cv2
from sklearn.model_selection import train_test_split,StratifiedKFold

from keras.optimizers import Adam, Adadelta
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda, Dense, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras.utils import Sequence
from keras.preprocessing.image import array_to_img, img_to_array, load_img


# --------------------------- metrics and losses ----------------------------- #


def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


def new_euc_dist(y_true, y_pred):
    val = np.array([3., 1.])
    var = K.constant(value=val)
    return euc_dist_keras(y_true, y_pred) + K.mean(var * K.abs(y_true - y_pred), axis=-1, keepdims=True)


def weighted_euc(y_true, y_pred):
    n = K.shape(y_true)[0]
    _p1300 = K.constant(value=np.array([1300]))
    _p250 = K.constant(value=np.array([250]))
    _p560 = K.constant(value=np.array([560]))
    _p80 = K.constant(value=np.array([80]))

    a = K.reshape(K.cast(K.greater(y_true[:, 0], _p1300), dtype="float32"), (n, 1))
    b = K.reshape(K.cast(K.less(y_true[:, 0], _p250), dtype="float32"), (n, 1))
    c = K.reshape(K.cast(K.greater(y_true[:, 1], _p560), dtype="float32"), (n, 1))
    d = K.reshape(K.cast(K.less(y_true[:, 1], _p80), dtype="float32"), (n, 1))

    binary_x = K.sum(K.concatenate([a, b], axis=1), axis=1, keepdims=True)
    binary_y = K.sum(K.concatenate([c, d], axis=1), axis=1, keepdims=True)
    weights = K.concatenate([binary_x, binary_y], axis=1)
    return K.mean(K.sqrt(K.square(y_true - y_pred)) * weights, axis=-1, keepdims=True)


def uniform_weight_euc(y_true, y_pred):
    n = K.shape(y_true)[0]
    xmid = K.constant(value=np.array([767]))
    ymid = K.constant(value=np.array([431]))
    ones = K.constant(value=np.array([1.]))

    a = K.reshape(K.abs(y_true[:, 0] - xmid) / 1535. + ones, (n, 1))
    b = K.reshape(K.abs(y_true[:, 1] - ymid) / 863. + ones, (n, 1))
    weights = K.concatenate([a, b], axis=1)

    return K.sqrt(K.sum(K.square(y_true - y_pred) * weights, axis=-1, keepdims=True))


def final_loss(y_true, y_pred):
    return (new_euc_dist(y_true, y_pred) + weighted_euc(y_true, y_pred))


def fourth_segment(y_true, y_pred):
    val = np.array([960., 540.])
    var = K.constant(value=val)
    return K.sum(K.cast(K.greater(y_true, val), dtype="float32") * K.abs(y_true - y_pred), axis=-1, keepdims=True)


def second_segment(y_true, y_pred):
    val = np.array([960., 540.])
    var = K.constant(value=val)
    return K.sum(K.cast(K.less_equal(y_true, val), dtype="float32") * K.abs(y_true - y_pred), axis=-1, keepdims=True)


# ---------------------------------------- load files --------------------------------------- #

img_size_y = 100
img_size_x = 200

# load and shuffle filenames
folder = './data'
filenames = os.listdir(folder)
random.shuffle(filenames)
# split into train and validation filenames
n_valid_samples = 900
train_filenames = filenames[n_valid_samples:]
valid_filenames = filenames[:n_valid_samples]
print('n train samples', len(train_filenames))
print('n valid samples', len(valid_filenames))
n_train_samples = len(filenames) - n_valid_samples


class generator(Sequence):

    def __init__(self, folder, filenames, batch_size=16, image_size=100, shuffle=True):
        self.folder = folder
        self.filenames = filenames
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __load__(self, filename):
        return 1, 1

    def __loadpredict__(self, filename):
        return 1

    def __getitem__(self, index):
        # select batch
        files = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        imagesl = []
        imagesr = []
        coords = []
        success = 0
        fail = 0
        for file in files:
            try:
                img = np.array((load_img(os.path.join(self.folder, file), color_mode="grayscale")))
                y = file[11:-4].split("_")
                y = [int(i) for i in y]
                # print(y)
                imagesl.append(img[:, :100])
                imagesr.append(img[:, 100:])
                coords.append(y)
                success += 1
            except:
                fail += 1
        imagesl = np.reshape(imagesl, (-1, 100, 100, 1))
        imagesr = np.reshape(imagesr, (-1, 100, 100, 1))
        coords = np.array(coords)
        return [imagesl, imagesr], coords

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    def __len__(self):
        return int(len(self.filenames) / self.batch_size)



# --------------------------------------- model ----------------------------------------- #

input_l = Input((img_size_y, 100, 1), name='imgl')
input_r = Input((img_size_y, 100, 1), name='imgr')

def downblock(x, d=4, channels = 4):
    c1 = Conv2D(channels, (3, 3), activation='relu', padding='same') (x)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(channels, (3, 3), activation='relu', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    d1 = Conv2D(channels, (1, 1), activation='relu', padding='same') (x)
    c1 = Add()([c1, d1])
    c1 = BatchNormalization()(c1)
    c1 = MaxPooling2D((d, d)) (c1)
    return c1


left = downblock(input_l, d=4, channels = 4)
right = downblock(input_r, d=4, channels = 4)

left = downblock(left, d=2, channels=8)
right = downblock(right, d=2, channels=8)

left = downblock(left, d=2, channels=16)
right = downblock(right, d=2, channels=16)

#left = downblock(left, d=2, channels=32)
#right = downblock(right, d=2, channels=32)

left = Flatten()(left)
left = Dense(128)(left)
right = Flatten()(right)
right = Dense(128)(right)

out = concatenate([left, right])
out = Dense(128)(out)
out = Dense(32)(out)
out = Dense(2)(out)

opt = Adam(lr=0.0008)
model = Model(inputs=[input_l, input_r], outputs=[out])
model.compile(loss=new_euc_dist, optimizer=opt, metrics=[euc_dist_keras])


# ------------------------------------------ training ---------------------------------------------- #

save_model_name = "eye.model"

model_checkpoint = ModelCheckpoint(save_model_name, save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_euc_dist_keras', mode = 'min',factor=0.5, patience=5, min_lr=0.0000001, verbose=1)


train_gen = generator(folder, train_filenames, batch_size=16, image_size=100, shuffle=True)
valid_gen = generator(folder, valid_filenames, batch_size=16, image_size=100, shuffle=True)

history = model.fit_generator(train_gen, validation_data=valid_gen, callbacks=[reduce_lr, model_checkpoint], epochs=50)


fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_score.plot(history.epoch, history.history["euc_dist_keras"], label="Train score")
ax_score.plot(history.epoch, history.history["val_euc_dist_keras"], label="Validation score")
ax_score.legend()


# ------------------------------------------------- X ----------------------------------------------- #