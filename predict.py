import cv2
import numpy as np
from keras.applications import ResNet50
from keras.layers import ConvLSTM2D
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences

from utils import BuildModel_basic

seq_len = 10
vid_len = 40
skip = int(vid_len / seq_len)


def build_model(weight_path='weights.h5'):
    size = 244
    learning_rate = 0.0001
    initial_weights = 'glorot_uniform'
    optimizer = (RMSprop, {})
    lstm_conf = (ConvLSTM2D, dict(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False))
    classes = 1
    cnn_arch = ResNet50
    cnn_train_type = 'retrain'
    pre_weights = 'imagenet'
    dropout = 0
    model = BuildModel_basic.build(size=size, seq_len=seq_len, learning_rate=learning_rate,
                                   optimizer_class=optimizer, initial_weights=initial_weights,
                                   cnn_class=cnn_arch, pre_weights=pre_weights, lstm_conf=lstm_conf,
                                   cnn_train_type=cnn_train_type, dropout=dropout, classes=classes)
    model.load_weights(weight_path)
    return model


def predict(video_path, model):
    vid = cv2.VideoCapture(video_path)
    ret, frame = vid.read()
    if not ret:
        print("Video not opened!")
        return
    i = 0
    buffer = []
    prvFrame = cv2.resize(frame, (244, 244))
    prvFrame = (prvFrame / 255.).astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    prvFrame = (prvFrame - mean) / std
    while True:
        ret, frame = vid.read()
        if i % skip == 0:
            frame = cv2.resize(frame, (244, 244))
            frame = (frame / 255.).astype(np.float32)
            frame = (frame - mean) / std
            subtractFrame = prvFrame - frame
            buffer.append(subtractFrame)
            prvFrame = frame
        i += 1
        if len(buffer) == seq_len:
            break
    X = pad_sequences([buffer], maxlen=seq_len, padding='pre', truncating='pre')
    X = np.array(X)
    result = model.predict(X)
    if result[0][0] > 0.5:
        print("Positive")
    else:
        print("Negative")


if __name__ == '__main__':
    model = build_model()
    while True:
        path = raw_input("enter video path to predict: ")
        predict(path, model)
