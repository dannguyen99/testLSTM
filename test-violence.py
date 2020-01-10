import cv2
import numpy as np
from keras.applications import ResNet50
from keras.layers import ConvLSTM2D
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences

from utils import BuildModel_basic
from utils.VideoBufferForLSTM import VideoBufferForLSTM


def test(model_path, video_path, perGPU=0.8, size=244, seq_len=10, classes=1, dropout=0):
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = perGPU
    # set_session(tf.Session(config=config))
    # model = BuildModel_basic.build(size=size, seq_len=seq_len,
    #                                dropout=dropout, classes=classes)
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

    # model.load_weights(model_path)
    vid = cv2.VideoCapture(video_path)
    ret, frame = vid.read()
    if not ret:
        print("Video not opened!")
        return
    dataHandler = VideoBufferForLSTM(len_seq=seq_len, startFrame=frame)
    i = 0
    buffer = []
    prvFrame = cv2.resize(frame, (244, 244))
    prvFrame = (prvFrame / 255.).astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    prvFrame = (prvFrame - mean) / std
    while True:
        ret, frame = vid.read()
        if i % 4 == 0:
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
        print("Violence detected!")
    else:
        print("Negative")


if __name__ == '__main__':
    test(model_path='saved-model-fight-PTZ-8frames-42-0.97.hdf5',
         video_path='data/raw_videos/HockeyFights/fi1_xvid.avi', perGPU=0.2)
