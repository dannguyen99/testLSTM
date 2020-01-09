from keras.preprocessing.sequence import pad_sequences
import numpy as np
import cv2


class VideoBufferForLSTM:
    def __init__(self, len_seq, startFrame):
        self.buffer = []
        self.prvFrame = cv2.resize(startFrame, (244, 244))
        self.prvFrame = (self.prvFrame / 255.).astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.prvFrame = (self.prvFrame - mean) / std

        self.len_seq = len_seq

    def buffer_for_input(self, frame):  # enough 20 frames, then create input for ConvLSTM
        frame = cv2.resize(frame, (244, 244))
        frame = (frame / 255.).astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        frame = (frame - mean) / std
        subtractFrame = self.prvFrame - frame
        print(subtractFrame.shape, len(self.buffer))
        self.prvFrame = frame
        self.buffer.append(subtractFrame)
        if len(self.buffer) == (self.len_seq + 1):
            del self.buffer[0]
            X = pad_sequences([self.buffer], maxlen=self.len_seq, padding='pre', truncating='pre')
            # del self.buffer[0:10]
            # print("X=",X.shape)
            return True, np.array(X)
        return False, None
