import os

import pandas as pd
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.layers import ConvLSTM2D
from keras.optimizers import RMSprop

from utils import BuildModel_basic, DatasetBuilder

# parameter
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
batch_size = 2
fix_lens = 10
datasets_frames = "data/raw_frames"
dataset_name = 'hocky'
dataset_videos = dict(dataset="data/raw_videos/HockeyFights")
force = True
epochs = 30
patience_es = 15
patience_lr = 5
batch_epoch_ratio = 0.5
res_path = "results"


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loss = []
        self.test_acc = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, batch_size=2, verbose=0)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def get_generators(dataset_name, dataset_videos, datasets_frames, fix_len, figure_size, force, classes=1, use_aug=False,
                   use_crop=True, crop_dark=None):
    train_path, valid_path, test_path, train_y, valid_y, test_y, avg_length = DatasetBuilder.createDataset(
        dataset_videos, datasets_frames, fix_len, force=force)

    if fix_len is not None:
        avg_length = fix_len
    crop_x_y = None
    if crop_dark:
        crop_x_y = crop_dark[dataset_name]

    len_train, len_valid = len(train_path), len(valid_path)
    train_gen = DatasetBuilder.data_generator(train_path, train_y, batch_size, figure_size, avg_length, use_aug=use_aug,
                                              use_crop=use_crop, crop_x_y=crop_x_y, classes=classes)
    validate_gen = DatasetBuilder.data_generator(valid_path, valid_y, batch_size, figure_size, avg_length,
                                                 use_aug=False, use_crop=False, crop_x_y=crop_x_y, classes=classes)
    test_x, test_y = DatasetBuilder.get_sequences(test_path, test_y, figure_size, avg_length, crop_x_y=crop_x_y,
                                                  classes=classes)

    return train_gen, validate_gen, test_x, test_y, avg_length, len_train, len_valid


def train():
    train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(dataset_name,
                                                                                            dataset_videos,
                                                                                            datasets_frames, fix_lens,
                                                                                            size,
                                                                                            force=force,
                                                                                            classes=classes,
                                                                                            use_aug=True,
                                                                                            use_crop=True)
    test_history = TestCallback((test_x, test_y))
    # create model
    model = BuildModel_basic.build(size=size, seq_len=seq_len, learning_rate=learning_rate,
                                   optimizer_class=optimizer, initial_weights=initial_weights,
                                   cnn_class=cnn_arch, pre_weights=pre_weights, lstm_conf=lstm_conf,
                                   cnn_train_type=cnn_train_type, dropout=dropout, classes=classes)
    # fit model
    history = model.fit_generator(
        steps_per_epoch=int(float(len_train) / float(batch_size * batch_epoch_ratio)),
        generator=train_gen,
        epochs=epochs,
        validation_data=validate_gen,
        validation_steps=int(float(len_valid) / float(batch_size)),
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience_es, ),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr, min_lr=1e-8, verbose=1),
                   test_history
                   ]
    )

    # save model after train
    model.save("model.h5")
    model.save_weights('weights.h5')

    # save model history
    history_to_save = history.history
    history_to_save['test accuracy'] = test_history.test_acc
    history_to_save['test loss'] = test_history.test_loss

    result = dict(dataset=dataset_name, cnn_train=cnn_train_type,
                  cnn=cnn_arch.__name__, lstm=lstm_conf[0].__name__, epochs=epochs,
                  learning_rate=learning_rate, batch_size=batch_size, dropout=dropout,
                  optimizer=optimizer[0].__name__, initial_weights=initial_weights, seq_len=seq_len)

    model_name = ""
    for k, v in result.items():
        model_name = model_name + "_" + str(k) + "-" + str(v).replace(".", "d")
    model_path = os.path.join(res_path, model_name)
    pd.DataFrame(history_to_save).to_csv(model_path + "_train_results.csv")
    result['validation loss'] = min(history.history['val_loss'])
    result['validation accuracy'] = max(history.history['val_acc'])
    result['last validation loss'] = history.history['val_loss'][-1]
    result['last validation accuracy'] = history.history['val_acc'][-1]

    result['train accuracy'] = max(history.history['acc'])
    result['train loss'] = min(history.history['loss'])
    result['last train accuracy'] = history.history['acc'][-1]
    result['last train loss'] = history.history['loss'][-1]

    result['test accuracy'] = max(test_history.test_acc)
    result['test loss'] = min(test_history.test_loss)
    result['last test accuracy'] = test_history.test_acc[-1]
    result['last test loss'] = test_history.test_loss[-1]

    result['final lr'] = history.history['lr'][-1]
    result['total epochs'] = len(history.history['lr'])
    pd.DataFrame(result).to_csv("results_datasets.csv")
    return result


def predict():
    train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(dataset_name,
                                                                                            dataset_videos,
                                                                                            datasets_frames, fix_lens,
                                                                                            size,
                                                                                            force=force,
                                                                                            classes=classes,
                                                                                            use_aug=True,
                                                                                            use_crop=True)
    model = BuildModel_basic.build(size=size, seq_len=seq_len, learning_rate=learning_rate,
                                   optimizer_class=optimizer, initial_weights=initial_weights,
                                   cnn_class=cnn_arch, pre_weights=pre_weights, lstm_conf=lstm_conf,
                                   cnn_train_type=cnn_train_type, dropout=dropout, classes=classes)

    y = model.predict_generator(generator=train_gen)
    print(y)


if __name__ == '__main__':
    predict()
