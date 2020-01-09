from keras import Input
from keras.layers import Dense, Flatten, Dropout, ConvLSTM2D, BatchNormalization, Activation
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import MaxPooling2D
from keras.applications import ResNet50
from keras.layers import Dense
from keras.models import Model


def build(size, seq_len , learning_rate=0.0001 , classes = 1, dropout = 0.0):
    input_layer = Input(shape=(seq_len, size, size, 3))
    cnn = ResNet50(weights='imagenet', include_top=False,input_shape =(size, size, 3))
    for layer in cnn.layers:
       layer.trainable = True

    cnn = TimeDistributed(cnn)(input_layer)

    lstm = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False)(cnn)
    lstm = MaxPooling2D(pool_size=(2, 2))(lstm)
    flat = Flatten()(lstm)

    flat = BatchNormalization()(flat)
    flat = Dropout(dropout)(flat)
    linear = Dense(1000)(flat)

    relu = Activation('relu')(linear)
    linear = Dense(256)(relu)
    linear = Dropout(dropout)(linear)
    relu = Activation('relu')(linear)
    linear = Dense(10)(relu)
    linear = Dropout(dropout)(linear)
    relu = Activation('relu')(linear)

    activation = 'sigmoid'
    loss_func = 'binary_crossentropy'

    if classes > 1:
        activation = 'softmax'
        loss_func = 'categorical_crossentropy'
    predictions = Dense(classes,  activation=activation)(relu)

    model = Model(inputs=input_layer, outputs=predictions)
    optimizer = RMSprop(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_func,metrics=['acc'])

    print(model.summary())

    return model
