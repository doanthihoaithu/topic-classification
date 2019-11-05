from keras.models import Sequential
from keras import layers
from keras.layers import Dropout, Dense


class DNNModel(object):
    __input_dim = None
    __model = None

    def __init__(self, _input_dim):
        self.__input_dim = _input_dim
        model = Sequential()
        node = 512  # number of nodes
        dropout = 100
        n_classes = 23
        n_layers = 4  # number of  hidden layer
        model.add(Dense(node, input_dim=self.__input_dim, activation='relu'))
        model.add(Dropout(dropout))
        for i in range(0, n_layers):
            model.add(Dense(node, input_dim=node, activation='relu'))
            model.add(Dropout(dropout))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.__model = model

    def get_model(self):
        return self.__model


