from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model

class Model(object):
    def __init__(self, size, learning_rate=0.001):
        self.model = Sequential()
        self.model.add(Dense(1024, activation="relu", input_shape=(size,)))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation="sigmoid"))

        opt = Adam(lr=learning_rate)
        self.model.compile(loss="mse", optimizer=opt,metrics=['accuracy'])
        self.model.summary()
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, epochs=5, batch_size=5):
         self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid))
    
    def evaluate(X_test, y_test, batch_size=128):
        return self.model.evaluate(X_test, y_test, batch_size=batch_size)
        
    def save(self, name):
        self.model.save('{}.h5'.format(name))