from keras.models import Sequential #type:ignore
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout #type:ignore
from keras.optimizers import Adam #type:ignore

def build_model(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

def compile_model(model, lr=1e-4, epochs=24):
    optimizer = Adam(learning_rate=lr, decay=lr/epochs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
