from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


def preprocess_data(data):
    data_norm = data.astype('float32')
    data_norm = data_norm / 255.0
    return data_norm


def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def define_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model():
    trainX, trainY, testX, testY = load_dataset()
    trainX = preprocess_data(trainX)
    testX = preprocess_data(testX)
    model = define_model()
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_iterator = data_generator.flow(trainX, trainY, batch_size=64)
    steps = int(trainX.shape[0] / 64)
    logs = model.fit(train_iterator, steps_per_epoch=steps, epochs=200, validation_data=(testX, testY),
                                  verbose=0)
    model.save("improved_model.h5")
    _, acc = model.evaluate(testX, testY, verbose=0)
    print("最后的正确率为：%.3f" % (acc * 100.0))
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(logs.history['loss'], color='blue', label='train')
    pyplot.plot(logs.history['val_loss'], color='red', label='test')
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(logs.history['accuracy'], color='blue', label='train')
    pyplot.plot(logs.history['val_accuracy'], color='red', label='test')
    pyplot.tight_layout()
    pyplot.savefig('4VGGblock_plot_improve_batch.png')
    pyplot.close()


if __name__ == '__main__':
    evaluate_model()
