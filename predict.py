from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np


def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img


def predict():
    img = load_image('horse5.png')
    model = load_model('improved_model.h5')
    predict = model.predict(img)
    predict = np.argmax(predict, axis=1)
    print(predict)


if __name__ == '__main__':
    predict()
