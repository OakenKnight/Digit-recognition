import time
from tkinter import *

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

MODEL = None
X_TEST = []


class Paint(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()
        self.predicted_number_text = ''
        self.predicted_number = 0
        self.image1 = PIL.Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)

        self.pen_button = Button(self.root, text='draw', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.delete_all_button = Button(self.root, text='clear all', command=self.delete_all)
        self.delete_all_button.grid(row=0, column=1)

        self.save_all_button = Button(self.root, text='save', command=self.save)
        self.save_all_button.grid(row=0, column=2)

        self.predict_button = Button(self.root, text='predict', command=self.predict)
        self.predict_button.grid(row=0, column=3)

        self.c = Canvas(self.root, bg='white', width=200, height=200)
        self.c.grid(row=1, columnspan=4)

        self.setup()
        self.predicted_number_text_label = Label(self.root, text=self.predicted_number_text)
        self.predicted_number_text_label.grid(row=2, columnspan=5)

        self.root.mainloop()

    def save(self):
        print("Saving picture...")
        millis = str(int(round(time.time() * 1000)))
        filename = str(millis) + ".png"
        self.image1.save(filename)
        print("Picture saved: SUCCESS")

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 2
        self.color = self.DEFAULT_COLOR
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        self.line_width = 2
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill='black',
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill="black", width=2)

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def delete_all(self):
        self.c.delete("all")
        self.image1 = PIL.Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)
        self.activate_button(self.delete_all_button)

    def predict(self):
        self.save_image()
        self.predicted_number_text = "Predicted number is: " + str(self.predicted_number)
        self.predicted_number_text_label['text'] = self.predicted_number_text

    def save_image(self):
        print("Saving picture...")
        filename = "number.png"
        self.image1.save(filename)
        print("Picture saved: SUCCESS")
        img = cv2.imread('number.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        mx = (0, 0, 0, 0)  # biggest bounding box so far
        mx_area = 0
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            area = w * h
            if area > mx_area:
                mx = x, y, w, h
                mx_area = area
        x, y, w, h = mx

        roi = img[y:y + h, x:x + w]
        cv2.imwrite('Image_crop.png', roi)

        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)
        cv2.imwrite('Image_cont.png', img)

        image = PIL.Image.open('Image_crop.png')

        new_image = image.resize((28, 28))
        new_image.save('image_28x28.png')

        let = self.prepare_image()

        self.predicted_number = test_prediction(MODEL, X_TEST)

    def prepare_image(self):
        im = PIL.Image.open('image_28x28.png').convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = PIL.Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

        if width > height:  # check which dimension is bigger
            # Width is bigger. Width becomes 20 pixels.
            nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
            if (nheight == 0):  # rare case but minimum is 1 pixel
                nheight = 1
                # resize and sharpen
            img = im.resize((20, nheight), PIL.Image.ANTIALIAS).filter(PIL.ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
            newImage.paste(img, (4, wtop))  # paste resized image on white canvas
        else:
            # Height is bigger. Heigth becomes 20 pixels.
            nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
            if (nwidth == 0):  # rare case but minimum is 1 pixel
                nwidth = 1
                # resize and sharpen
            img = im.resize((nwidth, 20), PIL.Image.ANTIALIAS).filter(PIL.ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
            newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

        # newImage.save("sample.png

        tv = list(newImage.getdata())  # get pixel values

        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        print(tva)
        newImage.save('test.png')
        return tva


def test_prediction(model, x_test):
    img_to_append = cv2.imread('test.png', 0)
    img_to_append = cv2.bitwise_not(img_to_append)

    x_test = np.append(x_test, [img_to_append], axis=0)
    x_test_len = len(x_test)

    predictions = model.predict(x_test)

    print("Test prediction...")
    plt.imshow(x_test[x_test_len - 1], cmap=plt.cm.binary)
    plt.show()
    predicted_number = np.argmax(predictions[x_test_len - 1])
    print("Predicted number is: ", predicted_number)
    print(x_test_len)
    return predicted_number


def save_model():
    print("Saving model..")
    MODEL.save(r'C:\Users\Wheatley\Desktop\ML python\Number Recognition\digit_model')
    print("Model saved!")


def load_model():
    return tf.keras.models.load_model(r'C:\Users\Wheatley\Desktop\ML python\Number Recognition\digit_model')


def nnetwork():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, 20)
    # evaluate(model,x_test,y_test)

    return model, x_test


def evaluate(model, x_test, y_test):
    print("Evaluating...")

    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy: ", accuracy)
    print("Loss: ", loss)


if __name__ == '__main__':
    MODEL, X_TEST = nnetwork()
    Paint()
