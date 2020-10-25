import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

MNIST = tf.keras.datasets.mnist
EPOCHS = 20


def main():
    (x_train, y_train), (x_test, y_test) = MNIST.load_data()

    # We can see what is fourth element in x_train
    # plt.imshow(x_train[4], cmap=plt.cm.binary)
    # plt.show()

    # print(x_train[0])
    # Because data is in range from 0-255 we need to normalize it
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # print(x_train[0])
    print(len(x_train[0]))  # = 28 the size of the x_train[i] is 28x28 so we will need to flatten it to 1x784

    model = tf.keras.models.Sequential()  # Sequential is a simple feed forward model
    model.add(tf.keras.layers.Flatten())  # this will take 28x28 and make it 1x784
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 means there is 128 neurons in one layer
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 means there is 128 neurons in one layer
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 means there is 128 neurons in one layer

    # because we have more than 2 outputs we are gonna use softmax
    # if we had binary output, we could've used sigmoid function
    # 10 is because we have 10 different numbers to recognize
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # there are 2 loss cathegories scce and cce
    # in example with 5 classes:
    # cce:  the one-hot target may be [0,1,0,0,0] and the model may predict [.2,.5,.1,.1,.1] (probably right)
    # scce: the target index may be [1] and the model may predict [.5]
    # scce will be used when classes are mutually exclusive

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # how will we calculate error to minimize the loss
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, EPOCHS)

    print("Evaluating...")

    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy: ", accuracy)
    print("Loss: ", loss)

    print("Saving model..")
    model.save(r'C:\Users\Wheatley\Desktop\ML python\Number Recognition\digit_model')
    print("Model saved!")

    # Loading the model
    # new_model = tf.keras.models.load_model(r'C:\Users\Wheatley\Desktop\ML python\Number Recognition\digit_model')

    predictions = model.predict(x_test)

    print("Test prediction...")
    plt.imshow(x_test[4], cmap=plt.cm.binary)
    plt.show()

    print("Predicted number is: ", np.argmax(predictions[4]))


if __name__ == '__main__':
    main()
