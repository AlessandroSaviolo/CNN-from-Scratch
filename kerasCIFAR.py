from inout import plot_learning_curve, plot_accuracy_curve, plot_sample, load_cifar, preprocess, plot_histogram
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras import Model
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt


class History(Callback):
    def __init__(self, model, validation_images, validation_labels):
        self.model = model
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.accuracy = [0]
        self.loss = [5]
        self.val_accuracy = [0]
        self.val_loss = [5]

    def on_batch_end(self, batch, logs={}):
        scores = self.model.evaluate(
            self.validation_images,
            self.validation_labels,
            verbose=0
        )
        print('\n', scores, '\n')
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_loss.append(scores[0])
        self.val_accuracy.append(scores[1])


def train(model, train_images, train_labels, validation_images, validation_labels, batch_size, num_epochs, learning_rate, verbose):
    opt = SGD(lr=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])         # compile model

    history = History(model, validation_images, validation_labels)                              # train model
    model.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[history]
    )

    if verbose:
        plot_learning_curve(history.loss)
        plot_accuracy_curve(history.accuracy, history.val_accuracy)


def evaluate(model):
    scores = model.evaluate(test_images, test_labels, verbose=1)            # evaluate model
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def predict(model, image_idx):
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4']          # names of layers from which we will take the output
    num_features = 4                                            # number of feature maps to display per layer

    dataset = load_cifar()
    dataset['test_images'] = np.moveaxis(dataset['test_images'], 1, 3)
    image = dataset['test_images'][image_idx]
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    pred = np.argmax(model.predict(image))

    plot_sample(dataset['test_images'][image_idx], classes[dataset['test_labels'][image_idx]], classes[pred])

    # extracting the output and appending to outputs
    feature_maps = []
    for name in layer_names:
        tmp_model = Model(inputs=model.input, outputs=model.get_layer(name).output)
        feature_maps.append(tmp_model.predict(image))

    fig, ax = plt.subplots(nrows=len(feature_maps), ncols=num_features, figsize=(20, 20))
    for i in range(len(feature_maps)):
        for z in range(num_features):
            ax[i][z].imshow(feature_maps[i][0, :, :, z])
            ax[i][z].set_title(layer_names[i])
            ax[i][z].set_xticks([])
            ax[i][z].set_yticks([])
    plt.savefig('feature_maps.png')


def plot_weights(model):
    for layer in model.layers:
        if 'conv' in layer.name:
            weights, _ = layer.get_weights()
            plot_histogram(layer.name, np.reshape(weights, -1))


if __name__ == '__main__':

    classes = [                                                             # CIFAR-10 classes
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    num_epochs = 10                                                        # hyper parameters
    learning_rate = 0.005
    batch_size = 100
    lam = 0.01
    verbose = 1

    print('\n--- Loading mnist dataset ---')                                # load dataset
    dataset = load_cifar()

    print('\n--- Processing the dataset ---')                               # pre process dataset
    dataset = preprocess(dataset)

    train_images = np.moveaxis(dataset['train_images'], 1, 3)               # pre process data for keras
    validation_images = np.moveaxis(dataset['validation_images'], 1, 3)
    test_images = np.moveaxis(dataset['test_images'], 1, 3)
    train_labels = to_categorical(dataset['train_labels'])
    validation_labels = to_categorical(dataset['validation_labels'])
    test_labels = to_categorical(dataset['test_labels'])

    if os.path.isfile('model.h5'):                                          # load model
        print('\n--- Loading model ---')
        model = load_model('model.h5')
    else:                                                                   # build model
        print('\n--- Building model ---')
        model = Sequential()
        model.add(Conv2D(32, 3, name='conv1', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam), input_shape=(32, 32, 3)))
        model.add(Conv2D(32, 3, name='conv2', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam)))
        model.add(MaxPooling2D(2, name='pool1'))
        model.add(Conv2D(64, 3, name='conv3', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam)))
        model.add(Conv2D(64, 3, name='conv4', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam)))
        model.add(MaxPooling2D(2, name='pool2'))
        model.add(Flatten())
        model.add(Dense(256, name='fullyconnected', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(lam)))
        model.add(Dense(10, name='dense', activation='softmax'))

    print('\n--- Training the model ---')                                   # train model
    train(
        model,
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        batch_size,
        num_epochs,
        learning_rate,
        verbose
    )

    print('\n--- Testing the model ---')                                    # test model
    evaluate(model)

    print('\n--- Predicting image from test set ---')
    image_idx = 40                                                          # index of image to predict
    predict(model, image_idx)

    print('\n--- Plotting weight distributions ---')
    plot_weights(model)

    print('\n--- Saving the model ---')                                     # save model
    model.save('model.h5')
