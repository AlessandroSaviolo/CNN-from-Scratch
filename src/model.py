from layer import Convolutional, Pooling, FullyConnected, Dense, regularized_cross_entropy, lr_schedule
from inout import plot_sample, plot_learning_curve, plot_accuracy_curve, plot_histogram
import numpy as np
import time


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def build_model(self, dataset_name):
        if dataset_name is 'mnist':
            self.add_layer(Convolutional(name='conv1', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=8 * 6 * 6, num_classes=10))
        else:
            self.add_layer(Convolutional(name='conv1', num_filters=32, stride=1, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=32, stride=1, size=3, activation='relu'))
            self.add_layer(Pooling(name='pool1', stride=2, size=2))
            self.add_layer(Convolutional(name='conv3', num_filters=64, stride=1, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv4', num_filters=64, stride=1, size=3, activation='relu'))
            self.add_layer(Pooling(name='pool2', stride=2, size=2))
            self.add_layer(FullyConnected(name='fullyconnected', nodes1=64 * 5 * 5, nodes2=256, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=256, num_classes=10))

    def forward(self, image, plot_feature_maps):                # forward propagate
        for layer in self.layers:
            if plot_feature_maps:
                image = (image * 255)[0, :, :]
                plot_sample(image, None, None)
            image = layer.forward(image)
        return image

    def backward(self, gradient, learning_rate):                # backward propagate
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, dataset, num_epochs, learning_rate, validate, regularization, plot_weights, verbose):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        for epoch in range(1, num_epochs + 1):
            print('\n--- Epoch {0} ---'.format(epoch))
            loss, tmp_loss, num_corr = 0, 0, 0
            initial_time = time.time()
            for i in range(len(dataset['train_images'])):
                if i % 100 == 99:
                    accuracy = (num_corr / (i + 1)) * 100       # compute training accuracy and loss up to iteration i
                    loss = tmp_loss / (i + 1)

                    history['loss'].append(loss)                # update history
                    history['accuracy'].append(accuracy)

                    if validate:
                        indices = np.random.permutation(dataset['validation_images'].shape[0])
                        val_loss, val_accuracy = self.evaluate(
                            dataset['validation_images'][indices, :],
                            dataset['validation_labels'][indices],
                            regularization,
                            plot_correct=0,
                            plot_missclassified=0,
                            plot_feature_maps=0,
                            verbose=0
                        )
                        history['val_loss'].append(val_loss)
                        history['val_accuracy'].append(val_accuracy)

                        if verbose:
                            print('[Step %05d]: Loss %02.3f | Accuracy: %02.3f | Time: %02.2f seconds | '
                                  'Validation Loss %02.3f | Validation Accuracy: %02.3f' %
                                  (i + 1, loss, accuracy, time.time() - initial_time, val_loss, val_accuracy))
                    elif verbose:
                        print('[Step %05d]: Loss %02.3f | Accuracy: %02.3f | Time: %02.2f seconds' %
                              (i + 1, loss, accuracy, time.time() - initial_time))

                    # restart time
                    initial_time = time.time()

                image = dataset['train_images'][i]
                label = dataset['train_labels'][i]

                tmp_output = self.forward(image, plot_feature_maps=0)       # forward propagation

                # compute (regularized) cross-entropy and update loss
                tmp_loss += regularized_cross_entropy(self.layers, regularization, tmp_output[label])

                if np.argmax(tmp_output) == label:                          # update accuracy
                    num_corr += 1

                gradient = np.zeros(10)                                     # compute initial gradient
                gradient[label] = -1 / tmp_output[label] + np.sum(
                    [2 * regularization * np.sum(np.absolute(layer.get_weights())) for layer in self.layers])

                learning_rate = lr_schedule(learning_rate, iteration=i)     # learning rate decay

                self.backward(gradient, learning_rate)                      # backward propagation

        if verbose:
            print('Train Loss: %02.3f' % (history['loss'][-1]))
            print('Train Accuracy: %02.3f' % (history['accuracy'][-1]))
            plot_learning_curve(history['loss'])
            plot_accuracy_curve(history['accuracy'], history['val_accuracy'])

        if plot_weights:
            for layer in self.layers:
                if 'pool' not in layer.name:
                    plot_histogram(layer.name, layer.get_weights())

    def evaluate(self, X, y, regularization, plot_correct, plot_missclassified, plot_feature_maps, verbose):
        loss, num_correct = 0, 0
        for i in range(len(X)):
            tmp_output = self.forward(X[i], plot_feature_maps)              # forward propagation

            # compute cross-entropy update loss
            loss += regularized_cross_entropy(self.layers, regularization, tmp_output[y[i]])

            prediction = np.argmax(tmp_output)                              # update accuracy
            if prediction == y[i]:
                num_correct += 1
                if plot_correct:                                            # plot correctly classified digit
                    image = (X[i] * 255)[0, :, :]
                    plot_sample(image, y[i], prediction)
                    plot_correct = 1
            else:
                if plot_missclassified:                                     # plot missclassified digit
                    image = (X[i] * 255)[0, :, :]
                    plot_sample(image, y[i], prediction)
                    plot_missclassified = 1

        test_size = len(X)
        accuracy = (num_correct / test_size) * 100
        loss = loss / test_size
        if verbose:
            print('Test Loss: %02.3f' % loss)
            print('Test Accuracy: %02.3f' % accuracy)
        return loss, accuracy
