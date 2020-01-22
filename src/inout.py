import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import idx2numpy
import numpy as np
from six.moves import cPickle
import platform
import cv2
sns.set(color_codes=True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)          # suppress messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_mnist():
    X_train = idx2numpy.convert_from_file('MNIST_data/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('MNIST_data/train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file('MNIST_data/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('MNIST_data/t10k-labels.idx1-ubyte')

    train_images = []                                                   # reshape train images so that the training set
    for i in range(X_train.shape[0]):                                   # is of shape (60000, 1, 28, 28)
        train_images.append(np.expand_dims(X_train[i], axis=0))
    train_images = np.array(train_images)

    test_images = []                                                    # reshape test images so that the test set
    for i in range(X_test.shape[0]):                                    # is of shape (10000, 1, 28, 28)
        test_images.append(np.expand_dims(X_test[i], axis=0))
    test_images = np.array(test_images)

    indices = np.random.permutation(train_images.shape[0])              # permute and split training data in
    training_idx, validation_idx = indices[:55000], indices[55000:]     # training and validation sets
    train_images, validation_images = train_images[training_idx, :], train_images[validation_idx, :]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return cPickle.load(f)
    elif version[0] == '3':
        return cPickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    X_batch = []
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        for i in range(datadict['data'].shape[0]):
            X_batch.append(np.reshape(datadict['data'][i], (3, 32, 32)))
        return np.array(X_batch), np.array(datadict['labels'])


def load_cifar():
    X_train, y_train = [], []
    for batch in range(1, 6):
        X_batch, y_batch = load_CIFAR_batch(os.path.join('CIFAR_data', 'data_batch_%d' % batch))
        X_train.append(X_batch)
        y_train.append(y_batch)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_CIFAR_batch(os.path.join('CIFAR_data', 'test_batch'))

    indices = np.random.permutation(X_train.shape[0])                       # permute and split training data in
    training_idx, validation_idx = indices[:49000], indices[49000:]         # training and validation sets
    X_train, X_val = X_train[training_idx, :], X_train[validation_idx, :]
    y_train, y_val = y_train[training_idx], y_train[validation_idx]

    return {
        'train_images': X_train,
        'train_labels': y_train,
        'validation_images': X_val,
        'validation_labels': y_val,
        'test_images': X_test,
        'test_labels': y_test
    }


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def preprocess(dataset):
    dataset['train_images'] = np.array([minmax_normalize(x) for x in dataset['train_images']])
    dataset['validation_images'] = np.array([minmax_normalize(x) for x in dataset['validation_images']])
    dataset['test_images'] = np.array([minmax_normalize(x) for x in dataset['test_images']])
    return dataset


def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, 'b', linewidth=3.0, label='Training accuracy')
    plt.plot(val_accuracy_history, 'r', linewidth=3.0, label='Validation accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy rate', fontsize=16)
    plt.legend()
    plt.title('Training Accuracy', fontsize=16)
    plt.savefig('training_accuracy.png')
    plt.show()


def plot_learning_curve(loss_history):
    plt.plot(loss_history, 'b', linewidth=3.0, label='Cross entropy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.title('Learning Curve', fontsize=16)
    plt.savefig('learning_curve.png')
    plt.show()


def plot_sample(image, true_label, predicted_label):
    plt.imshow(image)
    if true_label and predicted_label is not None:
        if type(true_label) == 'int':
            plt.title('True label: %d, Predicted Label: %d' % (true_label, predicted_label))
        else:
            plt.title('True label: %s, Predicted Label: %s' % (true_label, predicted_label))
    plt.show()


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights)
    plt.title('Histogram of ' + str(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()


def to_gray(image_name):
    image = cv2.imread(image_name + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray image', image)
    cv2.imwrite(image_name + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
