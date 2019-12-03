# CNN from scratch for MNIST and CIFAR-10 datasets

Implementation of a Convolutional Neural Network from scratch, without using high-level libraries such as Keras.

### Project files

```project/root> python main.py```

Main file. Set hyper parameters, load dataset, build, train and evaluate CNN model.

```project/root> python model.py```

Network class file. Implement the Convolutional Neural Network.

```project/root> python layer.py```

Layer class file. Implement each layer of the Convolutional Neural Network.

```project/root> python inout.py```

Import dataset, pre process dataset and plot diagnostic curves and weight distribution histograms.

```project/root> kerasCIFAR.py```

Implement the same model implemented from scratch using Keras. This is usefull to train using GPU computation.
