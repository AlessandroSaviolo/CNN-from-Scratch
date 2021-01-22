# Convolutional Neural Network from Scratch

This project is part of a series of projects for the course _Deep Learning_ that I attended during my exchange program at National Chiao Tung University (Taiwan). See `task.pdf` for the details of the assignment. See `report.pdf` for the report containing the representation and the analysis of the produced results.

The purpose of this project is to implement a Convolutional Neural Network from scratch for MNIST and CIFAR-10 datasets.

## 1. Dataset

- [MNIST](https://drive.google.com/open?id=1uvnD__FBdhp0m5r_dIsrr5y0XY1kn4WN)

- [CIFAR-10](https://drive.google.com/open?id=1B1YA2a-2AY4VRXFrBxJbD5h6YgJZLNez)

## 2. Project Structure

- `main.py` : main file. Set hyper parameters, load dataset, build, train and evaluate CNN model

- `model.py` : network class file. Implement the Convolutional Neural Network

- `layer.py` : layer class file. Implement each layer of the Convolutional Neural Network

- `inout.py` : import dataset, pre process dataset and plot diagnostic curves and weight distribution histograms

- `kerasCIFAR.py` : implement the same model implemented from scratch using Keras. This is usefull to train using GPU computation

## 3. License

Copyright (C) 2021 Alessandro Saviolo
```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```
