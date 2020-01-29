# DeepCL
Educational Deep Learning framework based on C++ and OpenCL

This is a Deep Learning framework I wrote during my Master thesis. It allowes the creation of Various Convolutional Neural Networks Architectures as well as normal Feed Forward Neural Networks. For this purpose it creates a static computation graph, with Template Metaprogramming, which will then be executed on the GPU or CPU.
The static computation graph is used for automatic differentation, allowing differentation through arbitrary mathematical expressions implemented in this framework.

## Example
The bin directory contains a executable that trains the LeNet model, using the CPU or GPU, on the MNIST dataset. 
In the following images an execution on the MNIST dataset is displayed.
![](images/ExampleRun1.png)
![](images/ExampleRun2.png)
![](images/ExampleRun3.png)

In order to run the example, download the .exe file as well as the Kernel folder.
http://yann.lecun.com/exdb/mnist/

## Installation
In the source directory a 

## Building
There 
