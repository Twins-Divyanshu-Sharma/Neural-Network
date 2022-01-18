# Neural-Network
A neural network library for C++
## Requirements
To use this library for your project, you need to include following files from this project to your project\
\
Insert the following headers in your include folder:\
[Algebra.h](incsrc/Algebra.h)\
[NeuralNetwork.h](incsrc/NeuralNetwork.h)\
\
Insert the following files in your source folder:\
[Algebra.cpp](incsrc/Algebra.cpp)\
[NeuralNetwork.cpp](incsrc/NeuralNetwork.cpp)\

## Example
```
#include "NeuralNetwork.h"

int main()
{
    int epoch = 2;
    int inputSize = 28*28;
    int outputSize = 10;

    FNN fnn(inputSize); // constructor
    fnn  = fnn + 50 + 20 + outputSize;

    fnn.train(epoch,"res/trainImages","res/trainLabels",1);
    fnn.test("res/trainImages","res/trainLabels",1);
}
```


## Creating a feed forward neural network
Let us assume the feed forward neural network we are creating have following layers
1. input layer of size `inputSize`
2. first hidden layer, size of which is 50
3. second hidden layer, size of which is 20
4. output layer of size `outputSize`

* First of all, an object of FNN is created, which takes inputSize as constructor
```
FNN fnn(inputSize);
```
* Then you can add layes to your feed forward neural network by simply adding the size of rest of layers in order like so:
```
fnn = fnn + 50 + 20 + outputSize;
```
just make sure the = is applied

## Using the FNN

### Training
It takes four arguements
1. no of epoch
2. the string that tells path to image data (MNIST format)
3. the string that tells path to labels of image (MNIST format)
4. an int which indicates whether to display the number of images left for each epoch in training, 1 displays while 0 does not displays
```
fnn.train(epoch,"res/trainImages","res/trainLabels",1);
```
### Testing
It takes three arguements
1. the string that tells path to image data (MNIST format)
2. the string that tells path to labels of image (MNIST format)
3. an int which indicates whether to display output for each image in test dataset or not, 1 displays while 0 does not displays
```
fnn.test("res/trainImages","res/trainLabels",1);
```
## Extra configurations

### Matrix initialization function
By default the library uses normal function for initialization of matrix, but you can use something else. This has to be done after the constructor and before you add layers
```
fnn.setMatrixRandomFunc(initNormal);
```
This function takes function pointer of type `(void)(funcName)(Mat&)`. Here you can substitute your own function here

### Setting activation function and its differentiation
These have to be set before training and testing\
Setting the activation function takes function pointer of type `(float)(act)(float)`. The function takes float as input and should apply activation function on it and return the result. The default function is sigmoid
```
fnn.setActivation(sigmoid);
```
Setting the differentiation of activation function also takes function pointer of type `(float)(diffAct)(float)`. The function takes the `output of activation function` as input and should return the differentiation of activation function using it.
```
fnn.setDiffActivation(diffSigmoid);
```
### Setting error function for output vector
These have to be set before training and testing\
Setting the error function of output vector, as you guessed, requires function pointer of type `(float)(error)(Vec&,Vec&)`.\
This function takes two vectors, the first is the target vector the second is output vector. Using them, the function should return a single float value representing the error
```
fnn.setErrorFunc(rootMeanSquare);
```
You must also set the differentiation of error function. This takes a function pointer of type `(Vec)(diffError)(Vec&,Vec&)`.\
This function takes two vectors, the first is target vector, the second is output vector. Using them, it should return a vector containing value of differentiation of funtion for each element of vector
```
fnn.setDiffError(diffRootMeanSquare);
```
### Saving and loading
You can save the values of weights of matrix
```
fnn.save("res/weights");
```
You can load weights either by constructing a new FNN or loading values in an old one
```
FNN fnn("res/weights"); // initializing weights using file res/weights
fnn.load("res/weights");
```


