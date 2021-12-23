# Neural-Network
A neural network library for C++
## Requirements
To use this library for your project, you need to include following files from this project to your project\
\
Insert the following headers in your include folder:\
[Algebra.h](incsrc/Algebra.h)\
[NeuralNetwork.h](incsrc/NeuralNetwork.h)\
\
Insert the following implementation files in your source folder:\
[Algebra.cpp](incsrc/Algebra.cpp)\
[NeuralNetwork.cpp](incsrc/NeuralNetwork.cpp)

## Creating a feed forward neural network
Let us assume that the feed forward neural network we are designing have following layers
1. input layer, size of which is `inputSize` 
2. first hidden layer, size of which is `h0Size` 
3. second hidden layer, size of which is `h1Size` 
4. output layer, size of which is `outputSize` 

* First of all, you need to create an object of FNN which takes size of input layer as constructor arguement like so:
```
FNN fnn(inputSize);
```

* next, you can optionally set a matrix initializer function
``` 
fnn.setMatrixRandomFunc(initNormal);
```
This function takes function pointer of type `(void)(funcName)(Mat&)`, therefore you can substitute your own function for initialization of matrix here. The default is initNormal, which initializes matrices using normal distribution. 

* Now you can add layers to your feed-forward neural network by simply adding the size of rest of layers in order like so:
``` 
fnn = fnn + h0Size + h1Size + outputSize;
``` 
Just make sure the = step
Now you have successfully created your FNN

## Using the FNN

### Forward Pass
* The FNN class provides you with a public Vec (vector class in Algebra.h) variable called `input`
* This input variable is used to store the input for your FNN. You need to set it like so:
```
          for(int i=0; i<inputSize; i++)
          {
              float fl;
              inputStringStream >> fl;
              fnn.input[i] = 1.0f - fl/255.0f;
          }
```
here inputStringStream collects float from input file.
* After setting the input, you can obtain the output from the FNN for this particular input like so:
```
Vec output = fnn.forwardPass(act);
```
here act is activation function, a function pointer of type `float sigmoid(float)`, which you can create your own, or you can use `sigmoid` provided by this library

### Backward Pass
Backward pass requires two arguement, first is a function pointer (diffActOut) which is the differentiation of activation function ( o = sigmoid(x), this function takes must take o as input and not x ). The second arguement is the error vector (constructed using target vector and output vector).\
The library provides two function for backward pass
* If you do not need backpropagated error to input use:
```
fnn.backwardPassButNotInput(diffActOut,diffError); 
```
* If you need the error backpropagated to input for further use, you can use:
```
Vec backError = fnn.backwardPass(diffActOut,diffError);
```

### Save and load
If you want to reuse the fnn after training, you can save the values in a file like so:
```
fnn.save("res/weights");
```
where `"res/weights"` is path to file where you will save the weights\
Then, to load these, use the constructor which takes path to the saved file as arguement:
```
FNN fnn("res/weights");
```
you can also load the weights to already present FNN, although it destroys the previous fnn object matrices stored in it
```
fnn.load("res/weights");
````

## Example code
-This section provides an example for better understanding. It input file res/data which contains multiple lines. Each line is 24 x 24 float values which represents greyscale image of size 24 x 24 stored row wise

```
#include "NeuralNetwork.h"
#include <fstream>
#include <string>
#include <sstream>

// function to calculated error vector
Vec diffRootMeanSquare(Vec& target,Vec& output)
{
    if(target.getSize() != output.getSize())
    {
        std::cerr << "Vector size does not match for rms" << std::endl;
        return target;
    }
    else
    {
        Vec res(target.getSize());
        for(int i=0; i<target.getSize(); i++)
        {
            res[i] = -2*(target[i] - output[i]);
        }
        return res;
    }
}

// function to calculate root mean square error
float rootMeanSquare(Vec& target, Vec& output)
{
    float error = 0;
    for(int i=0; i<target.getSize(); i++)
    {
        float er = target[i] - output[i];
        error += er*er;
    }
    return sqrt(error);
}

int main()
{

  int epoch = 100;

  int inputSize = 24*24;
  int h0Size = 50;
  int h1Size = 20;
  int outputSize = 10;

  float (*act)(float) = sigmoid;
  float (*diffActOut)(float) = diffSigmoidOut;

  FNN fnn(inputSize); // construct
  fnn.setMatrixRandomFunc(initNormal); // set normal function

  fnn = fnn + h0Size + h1Size + outputSize;
  
  Vec target(outputSize);  
  
  std::ifstream images;
    
  std::string line;
  
  // Training
  while(epoch--)
  {
      std::cout << epoch << " " << std::flush;
      images.open("res/data");
      while(std::getline(images,line))
      {
          std::istringstream iss(line);
          int targetIndex = 0;
          iss >> targetIndex;
          target.reset();
          target[targetIndex] = 1;

          
         
          for(int i=0; i<inputSize; i++)
          {
              float fl;
              iss >> fl;
              fnn.input[i] = 1.0f - fl/255.0f;
          }


          
          Vec output = fnn.forwardPass(act);    
          Vec diffError = diffRootMeanSquare(target,output);
          float error = rootMeanSquare(target,output);
          fnn.backwardPassButNotInput(diffActOut,diffError); 

          

      }
      images.close();
    
  }




  // Testing
  int noLines = 0;
  float avgError = 0;

  images.open("res/data");
  while(std::getline(images,line))
  {
          std::istringstream iss(line);
          int targetIndex = 0;
          iss >> targetIndex;
          target.reset();
          target[targetIndex] = 1;

          
          for(int i=0; i<inputSize; i++)
          {
              float fl;
              iss >> fl;
              fnn.input[i] = 1.0f - fl/255.0f;
          }


          Vec output = fnn.forwardPass(act);     
          float error = rootMeanSquare(target,output);
          avgError += error;
          noLines++;
          std::cout << "[ "<<targetIndex << "  ] : " <<  output << std::endl;
  }

  avgError = avgError/noLines;

  std::cout << "----------------***----------------\navgError : " << avgError << std::endl;

  std::string path("res/weights");
  fnn.save(path);
 
  return 0; 
}

```



