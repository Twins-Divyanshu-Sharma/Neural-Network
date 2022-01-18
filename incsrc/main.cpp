






#include "NeuralNetwork.h"
#include <fstream>
#include <string>
#include <sstream>

int main()
{

  int epoch = 1;

  int inputSize = 28*28;
  int h0Size = 50;
  int h1Size = 20;
  int outputSize = 10;

  float (*act)(float) = sigmoid;
  float (*diffActOut)(float) = diffSigmoidOut;

  FNN fnn(inputSize); // construct
  fnn.setMatrixRandomFunc(initNormal); // set normal function

  fnn = fnn + h0Size + h1Size + outputSize;
    
  fnn.train(2,"res/trainImages", "res/trainLabels", 1); 
  fnn.test("res/trainImages","res/trainLabels");

  return 0; 
}
