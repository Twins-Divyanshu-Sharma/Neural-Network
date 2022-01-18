#include "NeuralNetwork.h"
#include <fstream>
#include <string>
#include <sstream>

int main()
{

  int epoch = 2;

  int inputSize = 28*28;
  int outputSize = 10;

  FNN fnn(inputSize); // construct
  fnn.setMatrixRandomFunc(initNormal); // set normal function

  fnn = fnn + 50 + 20 + outputSize;
    
  fnn.train(epoch,"res/trainImages", "res/trainLabels", 1); 
  fnn.test("res/trainImages","res/trainLabels",1);

  return 0; 
}
