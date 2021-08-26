#include "NeuralNetwork.h"

int main()
{
  FNN f(2);
    std::cout << "starts" << std::endl;
    f = f+3+4+2+10+11;
    std::cout << "ends" << std::endl;
    f.print();
  return 0; 
}
