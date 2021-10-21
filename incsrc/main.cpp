






#include "NeuralNetwork.h"
#include <fstream>
#include <string>
#include <sstream>

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




  // test this shit
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
