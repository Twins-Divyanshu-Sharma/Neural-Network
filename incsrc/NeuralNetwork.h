#ifndef HH_NEURAL_NETWORK_HH
#define HH_NEURAL_NETWORK_HH

#include <math.h>

#include "Algebra.h"
#include <vector>
#include <random>

#define EXP 2.17128128

float sigmoid(float);
float diffSigmoidOut(float);

void initNormal(Mat& m);



class Layer
{
private:
    Vec out, dout;
    Mat m, dm;
    Layer();
public:
    static void (*initialize)(Mat& m);
    Layer(int i, int o);
    Layer(const Layer&);
    Layer(Layer&&);
    ~Layer();

    Layer& operator=(Layer&);
    Layer& operator=(Layer&&);

    void forwardPass(float(*act)(float),Vec& in);
    void backwardPass(float(*diffActOut)(float),Vec& in, Vec& din);
    void halfBackwardPass(float(*diffActOut)(float),Vec& in);
    int getVecSize();

    void print();
};

class FNN
{
private:
	std::vector<Layer> layers;		
	Vec input;
    FNN();
public:
    FNN(int r, int c);     // insert size of input image	
    FNN(int i);
    FNN(FNN&);
    FNN(FNN&&);
    FNN& operator=(FNN&);
    FNN& operator=(FNN&&);
   //FNN& operator+( int);    //insert size of next layer
    friend FNN operator+(FNN&&, int);
    friend FNN operator+(FNN&, int);
    void print();
  
};

#endif
