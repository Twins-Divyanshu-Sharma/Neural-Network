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
    Mat m, dm;
    Layer();

public:
    Vec out, dout;
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

    friend std::ostream& operator<<(std::ostream&,Layer&);

    void descend(float alpha);
};

class FNN
{
private:
	std::vector<Layer> layers;			
    FNN();
    float alpha=0.15f;

    void descend();

public:
    Vec input;
    
    FNN(int r, int c);     // insert size of input image	
    FNN(int i);
    FNN(Vec&);
    FNN(FNN&);
    FNN(FNN&&);
    FNN& operator=(FNN&);
    FNN& operator=(FNN&&);


   //FNN& operator+( int);    //insert size of next layer
    friend FNN operator+(FNN&&, int);
    friend FNN operator+(FNN&, int);
    
    friend std::ostream& operator<<(std::ostream&, FNN&);

    Vec forwardPass(float(*act)(float));
    Vec backwardPass(float(*diffAct)(float),Vec&);
    void backwardPassButNotInput(float(*diffAct)(float),Vec&);
    
    void setMatrixRandomFunc(void(*init)(Mat&));

    void setLearningRate(float);

    
};

#endif
