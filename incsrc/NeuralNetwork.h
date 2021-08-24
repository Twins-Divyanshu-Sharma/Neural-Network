#ifndef HH_NEURAL_NETWORK_HH
#define HH_NEURAL_NETWORK_HH

#include <math.h>
#include <random>
#include "Algebra.h"

#define EXP 2.17128128

float sigmoid(float);
float diffSigmoidOut(float);

void initNormal(Mat& m);


class Layer
{
private:
    Vec out, dout;
    Mat m, dm;
public:
    Layer(void(*initialize)(Mat& m),int i, int o);
    ~Layer();
    void forwardPass(float(*act)(float),Vec& in);
    void backwardPass(float(*diffActOut)(float),Vec& in, Vec& din);
    void halfBackwardPass(float(*diffActOut)(float),Vec& in);
};



#endif
