#ifndef HH_NEURAL_NETWORK_HH
#define HH_NEURAL_NETWORK_HH

#include "Algebra.h"
#include <vector>

class Layer
{
private:
    Vec out, dout;
    Mat m, dm;
public:
    Layer(int i, int o);
    int getVecSize();
};

class FNN
{
private:
	std::vector<Layer> layers;		
	Vec input;
public:
    FNN(int r, int c);     // insert size of input image	
    FNN(int i);
    FNN operator+(int);    //insert size of next layer
    
};

#endif
