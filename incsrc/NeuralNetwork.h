#ifndef HH_NEURAL_NETWORK_HH
#define HH_NEURAL_NETWORK_HH

#include "Algebra.h"

class Layer
{
private:
    Vec *out, *dout;
    Mat *m, *dm;
public:
    Layer(int i, int o);
};



#endif
