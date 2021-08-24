/*
 * =====================================================================================
 *
 *       Filename:  NeuralNetwork.cpp
 *
 *    Description 
 *
 *        Version:  1.0
 *        Created:  08/24/2021 05:23:48 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "NeuralNetwork.h"

float sigmoid(float f)
{
    return (1.0f/(1.0f + pow(EXP,-f)));
}

float diffSigmoidOut(float f)
{
    return (f * (1.0f - f));
}

void initNormal(Mat& m)
{

    int sqRoot = sqrt(m.getRow());
    std::random_device rd;
    std::mt19937 gen(rd());
    float sd = 1.0f/sqRoot;
    for(int i=0; i<m.getRow(); i++)
    {
        for(int j=0; j<m.getCol(); j++)
        {
            std::normal_distribution<float> nd(0.0, sd);
            m[j][i] = nd(gen);
        }
    }
}

Layer::Layer(void(*initialize)(Mat& m),int i, int o):out(o),dout(o),m(o,i),dm(o,i)
{
    initialize(m);
}

Layer::~Layer()
{

}

void Layer::forwardPass(float(*act)(float),Vec& in)
{
   out = m*in; 
   for(int o=0; o<out.getSize(); o++)
       out[o] = act(out[o]);
}

void Layer::backwardPass(float(*diffActOut)(float),Vec& in,Vec& din)
{
    // correcting matrix
    for(int i=0; i<m.getRow(); i++)
    {
        for(int j=0; j<m.getCol(); j++)
        {
            dm[i][j] = dout[i] * diffActOut(out[i]) * in[j];
        }
    }

    // correcting vector
    for(int j=0; j<din.getSize(); j++)
    {
        float fl = 0;
        for(int i=0; i<out.getSize(); i++)
        {
            fl += dout[i] * diffActOut(out[i]) * m[i][j];
        }
        din[j] = fl;
    }
}
