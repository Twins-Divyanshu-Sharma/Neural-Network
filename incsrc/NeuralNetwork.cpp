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

int Layer::getVecSize()
{
    return out.getSize();
}
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
            m[i][j] = nd(gen);
     //       m[j][i] = 0.5f;
        }
    }
}

void (*Layer::initialize)(Mat& m) = initNormal;

Layer::Layer(int i, int o):out(o),dout(o),m(o,i),dm(o,i)
{
    std::cout << "layer ctr" << std::endl; 
    initialize(m); 
}


Layer::Layer(const Layer& layer):out(layer.out),dout(layer.dout),m(layer.m),dm(layer.dm)
{

    std::cout << "layer copy ctr" << std::endl;
}


Layer::Layer(Layer&& layer)//: out(std::move(layer.out)),  m(std::move(layer.m)), dm(std::move(layer.dm)),dout(std::move(layer.dout)) 
{
    out = std::move(layer.out);
    dout = std::move(layer.dout);
    m = std::move(layer.m);
    dm = std::move(layer.dm);
      layer.out = NULL;

    layer.dout = NULL;
    Mat w(0,0);
   m = w;
    dm = w;
    std::cout << "layer move ctr " << out.getSize() << " gotSize"<< std::endl;
 //  layer.out = NULL;
  // layer.dout = NULL;
   
}



Layer::~Layer()
{
    std::cout << "layer dtr" << std::endl;
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

void Layer::halfBackwardPass(float(*diffActOut)(float),Vec& in)
{
     // correcting matrix
    for(int i=0; i<m.getRow(); i++)
    {
        for(int j=0; j<m.getCol(); j++)
        {
            dm[i][j] = dout[i] * diffActOut(out[i]) * in[j];
        }
    }

}

FNN::FNN(int r, int c) : input(r*c)
{ 
    std::cout<<"FNN cnst"<<std::endl;
}

FNN::FNN(int i) : input(i) 
{
    std::cout<<"FNN cnst"<<std::endl;
}

FNN::FNN(FNN& f) : input(f.input)
{
    std::cout << "FNN cpy ctr" << std::endl;
    for(int i=0; i<f.layers.size(); i++)
        layers.push_back(f.layers[i]);
}

FNN::FNN(FNN&& f) : input(std::move(f.input))
{
    std::cout << "FNN mv ctr " << std::endl;
    layers = std::move(f.layers); 
}

FNN& FNN::operator=(FNN& fnn)
{
    std::cout << "FNN =" << std::endl;
    if( this != &fnn )
    {
        input = fnn.input;
        for(int i=0; i<fnn.layers.size(); i++)
            layers.push_back(fnn.layers[i]); 
    }
    return *this;
}

FNN& FNN::operator=(FNN&&  fnn)
{
    std::cout << "FNN mv =" << std::endl;
    if ( this != &fnn )
    {
        input = std::move(fnn.input);
        layers = std::move(fnn.layers); 

    }
    return *this;
}
/* 
FNN& FNN::operator+(int outSize)
{
    std::cout << "FNN + " << std::endl;
    int inSize = 0;
    if(layers.size() == 0)
    {
       inSize =input.getSize(); 
    }
    else
    {
        inSize = layers.back().getVecSize();
    }
    
   layers.push_back(Layer(inSize,outSize));
    
    return *this;
}
*/

Layer::Layer()
{

}

FNN operator+(FNN& fnn, int outSize)
{
    std::cout << " l FNN + " <<outSize<< std::endl;
    int inSize = 0;
    if(fnn.layers.size() == 0)
    {
        inSize = fnn.input.getSize();
    }
    else
    {
        inSize = fnn.layers.back().getVecSize();
    }

    Layer *l = new Layer(inSize, outSize);
   fnn.layers.push_back(*l);

    return fnn;
}
 
FNN operator+(FNN&& fnn, int outSize)
{
    std::cout << "r FNN + " << outSize<< std::endl;
    int inSize = 0;
    if(fnn.layers.size() == 0)
    {
        inSize = fnn.input.getSize();
    }
    else
    {
        inSize = fnn.layers.back().getVecSize();
    }

//    Layer *l = new Layer(inSize, outSize);
   fnn.layers.push_back(std::move(Layer(inSize,outSize)));

    return std::move(fnn);
}

void Layer::print()
{
    std::cout <<"\n ----- out ------"<<std::endl;
    for(int i=0; i<out.getSize(); i++)
        std::cout<<out[i]<<" "<<std::flush;

    std::cout <<"\n ----- dout ------"<<std::endl;
    for(int i=0; i<dout.getSize(); i++)
        std::cout<<dout[i]<<" "<<std::flush;

    std::cout <<"\n --------- m ----------"<<std::endl;
    for(int i=0; i<m.getRow(); i++)
    {
        for(int j=0; j<m.getCol(); j++)
        {
            std::cout<<m[i][j]<<" "<<std::flush;
        }
        std::cout << std::endl;
    }

    std::cout <<"\n --------- dm ----------"<<std::endl;
    for(int i=0; i<dm.getRow(); i++)
    {
        for(int j=0; j<dm.getCol(); j++)
        {
            std::cout<<dm[i][j]<<" "<<std::flush;
        }
        std::cout << std::endl;
    }

}

void FNN::print()
{
    for(int i=0; i<layers.size(); i++)
    {
        layers[i].print();
        std::cout<<"\n\n###############################################"<<std::endl<<std::endl;;
    }
}
