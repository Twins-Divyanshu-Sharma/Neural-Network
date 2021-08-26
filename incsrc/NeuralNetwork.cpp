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


Layer::Layer()
{

}


Layer::Layer(int i, int o):out(o),dout(o),m(o,i),dm(o,i)
{
    std::cout << "layer ctr" << std::endl; 
    initialize(m); 
}


Layer::Layer(const Layer& layer):out(layer.out),dout(layer.dout),m(layer.m),dm(layer.dm)
{

    std::cout << "layer copy ctr" << std::endl;
}


Layer::Layer(Layer&& layer): out(std::move(layer.out)),  m(std::move(layer.m)), dm(std::move(layer.dm)),dout(std::move(layer.dout)) 
{
  new(&layer) Layer();  
}



Layer::~Layer()
{
    std::cout << "layer dtr" << std::endl;
}

Layer& Layer::operator=(Layer& l)
{
    if(this != nullptr)
    {
      this->~Layer();  
      new(this) Layer(l);
    }
    else
    {
        out = l.out;
        dout = l.dout;
        m = l.m;
        dm = l.dm;
    }

    return *this;
    
}

Layer& Layer::operator=(Layer&& l)
{
    if(this != nullptr)
    {
        this->~Layer();
        new(this) Layer(std::move(l));
    }
    else
    {
        out = std::move(l.out);
        dout = std::move(l.dout);
        m = std::move(l.m);
        dm = std::move(l.dm);
        new(&l) Layer();
    }
    return *this;
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

FNN::FNN(Vec& v) : input(v)
{}

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
   new(&f) FNN();
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
    new(&fnn) FNN();
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

   
   fnn.layers.push_back((Layer(inSize,outSize)));

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
   fnn.layers.push_back((Layer(inSize,outSize)));

    return std::move(fnn);
}

std::ostream& operator<<(std::ostream& os, Layer& l)
{
    os <<"\n ----- out ------"<<std::endl;
    for(int i=0; i<l.out.getSize(); i++)
        os<<l.out[i]<<" "<<std::flush;

    os <<"\n ----- dout ------"<<std::endl;
    for(int i=0; i<l.dout.getSize(); i++)
        os<<l.dout[i]<<" "<<std::flush;

    os <<"\n --------- m ----------"<<std::endl;
    for(int i=0; i<l.m.getRow(); i++)
    {
        for(int j=0; j<l.m.getCol(); j++)
        {
            os<<l.m[i][j]<<" "<<std::flush;
        }
        os << std::endl;
    }

    os <<"\n --------- dm ----------"<<std::endl;
    for(int i=0; i<l.dm.getRow(); i++)
    {
        for(int j=0; j<l.dm.getCol(); j++)
        {
            os<<l.dm[i][j]<<" "<<std::flush;
        }
        os << std::endl;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, FNN& f)
{
    for(int i=0; i<f.layers.size(); i++)
    {
        os<<f.layers[i]<<std::endl;
        os<<"\n\n###############################################"<<std::endl<<std::endl;;
    }

    return os;
}

Vec FNN::forwardPass(float(*act)(float))
{
    layers[0].forwardPass(act, input);
    for(int i=1; i<layers.size(); i++)
    {
        layers[i].forwardPass(act,layers[i-1].out);
    }
    return layers.back().out;
}

Vec FNN::backwardPass(float(*act)(float),Vec& v)
{
    int x = layers.size() -1;
    layers.back().out = v; 
    for(int i=x; i>0 ; i--)
    {
        layers[i].backwardPass(act, layers[i-1].out, layers[i-1].dout);
    }
    Vec dinput(input.getSize());
    layers[0].backwardPass(act,input,dinput);

    return dinput;
}

void FNN::backwardPassButNotInput(float(*act)(float),Vec& v)
{
  
    int x = layers.size() -1;
    layers.back().out = v; 
    for(int i=x; i>0 ; i--)
    {
        layers[i].backwardPass(act, layers[i-1].out, layers[i-1].dout);
    }

    layers[0].halfBackwardPass(act,input);
 
}
