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

Layer::Layer(int i, int o) : out(o), dout(o), m(o,i), dm(o,i) {}

int Layer::getVecSize()
{
    return out.getSize();
}

FNN::FNN(int r, int c) : input(r*c) { std::cout<<"cnst"<<std::endl;}

FNN::FNN(int i) : input(i) {std::cout<<"cnst"<<std::endl;}

FNN FNN::operator+(int outSize)
{
    int inSize = 0;
    if(layers.size() == 0)
    {
       inSize = input.getSize(); 
    }
    else
    {
        inSize = layers.back().getVecSize();
    }
    
    layers.push_back(Layer(inSize,outSize));
    
    return *this;
}




