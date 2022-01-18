#ifndef HH_NEURAL_NETWORK_HH
#define HH_NEURAL_NETWORK_HH

#include <math.h>

#include "Algebra.h"
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <sstream>

#define EXP 2.17128128

float sigmoid(float);
float diffSigmoidOut(float);

void initNormal(Mat& m);

Vec diffRootMeanSquare(Vec& target,Vec& output);
float rootMeanSquare(Vec& target, Vec& output);

namespace dataset
{

    int reverseInt(int i);

    int getNextDatasetNum(std::ifstream& file);

    void getConfig(std::ifstream& inp, int& magicNo, int& totalImages, int& row, int& col);

    void getLabelConfig(std::ifstream& file, int& magicNo, int& totalLabels);

    Vec getNextDatasetImage(std::ifstream& file, int row, int col);

    int getNextDatasetLabel(std::ifstream& file);
}

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

    void save(std::ofstream&);
    void load(std::ifstream&);
    void loadWeightsOnly(std::istringstream&);
};

class FNN
{
private:
	std::vector<Layer> layers;			
    FNN();
    float alpha=0.15f;

    void descend();
    float (*activation)(float) = sigmoid;
    float (*diffActivation)(float) = diffSigmoidOut;
    Vec (*diffError)(Vec&,Vec&) = diffRootMeanSquare;
    float (*errorFunc)(Vec&,Vec&) = rootMeanSquare;

public:
    Vec input;
    
    FNN(int r, int c);     // insert size of input image	
    FNN(int i);
    FNN(Vec&);
    FNN(FNN&);
    FNN(FNN&&);
    FNN(std::string path);  // load from file
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

    void save(std::string path); 
    void load(std::string path);
    
    void setActivation(float (*act)(float));
    void setDiffActivation(float (*diffAct)(float));
    void setErrorFunc(float (*error)(Vec&,Vec&));
    void setDiffError(Vec (*diffError)(Vec&,Vec&));

    void train(float epoch, std::string imagePath, std::string labelPath, int show);
    void test(std::string imagePath, std::string labelPath, int show);
    
};

#endif
