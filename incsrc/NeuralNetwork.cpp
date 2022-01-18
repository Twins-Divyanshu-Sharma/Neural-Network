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


Vec diffRootMeanSquare(Vec& target,Vec& output)
{
    if(target.getSize() != output.getSize())
    {
        std::cerr << "Vector size does not match for rms" << std::endl;
        return target;
    }
    else
    {
        Vec res(target.getSize());
        for(int i=0; i<target.getSize(); i++)
        {
            res[i] = -2*(target[i] - output[i]);
        }
        return res;
    }
}



float rootMeanSquare(Vec& target, Vec& output)
{
    float error = 0;
    for(int i=0; i<target.getSize(); i++)
    {
        float er = target[i] - output[i];
        error += er*er;
    }
    return sqrt(error);
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


int dataset::reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;

}

int dataset::getNextDatasetNum(std::ifstream& file)
{
   int number=0;
   file.read((char*)&number,sizeof(number));
   return reverseInt(number);
}

void dataset::getConfig(std::ifstream& inp, int& magicNo, int& totalImages, int& row, int& col)
{
    magicNo = getNextDatasetNum(inp);
    totalImages = getNextDatasetNum(inp);
    row = getNextDatasetNum(inp);
    col = getNextDatasetNum(inp);
}

void dataset::getLabelConfig(std::ifstream& file, int& magicNo, int& totalLabels)
{
   magicNo = getNextDatasetNum(file);
   totalLabels = getNextDatasetNum(file);
}

Vec dataset::getNextDatasetImage(std::ifstream& file, int row, int col)
{
    Vec ret(row*col);
    unsigned char temp=0;
    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            file.read((char*)&temp,sizeof(temp));
            int pixel = (int)temp;
            ret[i*col + j] = (float)pixel/255.0f;
        }
    }
    return ret;
}

int dataset::getNextDatasetLabel(std::ifstream& file)
{
  unsigned char temp = 0;
  file.read((char*)&temp,sizeof(temp));
  int value = (int)temp;
  return temp;
}

void (*Layer::initialize)(Mat& m) = initNormal;


Layer::Layer()
{

}




int Layer::getVecSize()
{
    return out.getSize();
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

void Layer::descend(float alpha)
{
    m += -alpha * dm;
}

void Layer::save(std::ofstream& output)
{
    int row = m.getRow();
    int col = m.getCol();
    output << row << " " << col << std::endl;  
    for(int r=0; r<row; r++)
    {
        for(int c=0; c<col; c++)
        {
            output << m[r][c] << " ";
        }
    }
    output << std::endl;
}

void Layer::load(std::ifstream& input)
{
   int row=0, col=0;
   std::string line;
   std::getline(input,line);
   std::istringstream rcLine(line);
   rcLine >> row >> col;
   if(m.getRow() != row || m.getCol() != col)
   {
       m = Mat(row,col);
   }
    
   std::getline(input,line);
   std::istringstream weightsLine(line);
   for(int r=0; r<row; r++)
   {
       for(int c=0; c<col; c++)
       {
           weightsLine >> m[r][c];
       }
   }
   
}

void Layer::loadWeightsOnly(std::istringstream& iss)
{
   for(int r=0; r<m.getRow(); r++)
   {
        for(int c=0; c<m.getCol(); c++)
        {
            iss >> m[r][c];
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
    layers.back().dout = v; 
    for(int i=x; i>0 ; i--)
    {
        layers[i].backwardPass(act, layers[i-1].out, layers[i-1].dout);
    }
    Vec dinput(input.getSize());
    layers[0].backwardPass(act,input,dinput);
    
    descend(); 

    return dinput;
}

void FNN::backwardPassButNotInput(float(*act)(float),Vec& v)
{
  
    int x = layers.size() -1;
    layers.back().dout = v; 
    for(int i=x; i>0 ; i--)
    {
        layers[i].backwardPass(act, layers[i-1].out, layers[i-1].dout);
    }

    layers[0].halfBackwardPass(act,input);

    descend();
 
}

void FNN::setMatrixRandomFunc(void(*init)(Mat&))
{
    Layer::initialize = init;
}

void FNN::setLearningRate(float f)
{
    alpha = f;
}

void FNN::descend()
{
   for(int i=0; i<layers.size(); i++)
        layers[i].descend(alpha);
}

FNN::FNN()
{

}

void FNN::save(std::string path)
{
    std::ofstream output(path);
    for(int i=0; i<layers.size(); i++)
    {
        layers[i].save(output);
    }
}

void FNN::load(std::string path)
{
    std::ifstream input(path);
    for(int i=0; i<layers.size(); i++)
    {
        layers[i].load(input);
    }
}

FNN::FNN(std::string path) : input(0)
{
    std::ifstream inputFile(path);
    std::vector<int> dimensions; 
    std::string line;
    int row, col;
    bool firstTime = true;

    while(std::getline(inputFile,line))
    {
        std::istringstream rcLine(line); 
        rcLine >> row >> col;
        if(firstTime)
        {
            firstTime = false;
            input = Vec(col);
        }
        layers.push_back(Layer(col,row)); // col is input, row is output
        std::getline(inputFile,line);
        std::istringstream weightsLine(line);
        layers.back().loadWeightsOnly(weightsLine);
    }
}



void FNN::setActivation(float (*act)(float))
{
    activation = act;
}

void FNN::setDiffActivation(float (*diffAct)(float))
{
    diffActivation = diffAct;
}

void FNN::setErrorFunc(float (*error)(Vec&,Vec&))
{
   errorFunc = error; 
}

void FNN::setDiffError(Vec (*diffErrorFnc)(Vec&,Vec&))
{
    diffError = diffErrorFnc;
}

void FNN::train(float epoch, std::string imagePath, std::string labelPath, int show)
{


  std::ifstream images;
  std::ifstream labels;

  int magicNo,totalImages,row,col,totalLabels;

  int outputSize = layers.back().getVecSize();

  Vec target(outputSize);

  while(epoch--)
  {
      if(show)
        std::cout << "-------------------------" << epoch << "------------------------ " << std::endl;
      images.open(imagePath);
      labels.open(labelPath);
          
        dataset::getConfig(images,magicNo,totalImages,row,col);
        // magic no should be 2051

        dataset::getLabelConfig(labels,magicNo,totalLabels);
        // magic no should be 2049

     int imagesLeft = totalImages;
      while(imagesLeft--)
      {
          if(imagesLeft%1000 == 0 && show)
          {
             std::cout << totalImages - imagesLeft << " " << std::flush;
          }
          int targetIndex = dataset::getNextDatasetLabel(labels);
          target.reset();
          target[targetIndex] = 1;

          input = dataset::getNextDatasetImage(images,row,col);
          
          Vec output = forwardPass(activation);    
          Vec diffErrorVec = diffError(target,output);
          float error = rootMeanSquare(target,output);
          backwardPassButNotInput(diffActivation,diffErrorVec); 

          

      }
      if(show)
        std::cout << "\n-------------------" << std::endl;
      images.close();
      labels.close(); 
  }


}


void FNN::test(std::string imagePath, std::string labelPath, int show)
{

    
  std::ifstream images;
  std::ifstream labels;

  int magicNo,totalImages,row,col,totalLabels;

  int outputSize = layers.back().getVecSize();

  Vec target(outputSize);
  float avgError = 0;


  images.open(imagePath);
  labels.open(labelPath);
          
  dataset::getConfig(images,magicNo,totalImages,row,col);
    // magic no should be 2051

   dataset::getLabelConfig(labels,magicNo,totalLabels);
    // magic no should be 2049
    
  float errorPerNumber[10];
  int count[10];
  for(int i=0; i<10; i++)
  {
      errorPerNumber[i] = 0;
      count[i] = 0;
  }
  int noLines = 0;
  while(totalImages--)
  {
          int targetIndex = dataset::getNextDatasetLabel(labels);
          target.reset();
          target[targetIndex] = 1;

          input = dataset::getNextDatasetImage(images,row,col);
 
          Vec output = forwardPass(activation);     
          if(show)
             std::cout << "[" << targetIndex << "] : " << output << std::endl;
          float error = errorFunc(target,output);

          errorPerNumber[targetIndex] += error;
          count[targetIndex]++;

          avgError += error;
          noLines++;
  }
  images.close();
  labels.close();

  avgError = avgError/noLines;

  std::cout << "----------------***----------------\navgError : " << avgError << std::endl;
  std::cout << " error per number " << std::endl;
  for(int i=0; i<10; i++)
  {
      std::cout << i << " : " << errorPerNumber[i]/count[i] << std::endl;
  }

}


