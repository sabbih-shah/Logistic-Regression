#include <iostream>
#include <cmath>
#include <ostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

using namespace std;

struct parameters
{
    double theta_0=0; // only used if bias is enabled
    double theta_1=0;
    double theta_2=0;
    double threshold = 1e-3;
    double learningRate = 0.001;
    int maxIterations = 1500000;
};

void read_data(const string& fileLocation, vector<double> &height, vector<double> &weight, vector<int> &gender)
{
    ifstream dataFile(fileLocation.c_str());
    string line;
    bool isHeader = true;
    vector <string>data;

    if (dataFile.is_open())
    {
        while(!dataFile.eof())
        {
            getline(dataFile, line);
            if (isHeader)
            {
                isHeader = false;
                continue;
            }

            stringstream lineStream(line);

            while(lineStream.good())
            {
                string substr;
                getline(lineStream, substr, ',');
                data.push_back(substr);
            }
        }

        for (int i=0; i<data.size(); i=i+3)
        {
            if (data[i] == "\"Male\"" )
            {
                gender.push_back(1);
            }
            else if(data[i] == "\"Female\"")
            {
                gender.push_back(0);
            }

            height.push_back(stod(data[i+1]));
            weight.push_back(stod(data[i+2]));
        }
    }

}

struct sigmoid { double operator() (double d) const
    {
        return 1/(1+exp(-d));
    }};

double findMean(const vector<double>& vec)
{
    double total = accumulate(vec.begin(), vec.end(),  );

    for (auto i : vec) total += i;
    double mean = total / vec.size();
    return mean;
}

double findStandardDeviation(const std::vector<double>& vec)
{
    double mean = findMean(vec);
    double sd= 0;
    for (double i : vec)
    {
        double powVal = (i - mean) * (i - mean); // (val-mean)*(val-mean)
        sd += powVal;
    }
    sd = sqrt(sd / vec.size());
    return sd;
}

void StandardScaleData(vector<double> *vec)
{
    double mean = findMean(*vec);
    double sd = findStandardDeviation(*vec);

    for (double & val : *vec)
    {
        double temp = (val - mean) / sd;
        val = temp;
    }

}

std::vector<double> VecSigmoid(const std::vector<double>& x) {
    const int n = x.size();
    std::vector<double> y(n);
    std::transform(x.begin(), x.end(), y.begin(), sigmoid());
    return y;
}

std::vector<double> calcuateZ(const std::vector<double>& height, const std::vector<double>& weight, parameters &params)
{
    int dim =  height.size();
    vector<double> sigmoid;
    sigmoid.reserve(dim);
    for (int i=0; i <dim ; i++)
    {
        sigmoid.push_back(/*params.theta_0 + */params.theta_1 * height[i] + params.theta_2 * weight[i]); // w.T * X +b
    }
    return sigmoid;
}

double stepCost( vector<double> &height, vector<double> &weight, vector<int> &gender, parameters &params)
{
    vector<double> sigmoid = VecSigmoid(calcuateZ(height, weight, params)); // sigmoid = 1/(1+e(-(w.T * X + b)))

    vector <double> logSigmoid;
    transform(sigmoid.begin(), sigmoid.end(), back_inserter(logSigmoid), [](double x){return log(x);}); // log(wTx)

    vector<double> temp1;
    transform(logSigmoid.begin(), logSigmoid.end(), gender.begin(), back_inserter(temp1),
                                       [](double x, int y){return (double)x*y;});

    double ylogSigmoid = accumulate(temp1.begin(), temp1.end(), decltype(temp1)::value_type(0));

    vector <double> sublogSigmoid;
    transform(sigmoid.begin(), sigmoid.end(), back_inserter(sublogSigmoid), [](double x){return log(1 - x);}); // log(wTx)

    vector<double> temp2;
    transform(sublogSigmoid.begin(), sublogSigmoid.end(), gender.begin(), back_inserter(temp2),
              [](double x, int y){return (double)(1-y)*x;});

    double subylogSigmoid = accumulate(temp2.begin(), temp2.end(), decltype(temp2)::value_type(0));

    double cost = -ylogSigmoid -subylogSigmoid;
    return  cost;
}

void partialDerivative(vector<double> &height, vector<double> &weight, vector<int> &gender, parameters &params)
{
    vector<double> sigmoid = VecSigmoid(calcuateZ(height, weight, params));
    transform(sigmoid.begin(), sigmoid.end(), gender.begin(),  sigmoid.begin(),[](double x, double y){return x-y;});

    vector<double> temp;
    transform(height.begin(), height.end(), sigmoid.begin(), back_inserter(temp), [](double x, double y){return  x*y;});
    params.theta_1 = accumulate(temp.begin(), temp.end(), decltype(temp)::value_type(0));

    temp.clear();

    transform(weight.begin(), weight.end(), sigmoid.begin(), back_inserter(temp), [](double x, double y){return  x*y;});
    params.theta_2 = accumulate(temp.begin(), temp.end(), decltype(temp)::value_type(0));

    cout << "";
}

void logisticRegression_Fit(vector<double> &height, vector<double> &weight, vector<int> &gender, parameters &params)
{
    double step = stepCost(height, weight, gender, params);
    double oldStep = step;
    int currentIteration = 0;
    double oldTheta_1 = params.theta_1;
    double oldTheta_2 = params.theta_2;
    double change = 0;

    while(currentIteration < params.maxIterations)
    {
        oldStep = step;
        partialDerivative(height, weight, gender, params); // updates params.theta_X
        params.theta_1 = oldTheta_1 - (params.learningRate * params.theta_1); // subtract from old theta
        params.theta_2 = oldTheta_2 - (params.learningRate * params.theta_2); // same for theta_2

        oldTheta_1 = params.theta_1;
        oldTheta_2 = params.theta_2;

        step = stepCost(height, weight, gender, params);
        change = oldStep - step;

        cout << "Iteration: " << currentIteration  <<" Change: " << change << " Theta_1: " << params.theta_1 << " Theta_2: " << params.theta_2 << endl;

        if (change <= params.threshold)
            break;

        currentIteration++;
    }


}

vector<int> LogisticRegression_test(parameters &params, const vector<double> &height,
                                    const vector<double> &weight, const vector<int> &gender)
{
    vector<int> result;
    double prediction;
    for (int i=0; i < gender.size(); i++)
    {
        prediction = 1/(1+ exp(-(params.theta_1 * height[i] + params.theta_2 * weight[i])));
        if(prediction >= 0.5)
        {
            result.emplace_back(1);
        }
        else
        {
            result.emplace_back(0);
        }
    }

    return result;
}

int main()
{
    vector<int> gender;
    vector<double> weight;
    vector<double> height;

    parameters params;

    params.theta_2 = 0;
    params.theta_1 = 0;
    params.theta_0 = 0;
    params.threshold = 1e-5;
    params.learningRate = 0.001;

    read_data("training data.txt",
              height, weight, gender);

  //  fill(bias.begin(), bias.end(), 1.0);

    //height = exp_c(height);
    StandardScaleData(&weight);
    StandardScaleData(&height);

    logisticRegression_Fit(height, weight, gender, params);

    height.clear();
    gender.clear();
    weight.clear();

    read_data("testing data.txt",
              height, weight, gender);

    StandardScaleData(&weight);
    StandardScaleData(&height);

    vector<int> result = LogisticRegression_test(params, height, weight, gender);

    int true_count = 0;
    for(int i=0; i< gender.size(); i++)
    {
        if (gender[i] == result[i])
        {
            true_count++;
        }
    }

    cout << "True: " <<true_count << " False: " << gender.size()- true_count << " Accuracy: " << (((float)true_count/(float)gender.size())*100) << endl;

    return 0;
}