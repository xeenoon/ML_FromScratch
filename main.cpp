#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>

// Using the Eigen library for matrix operations
using namespace Eigen;
using namespace std;

// Function to load CSV file into an Eigen matrix
MatrixXd loadCSV(const std::string &path)
{
    std::ifstream file(path);
    std::vector<double> values;
    std::string line;
    int rows = 0;
    while (std::getline(file, line))
    {

        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            try
            {
                values.push_back(std::stod(token));
            }
            catch (const exception &e)
            {
                cout << "Exception: " << e.what() << endl;
                cout << token << endl;
            }
        }

        rows++;
    }

    return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), rows, values.size() / rows);
}

// ReLU activation function
MatrixXd ReLU(const MatrixXd &Z)
{
    return Z.cwiseMax(0.0);
}

// Softmax activation function
Eigen::MatrixXd softmax(const Eigen::MatrixXd &Z)
{
    Eigen::MatrixXd expZ = (Z.array() - Z.array().maxCoeff()).exp(); // Subtract max for numerical stability
    return expZ.array().rowwise() / expZ.array().colwise().sum();    // Normalize
}
// ReLU derivative function
MatrixXd ReLU_deriv(const MatrixXd &Z)
{
    return (Z.array() > 0).cast<double>();
}

// Function to initialize parameters
void init_params(MatrixXd &W1, MatrixXd &b1, MatrixXd &W2, MatrixXd &b2)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    W1 = MatrixXd::Random(10, 784) * 0.5;
    b1 = MatrixXd::Random(10, 1) * 0.5;
    W2 = MatrixXd::Random(10, 10) * 0.5;
    b2 = MatrixXd::Random(10, 1) * 0.5;
}

// One-hot encoding function
MatrixXd one_hot(const VectorXd &Y)
{
    MatrixXd one_hot_Y = MatrixXd::Zero(10, Y.size());
    for (int i = 0; i < Y.size(); ++i)
    {
        int row = static_cast<int>(Y(i));
        int col = i;
        // std::cout << "Setting matrix at (" << row << ", " << col << ")" << std::endl;
        if (row < 0 || row >= 10 || col < 0 || col >= Y.size())
        {
            std::cerr << "Index out of bounds: (" << row << ", " << col << ")" << std::endl;
        }
        one_hot_Y(row, col) = 1;
    }

    return one_hot_Y;
}

// Forward propagation function
void forward_prop(const Eigen::MatrixXd &W1, const Eigen::MatrixXd &b1,
                  const Eigen::MatrixXd &W2, const Eigen::MatrixXd &b2,
                  const Eigen::MatrixXd &X, Eigen::MatrixXd &Z1,
                  Eigen::MatrixXd &A1, Eigen::MatrixXd &Z2, Eigen::MatrixXd &A2)
{
    // cout << "foward..." << endl;
    Z1 = W1 * X; // Compute Z1
                 // Manually add b1 to each column of Z1
    for (int i = 0; i < Z1.rows(); ++i)
    {
        Z1.row(i).array() += b1(i, 0);
    }
    // cout << "Z calculated" << endl;
    A1 = ReLU(Z1); // Compute A1 (activation after ReLU)
    // cout << "RELU calc" << endl;
    Z2 = W2 * A1; // Compute Z2
                  // Manually add b1 to each column of Z2
    for (int i = 0; i < Z2.rows(); ++i)
    {
        Z2.row(i).array() += b2(i, 0);
    }
    // cout << "Z2" << endl;
    A2 = softmax(Z2); // Compute A2 (softmax activation)
    // cout << "Softmax" << endl;
}

// Backward propagation function
void backward_prop(const MatrixXd &Z1, const MatrixXd &A1, const MatrixXd &Z2, const MatrixXd &A2, const MatrixXd &W1, const MatrixXd &W2, const MatrixXd &X, const VectorXd &Y, MatrixXd &dW1, MatrixXd &db1, MatrixXd &dW2, MatrixXd &db2, int m)
{
    // cout << "begin one_hot_Y" << endl;
    MatrixXd one_hot_Y = one_hot(Y);
    // cout << "end one_hot_Y" << endl;

    // cout << "begin dZ2" << endl;
    MatrixXd dZ2 = A2 - one_hot_Y;
    // cout << "end dZ2" << endl;

    // cout << "begin dW2" << endl;
    dW2 = (1.0 / m) * dZ2 * A1.transpose();
    // cout << "end dW2" << endl;

    // cout << "begin db2" << endl;
    db2 = (1.0 / m) * dZ2.rowwise().sum();
    // cout << "end db2" << endl;

    // cout << "begin dZ1" << endl;
    MatrixXd dZ1 = (W2.transpose() * dZ2).cwiseProduct(ReLU_deriv(Z1));
    // cout << "end dZ1" << endl;

    // cout << "begin dW1" << endl;
    dW1 = (1.0 / m) * dZ1 * X.transpose();
    // cout << "end dW1" << endl;

    // cout << "begin db1" << endl;
    db1 = (1.0 / m) * dZ1.rowwise().sum();
    // cout << "end db1" << endl;
}

// Update parameters function
void update_params(MatrixXd &W1, MatrixXd &b1, MatrixXd &W2, MatrixXd &b2, const MatrixXd &dW1, const MatrixXd &db1, const MatrixXd &dW2, const MatrixXd &db2, double alpha)
{
    W1 -= alpha * dW1;
    b1 -= alpha * db1;
    W2 -= alpha * dW2;
    b2 -= alpha * db2;
}

// Get predictions function
VectorXd get_predictions(const MatrixXd &A2)
{
    // Create a VectorXd to store the result
    VectorXd predictions(A2.cols());

    // Iterate over each column to find the index of the maximum element
    for (int i = 0; i < A2.cols(); ++i)
    {
        // Find the index of the maximum element in the i-th column
        int max_index = std::distance(A2.col(i).data(), std::max_element(A2.col(i).data(), A2.col(i).data() + A2.rows()));

        // Store the index in the predictions vector
        predictions(i) = max_index;
    }

    return predictions;
}

// Get accuracy function
double get_accuracy(const VectorXd &predictions, const VectorXd &Y)
{
    return (predictions.array() == Y.array()).cast<double>().mean();
}
MatrixXd Z1, A1, Z2, A2;
MatrixXd W1, b1, W2, b2;

// Gradient descent function
void gradient_descent(MatrixXd &X, VectorXd &Y, double alpha, int iterations)
{
    int m = X.cols(); // number of examples
    for (int i = 0; i < iterations; ++i)
    {
        // cout << "Foward propogation" << endl;
        // cout << "Dimensions of W1: " << W1.rows() << " x " << W1.cols() << endl;
        // cout << "Dimensions of b1: " << b1.rows() << " x " << b1.cols() << endl;
        // cout << "Dimensions of W2: " << W2.rows() << " x " << W2.cols() << endl;
        // cout << "Dimensions of b2: " << b2.rows() << " x " << b2.cols() << endl;
        // cout << "Dimensions of X: " << X.rows() << " x " << X.cols() << endl;

        // cout << "W1 * X" << endl;

        forward_prop(W1, b1, W2, b2, X, Z1, A1, Z2, A2);

        // cout << "Finish foward propogation" << endl;

        MatrixXd dW1, db1, dW2, db2;
        // cout << "Start back propogation" << endl;

        backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, dW1, db1, dW2, db2, m);
        // cout << "End back propogation" << endl;

        // cout << "Start update params" << endl;
        update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
        // cout << "End update params" << endl;

        if (i % 10 == 0)
        {
            VectorXd predictions = get_predictions(A2);
            // std::cout << predictions.transpose() << std::endl; // Print as a row vector

            cout << "Iteration: " << i << " Accuracy: " << get_accuracy(predictions, Y) << endl;
        }
    }
}

int main()
{
    int trainingdata = 1000;
    // Load the data
    cout << "Begin loading training data" << endl;
    MatrixXd data = loadCSV("csvfiles/train.csv");
    cout << "Loaded training data" << endl;

    int m = data.rows();
    int n = data.cols();

    cout << "Splitting data" << endl;
    // Split into development and training sets
    MatrixXd data_dev = data.block(0, 0, trainingdata, n).transpose();
    VectorXd Y_train = data_dev.row(0);
    MatrixXd X_train = data_dev.bottomRows(n - 1) / 255.0;
    cout << "Splitted data" << endl;

    std::cout << "Y_train: " << Y_train.transpose() << std::endl;

    cout << "Initializing paramaters" << endl;
    init_params(W1, b1, W2, b2);
    cout << "Initialized paramaters" << endl;

    cout << "Begin propogation" << endl;
    // Perform gradient descent
    gradient_descent(X_train, Y_train, 0.1, 500);
    cout << "End propogation" << endl;

    cout << "Begin testing" << endl;
    MatrixXd data_test = data.block(trainingdata, 0, data.rows() - trainingdata, n).transpose();
    VectorXd Y_test = data_test.row(0);
    MatrixXd X_test = data_test.bottomRows(n - 1) / 255.0;
    gradient_descent(X_test, Y_test, 0.1, 30);

    return 0;
}
