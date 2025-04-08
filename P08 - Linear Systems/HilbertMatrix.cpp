#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

double *HilbertMatrix(int n)
{
    double *H = new double[n * n];
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            H[i * n + j] = (double)1.0 / ((i + 1) + (j + 1) - 1.0);
        }
    }
    return H;
}

double* B(int n)
{
    double *b = new double[n];
    for (int i = 0; i < n; i++){
        b[i] = 1.0;
    }
    return b;
}

double* PerturbedB(int n){
    double *b = new double[n];
    srand(time(0));
    
    for (int i = 0; i < n; i++){
        double epsilon = static_cast<double>(rand()) / RAND_MAX; // value between in [0,1)
        if (epsilon == 0.0f){
            epsilon = 0.0001f; // ensure ε > 0
        }
        b[i] = 1.0f + epsilon;
    }
    return b;
}

void PrintMatrix(double *H, int n)
{
    cout << "H[" << n << "] = ";
    for (int i = 0; i < n; i++){
        cout << "\t| ";
        for (int j = 0; j < n; j++){
            cout << H[i * n + j];
            if (j < n - 1){
                cout << ", ";
            }
        }
        cout << "\t| " << endl;
    }
}

void PrintB(double* b, int n){
    cout << "b = [" << n << "] = [" << endl;
    for (int i = 0; i < n; i++){
        cout << "\t" << b[i];
        if (i < n - 1){
            cout << endl;
        }
    }
    cout << endl << "\t]" << endl;
}

int main(int argc, char *argv[])
{
    int size = 10;
    int *n = new int[size];

    for (int i = 0; i < size; i++){
        n[i] = pow(2, (i + 1));
    }

    double *H;
    double *b;
    double *perturbedB;

    for (int i = 0; i < 5; i++){
        H = HilbertMatrix(n[i]);
        b = B(n[i]);
        perturbedB = PerturbedB(n[i]);
        PrintMatrix(H, n[i]);
        cout << endl;
        PrintB(b, n[i]);
        cout << endl;
        PrintB(perturbedB, n[i]);
        cout << endl;
        delete[] H;
        delete[] b;
        delete[] perturbedB;
    }

    free(n);

    return 0;
}