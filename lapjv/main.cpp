#include "lap.h"
#include "lap.cpp"
#include<fstream>
#include<stdio.h>
#include<iostream>
#include<omp.h>
#include<random>
#include<cassert>

using namespace std;

double** get_cost_matrix(char* name, int& K){
    ifstream fin(name);
    fin >> K;
    double** c = new double*[K];
    for (int k1 = 0; k1 < K; k1++){
        c[k1] = new double[K];
        double* c_k1 = c[k1];
        for (int k2 = 0; k2 < K; k2++){
            fin >> c_k1[k2];
        }
    }
    fin.close();
    return c;
}

double** get_random_matrix(int K){
    default_random_engine generator;
    normal_distribution<double> distribution(0.00,10);
    double** c = new double*[K];
    double** u = new double*[K];
    double** v = new double*[K];
    int dim = 50;
    for (int i = 0; i < K; i++){
        u[i] = new double[dim];
        v[i] = new double[dim];
        for (int k = 0; k < dim; k++){
            u[i][k] = distribution(generator);
            v[i][k] = distribution(generator);
        }
    }
    
    for (int i = 0; i < K; i++){
        c[i] = new double[K];
        double* c_i = c[i];
        for (int j = 0; j < K; j++){
            double sum = 0.0;
            for (int k = 0; k < dim; k++){
                sum += u[i][k] * v[j][k];
            }
            c_i[j] = sum;
        }
    }
    for (int i = 0; i < K; i++){
        delete u[i];
        delete v[i];
    }
    delete[] u;
    delete[] v;
    return c;
}

int main(int argc, char** argv){
    if (argc <= 1){
        cerr << "./lapjv [test_file]" << endl;
        return 0;
    }
    int K;
    double** c;
    c = get_cost_matrix(argv[1], K);
    cout << "K=" << K << endl;
    //c = get_random_matrix(K);
    int* rowsol = new int[K];
    int* colsol = new int[K];
    double* u = new double[K];
    double* v = new double[K];
    double overall_time = -omp_get_wtime();
    cout << "starting lapjv..." << endl;
    double ans = lap(K, c, rowsol, colsol, u, v);
    overall_time += omp_get_wtime();
    cout << "ans=" << ans << ", overall_time=" << overall_time << endl;
    double cost = 0.0;
    for (int i = 0; i < K; i++){
        cout << rowsol[i] << endl;
        cost += c[i][rowsol[i]];
    }
    assert(fabs(cost-ans) < 1e-6);
}
