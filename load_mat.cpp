#include <vector>
#include <stdio.h>
using namespace std;
#include <iostream>
#include <omp.h>
#include <fstream>

typedef vector<double> vec_t;
typedef vector<vec_t> mat_t;

mat_t load_mat_t(FILE *fp, bool row_major){
	if (fp == NULL)
		fprintf(stderr, "input stream is not valid.\n");
	long m, n;
	fread(&m, sizeof(long), 1, fp);
	fread(&n, sizeof(long), 1, fp);
	vec_t buf(m*n);
	fread(&buf[0], sizeof(double), m*n, fp);
	mat_t A;
	if (row_major) {
		A = mat_t(m, vec_t(n));
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				A[i][j] = buf[idx++];
	} else {
		A = mat_t(n, vec_t(m));
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				A[j][i] = buf[idx++];
	}
	return A;
}


//this function is used to clean all-zero rows or columns in a matrix
mat_t clean_mat(mat_t &mat){
	mat_t result;
	for(int i = 0; i < mat.size(); ++i){
		bool valid = false;
		for(int j = 0; j < mat[i].size(); ++j){
			if(mat[i][j] != 0) valid = true;
			break;
		}
		if(valid == true){
			result.push_back(mat[i]);
		}
	}
	return result;
}

void print_mat(vector<vec_t> &mat){
	for(int a = 0; a < mat.size(); a++){
		for(int b = 0; b < mat[a].size(); b++){
			cerr << mat[a][b] << " ";
		}
		cerr << endl;
	}
	return;
}
double dot_product(vector<double> &a, vector<double> &b){
	int length = a.size();
	double product = 0;
	for(int i = 0; i < length; ++i){
		product += a[i]*b[i];
	}
	return product;
}

void multiply(vector<vec_t> &W, vector<vec_t> &H, vector<vec_t> &R){
	int height = W.size();
	int width = H.size();

	#pragma omp parallel for
	for(int i = 0; i < height ; ++i){
		for(int j = 0; j < width; ++j){
			R[i][j] = dot_product(W[i], H[j]);
		}
	}
}


void mul_print(vector<vec_t> &W, vector<vec_t> &H){
	int height = W.size();
	int width = H.size();
	double temp = 0;
	ofstream fout;
	fout.open("result");
	fout << height << " " << width << endl;
	
	//#pragma omp parallel for
	for(int i = 0; i < height; i++){
		//int n_index = 0;
		for(int j = 0; j < width; j++){
			temp = dot_product(W[i], H[j]);
			fout << temp << " ";
		}
		fout << endl;
		if(i % 100 == 0) cout <<"processing line:" << i << endl;
	}
	fout.close();
	return;
}

int main(){

	FILE* model_fp = fopen("50ratings.model","rb");
	
	//W's index is row first, that is to say, W(matrix) is a m*k matrix, W(vector) has m vectors and each a size k vector.
	//H is the opposite, which column goes first. That is to say, H(matrix) is k*n, but H(vector) has n vectors each a size k vector 
	mat_t W = load_mat_t(model_fp, true);
	mat_t H = load_mat_t(model_fp, true);
	mat_t R;
	W = clean_mat(W);
	H = clean_mat(H);

	int height = W.size();	//calculate the height and width of the result matrix
	int width = H.size();

	cout << height << endl << width;
	/*
	R.resize(height);
	for(int i = 0; i < height; i++){
		R[i].resize(width);
	}
	*/

	//multiply(W, H, R);

	//print_mat(R);
	//cerr << "W:" << W.size()<<" "<< W[0].size();
	//cerr << "H:" << H.size()<<" "<< H[0].size();
	
	mul_print(W,H);
	
	return 0;

}
