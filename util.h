#ifndef UTIL
#define UTIL

#include<cmath>
#include<vector>
#include<map>
#include<string>
#include<cstring>
#include<stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <unordered_map>
#include <time.h>
#include <queue>
#include <algorithm>
#include <iomanip>
#include <cassert>
using namespace std;

typedef float Float;
typedef int Int;
typedef vector<pair<Int,Float> > SparseVec;
typedef unordered_map<Int,Float> HashVec;
typedef vector<Int> Labels;
typedef pair<Float, Int>* ArrayHeap;
const Int LINE_LEN = 10000000;
const Int FNAME_LEN = 1000;
const Float INFI = 1e100;
const Int INIT_SIZE = 16;
const Float UPPER_UTIL_RATE = 0.75;

//#define PermutationHash HashClass;

class ScoreComp{
	
	public:
	ScoreComp(Float* _score){
		score = _score;
	}
	bool operator()(const Int& ind1, const Int& ind2){
		return score[ind1] > score[ind2];
	}
	private:
	Float* score;
};

class ScoreCompInc{
	
	public:
	ScoreCompInc(Float* _score){
		score = _score;
	}
	bool operator()(const Int& ind1, const Int& ind2){
		return score[ind1] < score[ind2];
	}
	private:
	Float* score;
};

// Hash function [K] ->[m]

class HashFunc{
	
	public:
	Int* hashindices;
	HashFunc(){
	}
	HashFunc(Int _K){
		srand(time(NULL));
		K = _K;
		l = 10000;
		r = 100000;
		
		// pick random prime number in [l, r]
		p = rand() % (r - l) + l - 1;
		bool isprime;
		do {
			p++;
			isprime = true;
			for (Int i = 2; i * i <= p; i++){
				if (p % i == 0){
					isprime = false;
					break;
				}
			}
		} while (!isprime);
		a = rand() % p;
		b = rand() % p;
		c = rand() % p;
		hashindices = new Int[K];
		for (Int i = 0; i < K; i++){
			hashindices[i] = ((a*i*i + b*i + c) % p) % INIT_SIZE;
			if (i < INIT_SIZE) cerr << hashindices[i] % INIT_SIZE << " ";
		}
		cerr << endl;
	}
	~HashFunc(){
		delete [] hashindices;
	}
	void rehash(){
		p = rand() % (r - l) + l - 1;
                bool isprime;
                do {
                        p++;
                        isprime = true;
                        for (Int i = 2; i * i <= p; i++){
                                if (p % i == 0){
                                        isprime = false;
                                        break;
                                }
                        }
                } while (!isprime);
		a = rand() % p;
                b = rand() % p;
		for (Int i = 0; i < K; i++){
                        hashindices[i] = (a * i + b) % p;
                }
	}
	private:
	Int K, l, r;
	Int a,b,c,p;
};

class PermutationHash {
	public:
	PermutationHash(){};
	PermutationHash(Int _K){	
		srand(time(NULL));
		K = _K;
		hashindices = new Int[K];
		for (Int i = 0; i < K; i++){
			hashindices[i] = i;
		}
		random_shuffle(hashindices, hashindices+K);
	}
	~PermutationHash(){
		delete [] hashindices;
	}
	Int* hashindices;
	private:
	Int K;
};

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}
	
	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

Float inner_prod(Float* w, SparseVec* sv){

	Float sum = 0.0;
	for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++)
		sum += w[it->first]*it->second;
	return sum;
}

Float prox_l1_nneg( Float v, Float lambda ){
	
	if( v < lambda )
		return 0.0;

	return v-lambda;
}

Float prox_l1( Float v, Float lambda ){
	
	if( fabs(v) > lambda ){
		if( v>0.0 )
			return v - lambda;
		else 
			return v + lambda;
	}
	
	return 0.0;
}

int nnz(Float** v, int m, int n, Float tol){
    //v of size m by n
    //check #nnz of v
    int nnz = 0;
    for (int i = 0; i < m; i++){
        Float* vi = v[i];
        for (int j = 0; j < n; j++){
            if (fabs(vi[j]) > tol)
                nnz++;
        }
    }
    return nnz;
}

Float norm_sq( Float* v, Int size ){

	Float sum = 0.0;
	for(Int i=0;i<size;i++){
		if( v[i] != 0.0 )
			sum += v[i]*v[i];
	}
	return sum;
}

Int argmax( Float* arr, Int size ){
	
	Int kmax;
	Float max_val = -1e300;
	for(Int k=0;k<size;k++){
		if( arr[k] > max_val ){
			max_val = arr[k];
			kmax = k;
		}
	}
	return kmax;
}


/** minimize \| x - (c + y) \|_2^2
 *  where c is sorted and dense, y is sparse
 *  sorted_dense: <index, value> pairs of dense array c
 *  sorted_sparse: <index, value> pairs sorted from {pair<c_i+y_i, i>| y_i != 0 }
 *
 */
inline vector<pair<Float, int>>* solve_simplex_full(Float* c, unordered_map<int, Float>& msg_map, pair<Float, int>* sorted_dense, int n){
    vector<pair<Float, int>>* msg = new vector<pair<Float, int>>();
    for (unordered_map<int, Float>::iterator it_map = msg_map.begin(); it_map != msg_map.end(); it_map++){
        int idx = it_map->first;
        Float val = it_map->second;
        Float cy = c[idx] + val;
        msg->push_back(make_pair(cy, idx));
    }
    sort(msg->begin(), msg->end(), std::greater<pair<Float, int>>());
    vector<pair<Float, int>>::iterator it_sparse = msg->begin();
    pair<Float, int>* it_dense = sorted_dense;
    vector<pair<Float, int>> cy;
    Float sum = 0.0;
    vector<pair<Float, int>> ans;
    int i = 0;
    while ((it_sparse != msg->end()) && (it_dense != sorted_dense+n)){
        if (it_dense == sorted_dense+n){
            cy.push_back(*it_sparse);
            it_sparse++;
        } else {
            if ((it_sparse == msg->end()) || (it_sparse->first <= it_dense->first)){
                cy.push_back(*it_dense);
                it_dense++;
            } else {
                cy.push_back(*it_sparse);
                it_sparse++;
            }
        }
        sum += cy[i].first;
        double t = 0.0;
        double l = (sum-1.0)/(i+1);
        double r = cy[i].first;
        if (l <= r){
            if (r <= 0){
                t = r;
            } else {
                if (l >= 0){
                    t = l;
                } else {
                    t = 0;
                }
            }
            for (int j = 0; j <= i; j++){
                Float x = cy[j].first - t;
                if (x < 0.0){
                    x = 0.0;
                }
                assert(x <= 1.0+1e-6);
                ans.push_back(make_pair(x, cy[j].second));
            }
            break;
        }
        i++;
    }

    return &ans;
}

// min_{\|y\|_1 = 1 and y >= 0} \| y - b\|_2^2
inline void solve_simplex(int n, Float* y, Float* b){
	int* index = new int[n];
	for (int i = 0; i < n; i++)
		index[i] = i;
	memset(y, 0.0, sizeof(Float)*n);
	sort(index, index+n, ScoreComp(b));
	double sum = 0.0;
	for (int i = 0; i < n; i++){
		sum += b[i];
	}
	for (int i = n-1; i >= 0; i--){
		double t = (sum - 1.0)/(i+1);
		if (/*b[index[i]] >= 0 &&*/ b[index[i]] >= t){
			//feasible	

			for (int j = 0; j < n; j++){
				y[index[j]] = b[index[j]] - t;
				if (y[index[j]] < 0.0)
					y[index[j]] = 0;
				assert(y[index[j]] <= 1 + 1e-6);
			}
			break;
		}
		sum -= b[index[i]];
	}
	delete[] index;
}

// min_{\|y\|_1 <= 1 and y >= 0} \| y - b\|_2^2
inline void solve_simplex2(int n, Float* y, Float* b){
	int* index = new int[n];
	for (int i = 0; i < n; i++)
		index[i] = i;
	memset(y, 0.0, sizeof(Float)*n);
	sort(index, index+n, ScoreComp(b));
	double sum = 0.0;
	for (int i = 0; i < n; i++){
		sum += b[i];
	}
	for (int i = n-1; i >= 0; i--){
		double t = (sum - 1.0)/(i+1);
		// t \in [l, r]
		double l = (sum - 1.0)/(i+1);
		double r = b[index[i]];
		if (r <= 0){
			t = r;
		} else {
			if (l >= 0){
				t = l;
			} else {
				t = 0;
			}
		}
		if (l <= r){
			for (int j = 0; j < n; j++){
				y[index[j]] = b[index[j]] - t;
				if (y[index[j]] < 0.0)
					y[index[j]] = 0;
				assert(y[index[j]] <= 1 + 1e-6);
			}
			break;
		}
		sum -= b[index[i]];
	}
	delete[] index;
}




inline double get_current_time(){
	//return (double)clock()/CLOCKS_PER_SEC;
	return omp_get_wtime();
}

//shift up, maintain reverse index
inline void siftUp(Float* heap, Int index){
	Float cur = heap[index];
	while (index > 0){
		Int parent = (index-1) >> 1;
		if (cur > heap[parent]){
			heap[index] = heap[parent];
			index = parent;
		} else {
			break;
		}
	}
	heap[index] = cur;
}

//shift down, maintain reverse index
inline void siftDown(Float* heap, Int index, Int size_heap){
	Float cur = heap[index];
	Int lchild = index * 2 +1;
	Int rchild = lchild+1;
	while (lchild < size_heap){
		Int next_index = index;
		if (heap[lchild] > heap[index]){
			next_index = lchild;
		}
		if (rchild < size_heap && heap[rchild] > heap[next_index]){
			next_index = rchild;
		}
		if (index == next_index) 
			break;
		heap[index] = heap[next_index];
		heap[next_index] = cur;
		index = next_index;
		lchild = index * 2 +1; rchild = lchild+1;
	}
}

//shift up, maintain reverse index
inline void siftUp(ArrayHeap heap, Int index, Int* rev_index){
	pair<Float, Int> cur = heap[index];
	while (index > 0){
		Int parent = (index-1) >> 1;
		if (cur > heap[parent]){
			heap[index] = heap[parent];
			rev_index[heap[parent].second] = index;
			index = parent;
		} else {
			break;
		}
	}
	rev_index[cur.second] = index;
	heap[index] = cur;
}

//shift down, maintain reverse index
inline void siftDown(ArrayHeap heap, Int index, Int* rev_index, Int size_heap){
	pair<Float, Int> cur = heap[index];
	Int lchild = index * 2 +1;
	Int rchild = lchild+1;
	while (lchild < size_heap){
		Int next_index = index;
		if (heap[lchild] > heap[index]){
			next_index = lchild;
		}
		if (rchild < size_heap && heap[rchild] > heap[next_index]){
			next_index = rchild;
		}
		if (index == next_index) 
			break;
		heap[index] = heap[next_index];
		rev_index[heap[index].second] = index;
		heap[next_index] = cur;
		index = next_index;
		lchild = index * 2 +1; rchild = lchild+1;
	}
	rev_index[cur.second] = index;
}

/*
inline void push(ArrayHeap heap, pair<Float, Int> p, Int* rev_index, Int& size_heap){
	heap[size_heap++] = p;
	siftUp(heap, size_heap-1, rev_index);
}

inline pair<Float, Int> pop(ArrayHeap heap, Int* rev_index, Int& size_heap){
	pair<Float, Int> p = heap[0];
	heap[0] = heap[--size_heap];
	siftDown(heap, 0, rev_index, size_heap);
}
*/

long long line_bottom = 0, line_top = 0;
long long mat_bottom = 0, mat_top = 0;

// Given msg_L[k1] > 0, this function checks part of row k1, i.e. the set {(k1, k2) | msg_R[k2] <= 0} (Taking row case as example, can also work for columns)
// heap is either col_heap or row_heap with size K
// visit k2 in decreasing order of v ( by search top-down the heap )
// stopping condition 1: msg_R[k2] >= 0.0. (Since v is decreasing and msg_R cant be larger in this area, the gradients in this area are upper bounded by msg_R[k2]+v[k1][k2] + msg_L[k1])
// stopping condition 2: v[k1k2] + msg_L[k1] <= max_val. (Since msg_R[k2] is bounded by 0)
inline void search_line(ArrayHeap heap, Float msg_L_k1, Float* msg_R, Float& max_val, Int& max_k1k2, Int size_heap, bool* inside, Int* dir, Float eta){
	vector<Int> q;
	q.push_back(0);
	line_bottom++;
	while (!q.empty()){
		line_top++;
		Int index = q.back();
		q.pop_back();
		pair<Float, Int> p = heap[index];
		Int k1k2 = p.second;
		Int k2 = dir[k1k2];
		//cout << p.first << " " << msg_L_k1 << " " << msg_R[k2] << " " << k1k2 << ", k2=" << k2 << endl;
		if (max_val < p.first + eta*(msg_L_k1 + msg_R[k2])){
			if (!inside[k1k2]){
				//cout << "updating: " << p.first << " " << msg_L_k1 << " " << msg_R[k2] << " " << k1k2 << " " << dir[k1k2] << endl;
				max_val = p.first + eta*(msg_L_k1 + msg_R[k2]);
				max_k1k2 = k1k2;
				if (msg_R[k2] >= 0.0){
					continue;
				}
			}
		} else {
			if (msg_R[k2] >= 0.0)
				continue;
		}
		if (p.first + eta*msg_L_k1 <= max_val){
			continue;
		}
		if (index*2+1 < size_heap){
			q.push_back(index*2+1);
			if (index*2+2 < size_heap){
				q.push_back(index*2+2);
			}
		}
	}
}

// This function checks a submatrix of v, i.e. the set {(k1, k2)| msg_L[k1] <= 0 and msg_R[k2] <= 0}
// heap is v_heap with size K * K
// visit k1k2 in decreasing order of v ( by search top-down the heap )
// stopping condition 1: msg_R[k2] + msg_L[k1] >= 0.0. (Since v is decreasing and msg_L + msg_R cant be larger in this area, the gradients in this area are upper bounded by msg_R[k2]+v[k1][k2]+msg_L[k1])
// stopping condition 2: v[k1k2] <= max_val. (Since msg_L + msg_R is bounded by 0)
inline void search_matrix(ArrayHeap heap, Float* msg_L, Float* msg_R, Float& max_val, Int& max_k1k2, Int size_heap, bool* inside, Int K, Float eta){
	vector<Int> q;
	q.push_back(0);
	mat_bottom++;
	while (!q.empty()){
		mat_top++;
		Int index = q.back();
		q.pop_back();
		pair<Float, Int> p = heap[index];
		Int k1k2 = p.second;
		Int k2 = k1k2 % K;
		Int k1 = (k1k2 - k2)/K;
		//cout << p.first << " " << msg_L[k1] << " " << msg_R[k2] << " " << k1k2 << " k1,k2=" << k1 << "," << k2 << endl;
		if (max_val < p.first + eta*(msg_L[k1] + msg_R[k2]) ){
			if (!inside[k1k2]){
				//cout << "updating: " << p.first << " " << msg_L[k1] << " " << msg_R[k2] << " " << k1 << " " << k2 << endl;
				max_val = p.first + eta*(msg_L[k1] + msg_R[k2]);
				max_k1k2 = k1k2;
				if (msg_L[k1] + msg_R[k2] >= 0.0){
					continue;
				}
			}
		} else {
			if (msg_L[k1] + msg_R[k2] >= 0.0)
				continue;
		}
		if (p.first <= max_val){
			continue;
		}
		if (index*2+1 < size_heap){
			q.push_back(index*2+1);
			if (index*2+2 < size_heap)
				q.push_back(index*2+2);
		}
	}
}

#endif
