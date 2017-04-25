#ifndef PROBLEM_H
#define PROBLEM_H 

#include "util.h"
//#include "extra.h"
#include <cassert>

extern double prediction_time;

//parameters of a task
class Param{

	public:
		char* testFname;
		char* modelFname;
		int solver;
		int max_iter;
		Float eta, rho;
		string problem_type; // problem type
		Float infea_tol; // tolerance of infeasibility
		Float grad_tol; // stopping condition for gradient
		Float nnz_tol; // threshold to shrink to zero
		bool MultiLabel;
		bool print_to_loguai2;
		string loguai2fname;
        bool agd;

		Param(){
			print_to_loguai2 = false;
			solver = 0;
			max_iter = 1000;
			eta = 1.0;
			rho = 1.0;
			testFname = NULL;
			modelFname = NULL;
			problem_type = "NULL";
			infea_tol = 1e-4;
			grad_tol = 1e-4;
			nnz_tol = 1e-8;
			MultiLabel = false;
            agd = false;
		}
};

class ScoreVec{
	public:
		Float* c; // score vector: c[k1k2] = -v[k1k2/K][k1k2%K];
		pair<Float, int>* sorted_c = NULL; // sorted <score, index> vector; 
		pair<Float, int>** sorted_row = NULL; // sorted <score, index> vector of each row
		pair<Float, int>** sorted_col = NULL; // sorted <score, index> vector of each column
		int K1, K2;
		ScoreVec(Float* _c, int _K1, int _K2){
			//sort c as well as each row and column in increasing order
			c = _c;
			K1 = _K1;
			K2 = _K2;
			internal_sort();
		}
		ScoreVec(int _K1, int _K2, Float* _c){
			c = _c;
			K1 = _K1;
			K2 = _K2;
		}

		void internal_sort(){
			if (sorted_row != NULL && sorted_col != NULL && sorted_c != NULL){
				return;
			}
			sorted_row = new pair<Float, int>*[K1];
			for (int k1 = 0; k1 < K1; k1++){
				sorted_row[k1] = new pair<Float, int>[K2];
			}
			sorted_col = new pair<Float, int>*[K2];
			for (int k2 = 0; k2 < K2; k2++){
				sorted_col[k2] = new pair<Float, int>[K1];
			}
			sorted_c = new pair<Float, int>[K1*K2];
			for (int k1 = 0; k1 < K1; k1++){
				int offset = k1*K2;
				pair<Float, int>* sorted_row_k1 = sorted_row[k1];
				for (int k2 = 0; k2 < K2; k2++){
					Float val = c[offset+k2];
					sorted_c[offset+k2] = make_pair(val, offset+k2);
					sorted_row_k1[k2] = make_pair(val, k2);
					sorted_col[k2][k1] = make_pair(val, k1);
				}
			}
			for (int k1 = 0; k1 < K1; k1++){
				sort(sorted_row[k1], sorted_row[k1]+K2, less<pair<Float, int>>());
			}
			for (int k2 = 0; k2 < K2; k2++){
				sort(sorted_col[k2], sorted_col[k2]+K1, less<pair<Float, int>>());
			}
			sort(sorted_c, sorted_c+K1*K2, less<pair<Float, int>>());
		}

		/*void normalize(Float smallest, Float largest){
			assert(normalized == false);
			normalized = true;
			Float width = max(largest - smallest, 1e-12);
			for (int i = 0; i < K1*K2; i++){
				c[i] = (c[i]-smallest)/width;
				sorted_c[i].first = (sorted_c[i].first - smallest)/width;
				assert(c[i] >= 0 && c[i] <= 1);
				assert(sorted_c[i].first >= 0 && sorted_c[i].first <= 1);
			}
			for (int k1 = 0; k1 < K1; k1++){
				for (int k2 = 0; k2 < K2; k2++){
					sorted_row[k1][k2].first = (sorted_row[k1][k2].first - smallest)/width;
					sorted_col[k2][k1].first = (sorted_col[k2][k1].first - smallest)/width;
					assert(sorted_row[k1][k2].first >= 0 && sorted_row[k1][k2].first <= 1);
					assert(sorted_col[k2][k1].first >= 0 && sorted_col[k2][k1].first <= 1);
				}
			}
		}*/

		~ScoreVec(){
			delete[] c;
			delete[] sorted_c;
			for (int i = 0; i < K1; i++){
				delete[] sorted_row[i];
			}
			delete[] sorted_row;
			for (int i = 0; i < K2; i++){
				delete[] sorted_col[i];
			}
			delete[] sorted_col;
		}
	private: bool normalized = false;

};

class Problem{
	public:
		Problem(){
		};
		Param* param;
		int a,b; //used to store the size of the data matrix
		vector<Float*> node_score_vecs; //store matrix from row & column direction
        vector<int*> node_index_vecs;
        int* size;
		Problem(Param* _param) : param(_param) {}
		virtual void construct_data(){
			cerr << "NEED to implement construct_data() for this problem!" << endl;
			assert(false);
		}
};

inline void readLine(ifstream& fin, char* line){
	fin.getline(line, LINE_LEN);
	while (!fin.eof() && strlen(line) == 0){
		fin.getline(line, LINE_LEN);
	}
}

class BipartiteMatchingProblem : public Problem{
	public:
		BipartiteMatchingProblem(Param* _param) : Problem(_param) {}
		~BipartiteMatchingProblem(){}
		void construct_data(){
			cerr << "constructing from " << param->testFname << " ";
			ifstream fin(param->testFname);
			char* line = new char[LINE_LEN];
			readLine(fin, line);
			//K = stoi(string(line));	//stoi changes string to an int.(it must starts with a digit. it could contain letter after digits, but they will be ignored. eg: 123gg -> 123; gg123 -> fault)
			sscanf(line, "%d%d", &a, &b); // read in matrix row_number a 
			//Float* c = new Float[K*K];	//c is the matrix from data file.
			Float** c = new Float*[a+b];	//c is the matrix from data file.
            int** index = new int*[a+b];
            size = new int[a+b];
            memset(size, 0, sizeof(int)*(a+b));
			for (int i = 0; i < a; i++){
				readLine(fin, line);
				while (strlen(line) == 0){
					readLine(fin, line);
				}
				string line_str(line); //transfer char* type line into string type line_str
				vector<string> tokens = split(line_str, " ");
                size[i] = tokens.size();
                Float* c_i = new Float[size[i]];
                int* index_i = new int[size[i]];
                int count = 0;
                for (auto it = tokens.begin(); it != tokens.end(); it++, count++){
                    vector<string> idx_val = split(*it, ":");
                    int j = stoi(idx_val[0]);
                    Float c_ij = stod(idx_val[1])/(-2.0);
                    index_i[count] = j;
                    c_i[count] = c_ij;
                    size[a+j]++;
                }
                index[i] = index_i;
                c[i] = c_i;
			}
            for (int j = 0; j < b; j++){
                index[a+j] = new int[size[a+j]];
                c[a+j] = new Float[size[a+j]];
                size[a+j] = 0;
            }
            for (int i = 0; i < a; i++){
                int* index_i = index[i];
                Float* c_i = c[i];
                for (int s = 0; s < size[i]; s++){
                    int j = index_i[s];
                    Float c_ij = c_i[s];
                    index[a+j][size[a+j]] = i;
                    c[a+j][size[a+j]] = c_ij;
                    size[a+j]++;
                }
            }
            
            //cout << "rows" << endl;
            //for (int i = 0; i < a; i++){
            //    cout << i << ":";
            //    for (int s = 0; s < size[i]; s++){
            //        cout << " (" << index[i][s] << "," << c[i][s] << ")";
            //    }
            //    cout << " size=" << size[i];
            //    cout << endl;
            //}
            //
            //cout << "cols" << endl;
            //for (int j = 0; j < b; j++){
            //    cout << j << ":";
            //    for (int s = 0; s < size[a+j]; s++){
            //        cout << " (" << index[a+j][s] << "," << c[a+j][s] << ")";
            //    }
            //    cout << " size=" << size[a+j];
            //    cout << endl;
            //}

			fin.close();

			//node_score_vecs store the c matrix twice. From 0 to (a-1) it stores the matrix based on rows, a to (a+b-1) based on columns
			for (int i = 0; i < a; i++){
				node_score_vecs.push_back(c[i]);
                node_index_vecs.push_back(index[i]);
			}
			for (int j = 0; j < b; j++){
				node_score_vecs.push_back(c[a+j]);
                node_index_vecs.push_back(index[a+j]);
			}
		}
};


//map<string,Int> Problem::label_index_map;
//vector<string> Problem::label_name_list;
//Int Problem::D = -1;
//Int Problem::K = -1;
//Int* Problem::remap_indices=NULL;
//Int* Problem::rev_remap_indices=NULL;



#endif
