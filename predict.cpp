#include "problem.h"
#include "factor.h"
#include <time.h>

double prediction_time = 0.0;
extern Stats* stats;
bool debug = false;

void exit_with_help(){
	cerr << "Usage: ./predict (options) [testfile] [model]" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Viterbi(chain)" << endl;
	cerr << "	1 -- sparseLP" << endl;
	cerr << "       2 -- GDMM" << endl;
	cerr << "-p problem_type: " << endl;
	cerr << "   chain -- sequence labeling problem" << endl;
	cerr << "   network -- network matching problem" << endl;
	cerr << "   uai -- uai format problem" << endl;
	cerr << "-e eta: GDMM step size" << endl;
	cerr << "-o rho: coefficient/weight of message" << endl;
	cerr << "-m max_iter: max number of iterations" << endl;
	exit(0);
}

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	vector<string> args;
	for (i = 1; i < argc; i++){
		string arg(argv[i]);
		//cerr << "arg[i]:" << arg << "|" << endl;
		args.push_back(arg);
	}
	for(i=0;i<args.size();i++){
		string arg = args[i];
		if (arg == "-debug"){
			debug = true;
			continue;
		}
		if( arg[0] != '-' )
			break;

		if( ++i >= args.size() )
			exit_with_help();

		string arg2 = args[i];

		if (arg == "--printmodel"){
			param->print_to_loguai2 = true;
			param->loguai2fname = arg2;
			continue;
		}
		switch(arg[1]){
			case 's': param->solver = stoi(arg2);
				  break;
			case 'e': param->eta = stof(arg2);
				  break;
			case 'o': param->rho = stof(arg2);
				  break;
			case 'm': param->max_iter = stoi(arg2);
				  break;
			case 'p': param->problem_type = string(arg2);
				  break;
			default:
				  cerr << "unknown option: " << arg << " " << arg2 << endl;
				  exit(0);
		}
	}

	if(i>=args.size())
		exit_with_help();

	param->testFname = argv[i+1];
	i++;
	if( i<args.size() )
		param->modelFname = argv[i+1];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}

double struct_predict(Problem* prob, Param* param){ 
	Float hit = 0.0;
	Float N = 0.0;
	int n = 0;
	stats = new Stats();

	//the work of this part below is to load in the data and to initiate the permutation matrices
	//int K = prob->K;	//K stands for the size of the data matrix
	int a = prob->a;	//a stands for the height of the data matrix
	int b = prob->b;	//b stands for the width of the data matrix
	cout << "constructing factors...";
	vector<AFactor*> x;	//x is the permutation matrix sliced depend on rows.
	for (int i = 0; i < a; i++){
		AFactor* x_i = new AFactor(b, prob->node_score_vecs[i], param, true);
		x.push_back(x_i);
	}
	vector<AFactor*> xt;	//xt is the transpose matrix of x. a permutation matrix slieced depend on columns;
	for (int i = 0; i < b; i++){
		AFactor* xt_i = new AFactor(a, prob->node_score_vecs[a+i], param, false);
		xt.push_back(xt_i);
	}
	cout << "done" << endl;

	int iter = 0;
	Float rho = param->rho;    //rho = p, an coefficient.
	int* indices = new int[a+b];
	for (int i = 0; i < a+b; i++){
		indices[i] = i;
	}
	bool* taken = new bool[a]; //need to check the meaning again later 
	Float best_decoded = 1e100;	//the result of matrix C dot product P(or X)
	while (iter++ < param->max_iter){
		stats->maintain_time -= get_current_time(); 
		//random_shuffle(indices, indices+K*2);
		stats->maintain_time += get_current_time(); 
		Float act_size_sum = 0;
		Float ever_nnz_size_sum = 0;
		for (int k = 0; k < a+b; k++){
			if (k < a){
				int i = k;
				AFactor* node = x[i];		// the i th row of permutation matrix P.

				//given active set, solve subproblem
				vector<pair<Float, int>>* new_x  = node->subsolve();
				
                stats->maintain_time -= get_current_time(); 
                vector<pair<Float, int>>* act_set = node->act_set;
                unordered_map<int, Float>& msg_map = node->msg_map;
                for (vector<pair<Float, int>>::iterator it_x = act_set->begin(); it_x != act_set->end(); it_x++){
                    Float old_x = it_x->first;
                    int idx = it_x->second;
                    unordered_map<int, Float>::iterator it = msg_map.find(idx);
                    it->second -= old_x;
                    unordered_map<int, Float>::iterator it2 = xt[idx]->msg_map.find(i);
                    it2->second += old_x;
                }
                for (vector<pair<Float, int>>::iterator it_x = new_x->begin(); it_x != new_x->end(); it_x++){
                    Float val = it_x->first;
                    int idx = it_x->second;
                    unordered_map<int, Float>::iterator it = msg_map.find(idx);
                    if (it == msg_map.end()){
                        msg_map.insert(make_pair(idx, val));
                    } else {
                        it->second += val;
                    }
                    unordered_map<int, Float>::iterator it2 = xt[idx]->msg_map.find(i);
                    if (it2 == xt[idx]->msg_map.end()){
                        xt[idx]->msg_map.insert(make_pair(i, -val));
                    } else {
                        it2->second -= val;
                    }
                }
                act_set->clear();
                act_set = new_x;

				act_size_sum += node->act_set->size();
				ever_nnz_size_sum += node->msg_map.size();
				stats->maintain_time += get_current_time(); 
			} else {
				int j = k - a;
				AFactor* node = xt[j];
                
				stats->maintain_time -= get_current_time(); 
				vector<pair<Float, int>>* new_x  = node->subsolve();
				
                stats->maintain_time -= get_current_time(); 
                vector<pair<Float, int>>* act_set = node->act_set;
                unordered_map<int, Float>& msg_map = node->msg_map;
                for (vector<pair<Float, int>>::iterator it_x = act_set->begin(); it_x != act_set->end(); it_x++){
                    Float old_x = it_x->first;
                    int idx = it_x->second;
                    unordered_map<int, Float>::iterator it = msg_map.find(idx);
                    it->second -= old_x;
                    unordered_map<int, Float>::iterator it2 = x[idx]->msg_map.find(j);
                    it2->second += old_x;
                }
                for (vector<pair<Float, int>>::iterator it_x = new_x->begin(); it_x != new_x->end(); it_x++){
                    Float val = it_x->first;
                    int idx = it_x->second;
                    unordered_map<int, Float>::iterator it = msg_map.find(idx);
                    if (it == msg_map.end()){
                        msg_map.insert(make_pair(idx, val));
                    } else {
                        it->second += val;
                    }
                    unordered_map<int, Float>::iterator it2 = x[idx]->msg_map.find(j);
                    if (it2 == x[idx]->msg_map.end()){
                        x[idx]->msg_map.insert(make_pair(j, -val));
                    } else {
                        it2->second -= val;
                    }
                }
                act_set->clear();
                act_set = new_x;
				act_size_sum += node->act_set->size();
				ever_nnz_size_sum += node->msg_map.size();
				stats->maintain_time += get_current_time(); 
			}
		}
		// msg[i] = (x[i][j] - xt[j][i] + mu[i][j])
		// msg[i] = (x[i][j] - xt[j][i] + mu[i][j])
		
        //stats->maintain_time -= get_current_time(); 
		//for (int i = 0; i < a; i++){
		//	for (vector<int>::iterator it = x[i]->act_set.begin(); it != x[i]->act_set.end(); it++){
		//		int j = *it;
		//		Float delta = x[i]->y[j];
		//		x[i]->msg[j] += delta;
		//		xt[j]->msg[i] -= delta;
		//		if (abs(x[i]->msg[j]) > 1e-12){
		//			x[i]->add_ever_nnz(j);
		//			xt[j]->add_ever_nnz(i);
		//		}
		//	}
		//}
		//for (int j = 0; j < b; j++){
		//	for (vector<int>::iterator it = xt[j]->act_set.begin(); it != xt[j]->act_set.end(); it++){
		//		int i = *it;
		//		Float delta = -xt[j]->y[i];
		//		x[i]->msg[j] += delta;
		//		xt[j]->msg[i] -= delta;
		//		if (abs(x[i]->msg[j]) > 1e-12){
		//			x[i]->add_ever_nnz(j);
		//			xt[j]->add_ever_nnz(i);
		//		}
		//	}
		//}
		//Float cost = 0.0, infea = 0.0;
		//for (int i = 0; i < a; i++){
		//	for (vector<int>::iterator it = x[i]->act_set.begin(); it != x[i]->act_set.end(); it++){
		//		int j = *it;
		//		cost += x[i]->y[j] * x[i]->c[j];
		//		infea += abs(xt[j]->y[i] - x[i]->y[j]);
		//		//cout << x[i]->y[j] << "\t";
		//	}
		//}

		//for (int j = 0; j < b; j++){
		//	for (vector<int>::iterator it = xt[j]->act_set.begin(); it != xt[j]->act_set.end(); it++){
		//		int i = *it;
		//		cost += xt[j]->y[i] * xt[j]->c[i];
		//		infea += abs(xt[j]->y[i] - x[i]->y[j]);
		//		//cout << xt[j]->y[i] << "\t";
		//	}
		//	//cout << endl;
		//}
		//if (iter % 50 == 0){
		//	memset(taken, false, sizeof(bool)*a);
		//	Float decoded = 0.0;
		//	int* row_index = new int[a];
		//	for(int i = 0; i < a; i++){
		//		row_index[i] = i;
		//	}
		//	random_shuffle(row_index, row_index+a);
		//	//random_shuffle(indices, indices+a+b);
		//		
		//	for (int k = 0; k < a; k++){
		//		/*if (indices[k] >= a){
		//			continue;
		//		}*/
		//		int i = row_index[k];
		//		Float max_y = 0.0;
		//		int index = -1;
		//		for (vector<int>::iterator it = x[i]->act_set.begin(); it != x[i]->act_set.end(); it++){
		//			int j = *it;
		//			if (!taken[j] && (x[i]->y[j] > max_y)){
		//				max_y = x[i]->y[j];
		//				index = j;
		//			}
		//		}
		//		if (index == -1){
		//			for (int j = 0; j < a; j++){
		//				if (!taken[j]){
		//					index = j;
		//					break;
		//				}
		//			}
		//		}
		//		taken[index] = true;
		//		decoded += x[i]->c[index];
		//	}
		//	delete row_index;
		//	if (decoded < best_decoded){
		//		best_decoded = decoded;
		//	}
		//}
		//stats->maintain_time += get_current_time(); 

		////cout << endl;
		//cout << "iter=" << iter;
		//cout << ", recall_rate=" << recall_rate/(a+b);
		//cout << ", act_size=" << act_size_sum/(a+b);
		//cout << ", ever_nnz_size=" << ever_nnz_size_sum/(a+b);
		//cout << ", cost=" << cost/2.0 << ", infea=" << infea << ", best_decoded=" << best_decoded;
		//cout << ", search=" << stats->uni_search_time;
		//cout << ", subsolve=" << stats->uni_subsolve_time;
		//cout << ", maintain=" << stats->maintain_time;
		//cout << endl;
		//if (infea < 1e-5){
		//	break;
		//}
	}
	delete taken;
	return 0;
}


int main(int argc, char** argv){
	if (argc < 2){
		exit_with_help();
	}

	prediction_time = -get_current_time();
	srand(time(NULL));
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);

	Problem* prob = NULL;
	if (param->problem_type=="bipartite"){
		prob = new BipartiteMatchingProblem(param);
		prob->construct_data();
		//int K = ((BipartiteMatchingProblem*)prob)->K;
		int a = ((BipartiteMatchingProblem*)prob)->a;
		int b = ((BipartiteMatchingProblem*)prob)->b;
		cerr << "prob.a=" << a << endl;
		cerr << "prob.b=" << b << endl;
	}

	if (prob == NULL){
		cerr << "Need to specific problem type!" << endl;
	}

	cerr << "param.rho=" << param->rho << endl;
	cerr << "param.eta=" << param->eta << endl;

	/*
	   double t1 = get_current_time();
	   vector<Float*> cc;
	   for (int i = 0; i < 200; i++){
	   Float* temp_float = new Float[4];
	   cc.push_back(temp_float);
	   }
	   for (int tt = 0; tt < 3000*1000; tt++)
	   for (int i = 0; i < 200; i++){
	   Float* cl = cc[rand()%200];
	   Float* cr = cc[rand()%200];
	   for (int j = 0; j < 4; j++)
	   cl[j] = cr[j];
	   }
	   cerr << get_current_time() - t1 << endl;
	 */

	if (param->solver == 2){
		cerr << "Acc=" << struct_predict(prob, param) << endl;
	}
	prediction_time += get_current_time();
	cerr << "prediction time=" << prediction_time << endl;
	return 0;
}
