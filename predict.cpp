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
            case 'a': param->agd = true; i--;
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
	for (int j = 0; j < b; j++){
		AFactor* xt_j = new AFactor(a, prob->node_score_vecs[a+j], param, (a==b));
		xt.push_back(xt_j);
	}
	cout << "done" << endl;

	int iter = 0;
	Float rho = param->rho;    //rho = p, an coefficient.
	Float eta = param->eta; 
	int* indices = new int[a+b];
	for (int i = 0; i < a+b; i++){
		indices[i] = i;
	}
	bool* taken = new bool[b]; //need to check the meaning again later 
	Float best_decoded = 1e100;	//the result of matrix C dot product P(or X)
    double gamma = 0.0;
    double lambda = 0.0;
    bool agd = param->agd;
    omp_lock_t* locks = new omp_lock_t[a+b];
    for (int i = 0; i < a+b; i++){
        omp_init_lock(&locks[i]);
    }
    ifstream truth("log.5000");
    int* rowsol = new int[a];
    for (int i = 0; i < a; i++){
        truth >> rowsol[i];
    }
	while (iter++ < param->max_iter){
		stats->maintain_time -= get_current_time(); 
		random_shuffle(indices, indices+a+b);
		stats->maintain_time += get_current_time(); 
		Float act_size_sum = 0;
		Float ever_nnz_size_sum = 0;
        double old_lambda = lambda;
        lambda = (1.0+sqrt(1.0+4*lambda*lambda))/2;
        gamma = (1.0-old_lambda)/lambda;

		stats->uni_subsolve_time -= get_current_time();
        #pragma omp parallel for
		for (int kk = 0; kk < a+b; kk++){
			int k = indices[kk];
            if (k < a){
				int i = k;
				AFactor* node = x[i];		// the i th row of permutation matrix P.

				//given active set, solve subproblem
				node->subsolve();

                //stats->maintain_time -= get_current_time(); 
                #pragma omp atomic
				act_size_sum += node->act_set->size();
				ever_nnz_size_sum += node->msg_heap->size();
				//stats->maintain_time += get_current_time(); 
			} else {
				int j = k - a;
				AFactor* node = xt[j];
                
				node->subsolve();
				
                //stats->maintain_time -= get_current_time(); 
                #pragma omp atomic
				act_size_sum += node->act_set->size();
				ever_nnz_size_sum += node->msg_heap->size();
				//stats->maintain_time += get_current_time(); 
			}
		}
		stats->uni_subsolve_time += get_current_time();
		// msg[i] = (x[i][j] - xt[j][i] + mu[i][j])
		// msg[i] = (x[i][j] - xt[j][i] + mu[i][j])
        stats->maintain_time -= get_current_time();
        int recall = 0;
        #pragma omp parallel for
		for (int i = 0; i < a; i++){
			for (vector<pair<Float, int>>::iterator it = x[i]->act_set->begin(); it != x[i]->act_set->end(); it++){
				int j = it->second;
                if (j == rowsol[i]){
                    #pragma omp atomic 
                    recall++;
                }
                double delta = eta*(xt[j]->x[i]-it->first);
                if (abs(delta) < 1e-20){
                    continue;
                }
                double msgx_ij, msgxt_ji;
                if (!x[i]->msg_heap->hasKey(j)){
                    msgx_ij = x[i]->c[j] + delta;
                } else {
                    msgx_ij = x[i]->msg_heap->get_value(j) + delta;
                }
                if (agd){
                    x[i]->msg_heap->update(j, x[i]->yacc[j]*gamma + (1.0-gamma)*msgx_ij);
                    x[i]->yacc[j] = msgx_ij;
                } else {
                    x[i]->msg_heap->update(j, msgx_ij);
                }

                omp_set_lock(&locks[a+j]);
                if (!xt[j]->msg_heap->hasKey(i)){
                    msgxt_ji = xt[j]->c[i] - delta;
                } else {
                    msgxt_ji = xt[j]->msg_heap->get_value(i) - delta;
                }
                if (agd){
                    xt[j]->msg_heap->update(i, xt[j]->yacc[i]*gamma + (1.0-gamma)*msgxt_ji);
                    xt[j]->yacc[i] = msgxt_ji;
                } else {
                    xt[j]->msg_heap->update(i, msgxt_ji);
                }
                omp_unset_lock(&locks[a+j]);
			}
		}
        #pragma omp parallel for
		for (int j = 0; j < b; j++){
			for (vector<pair<Float, int>>::iterator it = xt[j]->act_set->begin(); it != xt[j]->act_set->end(); it++){
                int i = it->second;
                if (abs(x[i]->x[j]) > 1e-20){
                    continue;
                }
                double delta = eta*(it->first-x[i]->x[j]);
                if (abs(delta) < 1e-20){
                    continue;
                }
                double msgx_ij, msgxt_ji;
                if (!xt[j]->msg_heap->hasKey(i)){
                    msgxt_ji = xt[j]->c[i] - delta;
                } else {
                    msgxt_ji = xt[j]->msg_heap->get_value(i) - delta;
                }
                if (agd){
                    xt[j]->msg_heap->update(i, xt[j]->yacc[i]*gamma + (1.0-gamma)*msgxt_ji);
                    xt[j]->yacc[i] = msgxt_ji;
                } else {
                    xt[j]->msg_heap->update(i, msgxt_ji); 
                }

                omp_set_lock(&locks[i]);
                if (!x[i]->msg_heap->hasKey(j)){
                    msgx_ij = x[i]->c[j] + delta;
                } else {
                    msgx_ij = x[i]->msg_heap->get_value(j) + delta;
                } 
                if (agd){
                    x[i]->msg_heap->update(j, x[i]->yacc[j]*gamma + (1.0-gamma)*msgx_ij);
                    x[i]->yacc[j] = msgx_ij;
                } else {
                    x[i]->msg_heap->update(j, msgx_ij);
                }
                omp_unset_lock(&locks[i]);
            }
		}
        
		Float cost = 0.0, infea = 0.0, dual_obj = 0.0;
        #pragma omp parallel for
		for (int i = 0; i < a; i++){
            double dual_obj_i = x[i]->dual_obj();
            double sub_cost = 0.0;
            double sub_infea = 0.0;
			for (vector<pair<Float, int>>::iterator it = x[i]->act_set->begin(); it != x[i]->act_set->end(); it++){
                int j = it->second;
				sub_cost += it->first * x[i]->c[j];
				sub_infea += abs(xt[j]->x[i] - it->first);
			};
            #pragma omp atomic
            dual_obj += dual_obj_i;
            #pragma omp atomic
            cost += sub_cost;
            #pragma omp atomic
            infea += sub_infea;
		}
        #pragma omp parallel for
		for (int j = 0; j < b; j++){
            double dual_obj_j = xt[j]->dual_obj();
            double sub_cost = 0.0;
            double sub_infea = 0.0;
			for (vector<pair<Float, int>>::iterator it = xt[j]->act_set->begin(); it != xt[j]->act_set->end(); it++){
                int i = it->second;
				sub_cost += it->first * xt[j]->c[i];
				sub_infea += abs(it->first - x[i]->x[j]);
			}
            #pragma omp atomic
            dual_obj += dual_obj_j;
            #pragma omp atomic
            cost += sub_cost;
            #pragma omp atomic
            infea += sub_infea;
		}
		if (iter % 200 == 0){
			memset(taken, false, sizeof(bool)*b);
			Float decoded = 0.0;
			int* row_index = new int[a];
			for(int i = 0; i < a; i++){
				row_index[i] = i;
			}
			random_shuffle(row_index, row_index+a);
			//random_shuffle(indices, indices+a+b);
				
			for (int k = 0; k < a; k++){
				/*if (indices[k] >= a){
					continue;
				}*/
				int i = row_index[k];
				Float max_y = 0.0;
				int index = -1;
				for (vector<pair<Float, int>>::iterator it = x[i]->act_set->begin(); it != x[i]->act_set->end(); it++){
                    int j = it->second;
					if (!taken[j] && (it->first > max_y)){
						max_y = it->first;
						index = j;
					}
				}
				if (index == -1){
					for (int j = 0; j < b; j++){
						if (!taken[j]){
							index = j;
							break;
						}
					}
				}

				taken[index] = true;
    		    decoded += x[i]->c[index]/prob->upper*prob->max_c;
			}
            decoded = (decoded+prob->offset)*(-2);
            cout << "offset=" << prob->offset << endl;
			delete row_index;
			if (decoded < best_decoded){
				best_decoded = decoded;
			}
		}
		stats->maintain_time += get_current_time(); 

		////cout << endl;
		cout << "iter=" << iter;
		cout << ", recall_rate=" << recall*1.0/a;
		cout << ", act_size=" << act_size_sum/(a+b);
		cout << ", ever_nnz_size=" << ever_nnz_size_sum/(a+b);
		cout << ", dual_obj=" << dual_obj;
        cout << ", cost=" << cost;
        cout << ", infea=" << infea;
        cout << ", best_decoded=" << best_decoded;
        //cout << ", search=" << stats->uni_search_time;
		cout << ", subsolve=" << stats->uni_subsolve_time;
		cout << ", maintain=" << stats->maintain_time;
		cout << endl;
        //if (infea < 1e-3){
        //    for (int i = 0; i < a; i++){
        //        x[i]->rho *= 0.99;
        //    }
        //    for (int j = 0; j < b; j++){
        //        xt[j]->rho *= 0.99;
        //    }
        //}
		if (infea < 1e-5){
			break;
		}
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
