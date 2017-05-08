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

bool find(int i, vector<vector<int>*>& adj, int* colsol, bool* visit){
    for (vector<int>::iterator it = adj[i]->begin(); it != adj[i]->end(); it++){
        int j = *it;
        if (visit[j]){
            continue;
        }
        visit[j] = true;
        if (colsol[j] == -1 || find(colsol[j], adj, colsol, visit)){
            colsol[j] = i;
            return true;
        }
    }
    return false;
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
    Float* sum = new Float[b];
    Float* msg = new Float[b];
    Float* msg_acc = new Float[b];
    for (int j = 0; j < b; j++){
        sum[j] = -1.0;
    }
    memset(msg, 0, sizeof(Float)*b);
    memset(msg_acc, 0, sizeof(Float)*b);
	for (int i = 0; i < a; i++){
		AFactor* x_i = new AFactor(b, prob->node_score_vecs[i], param, sum, msg);
		x.push_back(x_i);
	}
	cout << "done" << endl;

	int iter = 0;
	Float rho = param->rho;    //rho = p, an coefficient.
	Float eta = param->eta; 
	int* indices = new int[a];
	for (int i = 0; i < a; i++){
		indices[i] = i;
	}
	bool* taken = new bool[b]; //need to check the meaning again later 
	Float best_decoded = 1e100;	//the result of matrix C dot product P(or X)
    double gamma = 0.0;
    double theta = 1.0;
    double lambda = 0.0;
    bool agd = param->agd;
    vector<int> active_rows;
    bool* visit = new bool[b];
    int* colsol = new int[b];
	while (iter++ < param->max_iter){
		stats->maintain_time -= get_current_time(); 
		random_shuffle(indices, indices+a);
		stats->maintain_time += get_current_time(); 
		Float act_size_sum = 0;
		Float ever_nnz_size_sum = 0;
        double old_lambda = lambda;
        lambda = (1.0+sqrt(1.0+4*lambda*lambda))/2;
        gamma = (1.0-old_lambda)/lambda;
        /*for (int j = 0; j < b; j++){
        `    msg[j] = (1.0-theta) * msg[j] + theta*msg[j];
        }*/
        active_rows.clear();
        stats->delta_x = 0.0;
        stats->uni_act_size = 0.0;
		for (int kk = 0; kk < a; kk++){
			int i = indices[kk];
            AFactor* node = x[i];

            //given active set, solve subproblem
            node->subsolve();

            stats->maintain_time -= get_current_time(); 
            stats->maintain_time += get_current_time(); 
		}
		// msg[i] = (x[i][j] - xt[j][i] + mu[i][j])
		// msg[i] = (x[i][j] - xt[j][i] + mu[i][j])
        stats->maintain_time -= get_current_time(); 
        Float msg_delta = 0.0;
        int num_active_col = 0;
        for (int j = 0; j < b; j++){
            Float delta = eta*sum[j];
            /*if (fabs(delta) < 1e-6){
                continue;
            }*/
            if (agd){
                //x[i]->msg_heap->update(idx_ij, x[i]->yacc[idx_ij]*gamma + (1.0-gamma)*msgx_ij);
                //xt[j]->msg_heap->update(idx_ji, xt[j]->yacc[idx_ji]*gamma + (1.0-gamma)*msgxt_ji);
                //x[i]->yacc[idx_ij] = msgx_ij;
                //xt[j]->yacc[idx_ji] = msgxt_ji;
                Float last_msg_acc = msg_acc[j];
                msg_acc[j] = msg[j] + delta;
                msg[j] = gamma*last_msg_acc + (1.0-gamma)*msg_acc[j];
                //msg[j]+= delta;
                //msg_acc[j] = (1-theta)*msg_acc[j] + theta*msg[j];
            } else {
                msg[j] += delta;
                //x[i]->msg_heap->update(idx_ij, msgx_ij);
                //xt[j]->msg_heap->update(idx_ji, msgxt_ji);
            }
        }
        
		Float cost = prob->offset, infea = 0.0, dual_obj = 0.0;
        Float infea2 = 0.0;
		for (int i = 0; i < a; i++){
            Float max_y = -1.0;
			for (int j = 0; j < b; j++){
				cost += x[i]->x[j] * (x[i]->c[j]);
                if (x[i]->x[j] > max_y){
                    max_y = x[i]->x[j];
                }
			}
            infea2 += max(0.50005 - max_y, 0.0);
		}

        //sort(rows.begin(), rows.end(), less<pair<int, int>>());
        //assert(rows[0].first >= rows[1].first);
        for (int j = 0; j < b; j++){
            infea += fabs(sum[j]);
        }
        int conflict = 0;
		if (iter % 100 == 0){
			memset(taken, false, sizeof(bool)*b);
			Float decoded = 0.0;
			int* row_index = new int[a];
			for(int i = 0; i < a; i++){
				row_index[i] = i;
			}
			random_shuffle(row_index, row_index+a);
			//random_shuffle(indices, indices+a+b);
			//ofstream fout("pred");
			for (int k = 0; k < a; k++){
				/*if (indices[k] >= a){
					continue;
				}*/
				int i = row_index[k];
				Float min_c = 1e100;
                Float true_min = 1e100;
                for (int j = 0; j < b; j++){
                    if (x[i]->c[j]+msg[j] < true_min){
                        true_min = x[i]->c[j] + msg[j];
                    }
                }
				int index = -1;
				for (int j = 0; j < b; j++){
                    assert(x[i]->x[j] >= -1e-6);
                    if (!taken[j] && (x[i]->c[j]+msg[j] < min_c)){
						min_c = x[i]->c[j]+msg[j];
						index = j;
					}
                }
                if (fabs(min_c - true_min) > 1e-8){
                    conflict++;
                }
                //if (max_y <= 0.5){
                //    conflict ++;
                //    cout << max_y << endl;
                //}
                //Float min_c = 1e100;
                //int index = -1;
				//for (int j = 0; j < b; j++){
                //    if (msg){
				//		max_y = x[i]->x[j];
				//		index = j;
				//	}
                //}
                //cout << max_y << endl;
                assert(index != -1);
				taken[index] = true;
                Float c_ij = x[i]->c[index] + msg[index];
                decoded += c_ij/1000.0*prob->max_c;
			}
            decoded += prob->offset;
            //fout.close();
			delete row_index;
			if (decoded < best_decoded){
				best_decoded = decoded;
			}
		}
        
		stats->maintain_time += get_current_time(); 

		////cout << endl;
		cout << "iter=" << iter;
		//cout << ", recall_rate=" << recall_rate/(a+b);
		cout << ", act_size=" << stats->uni_act_size/a;
		//cout << ", ever_nnz_size=" << ever_nnz_size_sum/(a+b);
		//cout << ", dual_obj=" << dual_obj;
        cout << ", conflict=" << conflict << endl;
        //cout << ", delta_x=" << stats->delta_x;
        //cout << ", num_active_col=" << num_active_col;
        cout << ", cost=" << cost;
        cout << ", infea=" << infea;
        cout << ", infea2=" << infea2;
        cout << ", best_decoded=" << best_decoded;
		//cout << ", msg_delta=" << msg_delta;
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
		if (infea < 1e-7){
			break;
		}
        //theta = (sqrt(theta*theta*theta*theta + 4 * theta * theta) - theta * theta)/2.0;
	}
    //for (int i = 0; i < a; i++){
    //    for (int j = 0; j < b; j++){
    //        cout << x[i]->c[j] + msg[j] << " ";
    //    }
    //    cout << endl;
    //}
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
