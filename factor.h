#ifndef FACTOR_H
#define FACTOR_H

#include "util.h"
#include "stats.h"

Stats* stats = new Stats();

class Factor{
	public:
		//virtual inline void search(){
		////}
        virtual inline void subsolve(){
		}
};

class AFactor: public Factor{
    public:
		int K; //K stands for the size of this Factor
		Float rho; //parameter for Augmented Lagrangian Method
		Float* c; // score vector for this factor

		Float nnz_tol; // tolerance of the algorithm, determines when to stop
		//pair<Float, int>* sorted_c; // sorted <value, index> pairs of c
        //unordered_map<int, int> rev_index_map;
        //int* index;

		//maintained
		//Float* grad;
		Float* x;
        Float* b;
        //Float* yacc;
        Float* sum;
        Float* msg;	//msg(j) = xi(j) - xtj(i) + uij	;  //within Factor xi
		//vector<pair<Float, int>>* act_set;  //act_set is a set of indexes. it represents all the locations in x & xt which are meaningful
        //vector<int> ever_nnz_msg; //ever-none-zero: indicate this coordinate has never become nonezero. Once it becomes none-zero, it will remain in the act_set forever.
		//bool* is_ever_nnz;
        //IndexedHeap* msg_heap;
        
		inline AFactor(int _K, Float* _c, Param* param, Float* _sum, Float* _msg){
            K = _K;
			rho = param->rho;
			nnz_tol = param->nnz_tol;
            //index = _index;
            
			//compute score vector
			c = _c;
			//sorted_c = new pair<Float, int>[K];
			//for (int k = 0; k < K; k++){
			//	sorted_c[k] = make_pair(c[k], k);
			//}
			//sort(sorted_c, sorted_c+K, greater<pair<Float, int>>());

			//relaxed prediction vector
			x = new Float[K];
            b = new Float[K];
			memset(x, 0.0, sizeof(Float)*K);
			//yacc = new Float[K];
            //for (int i = 0; i < K; i++){
            //    yacc[i] = c[i];
            //}

			//inside = new bool[K];
			//memset(inside, false, sizeof(bool)*K);
			//is_ever_nnz = new bool[K];
			//memset(is_ever_nnz, false, sizeof(bool)*K);

            msg = _msg;
            sum = _sum;
			//msg = new Float[K];
			//memset(msg, 0.0, sizeof(Float)*K);
			//act_set = new vector<pair<Float, int>>();
            //act_set->clear();
			//ever_nnz_msg.clear();
            
            //msg_heap = new IndexedHeap(K);

			//fill_act_set();
			//shrink = true;
		}
		~AFactor(){
			delete x;
            delete b;
            //delete yacc;
			//delete[] inside;
			//delete[] is_ever_act;
			//act_set->clear();
			//delete msg;
            //rev_index_map.clear();
			//delete sorted_c;
			//delete is_ever_nnz;
		}
		
        /** min_x <c/2 + msg, x> + rho/2 \|x\|_2^2
         *  min_x \| x - (- (c/2 + msg)/rho) \|_2^2
         *  b = - (c/2 + msg)/rho need to be sorted in decreasing order
         *  need c sorted in decreasing order and a list of non-zero msg index
         *  
         *  
         */
        inline void subsolve(){
			
            int act_count = 0;
            
            for (int k = 0; k < K; k++){
                b[k] = -(c[k] + msg[k])/rho;
            }
            for (int k = 0; k < K; k++){
                sum[k] -= x[k];
            }
			stats->uni_subsolve_time -= get_current_time();
            solve_simplex(K, x, b);
			stats->uni_subsolve_time += get_current_time();
            for (int k = 0; k < K; k++){
                sum[k] += x[k];
            }
            
		}

        inline Float dual_obj(){
            Float dual_obj = 0.0;
            //for (vector<pair<double, int>>::const_iterator it = msg_heap->begin(); it != msg_heap->end(); it++){
            //    dual_obj -= it->first*x[it->second];
            //}
            //for (auto it = act_set->begin(); it!= act_set->end(); it++){
            //    if (!msg_heap->hasKey(it->second)){
            //        dual_obj -= it->first*c[it->second];
            //    }
            //    dual_obj += rho/2.0*it->first*it->first;
            //}
            return dual_obj;
        }
};

//unigram factor, y follows simplex constraints
class UniFactor : public Factor{
	public:
		//fixed
		int K; //K stands for the size of this Factor
		Float rho;
		Float* c; // score vector, c[k] = -<w_k, x>
		Float nnz_tol;
		pair<Float, int>* sorted_index;

		bool shrink;
		bool tight;

		//maintained
		Float* grad;
		Float* y;
		bool* inside;
		Float* msg;	//msg(j) = xi(j) - xtj(i) + uij	;  //within Factor xi
		vector<int> act_set;  //act_set is a set of indexes. it represents all the locations in x & xt which are meaningful
		vector<int> ever_nnz_msg; //ever-none-zero: indicate this coordinate has never become nonezero. Once it becomes none-zero, it will remain in the act_set forever.
		bool* is_ever_nnz;
		int searched_index;                                                                             

		inline UniFactor(int _K, Float* _c, Param* param, bool _tight){
			K = _K;
			rho = param->rho;
			nnz_tol = param->nnz_tol;
			tight = _tight;
			//compute score vector
			c = _c;
			//cache of gradient
			grad = new Float[K];
			memset(grad, 0.0, sizeof(Float)*K);

			//relaxed prediction vector
			y = new Float[K];
			memset(y, 0.0, sizeof(Float)*K);

			inside = new bool[K];
			memset(inside, false, sizeof(bool)*K);
			is_ever_nnz = new bool[K];
			memset(is_ever_nnz, false, sizeof(bool)*K);

			msg = new Float[K];
			memset(msg, 0.0, sizeof(Float)*K);
			act_set.clear();
			ever_nnz_msg.clear();
			sorted_index = new pair<Float, int>[K];
			for (int k = 0; k < K; k++){
				sorted_index[k] = make_pair(c[k], k);
			}
			sort(sorted_index, sorted_index+K, less<pair<Float, int>>());

			//fill_act_set();
			shrink = true;
		}

		~UniFactor(){
			delete[] y;
			delete[] grad;
			delete[] inside;
			//delete[] is_ever_act;
			act_set.clear();
			delete msg;
			delete sorted_index;
			delete is_ever_nnz;
		}

		inline void add_ever_nnz(int k){
			if (!is_ever_nnz[k]){
				is_ever_nnz[k] = true;
				ever_nnz_msg.push_back(k);
			}
		}
		
		//the function below is not used in the main procedure of the project. it's used to debug.
		void fill_act_set(){
			act_set.clear();
			ever_nnz_msg.clear();
			for (int k = 0; k < K; k++){
				act_set.push_back(k);
				add_ever_nnz(k);
				inside[k] = true;
			}
		}

		//uni_search()
		inline void search(){
			stats->uni_search_time -= get_current_time();
			//compute gradient of y_i
			Float gmax = -1e100;
			int max_index = -1;
			//the loop below aims to find out max gratitude of subproblem:f(x) =  ct*x+rho/2*sigma[(msg(j))^2], st. we can reach the constrain at the fastest speed.
			for (vector<int>::iterator it = ever_nnz_msg.begin(); it != ever_nnz_msg.end(); it++){
				int k = *it;
				if (inside[k]){
					continue;
				}
				Float grad_k = c[k] + rho*msg[k];
				if (-grad_k > gmax){
					gmax = -grad_k;
					max_index = k;
				}
			}

			for (int i = 0; i < K; i++){
				pair<Float, int> p = sorted_index[i];
				int k = p.second;
				if (is_ever_nnz[k] || inside[k]){
					continue;
				}
				Float grad_k = p.first;
				if (-grad_k > gmax){
					gmax = -grad_k;
					max_index = k;
				}
				break;
			}
			//above: first loop, check all the blocks that are active,find the max; second loop: find the first block that are not in the act_set(the c matrix is already sorted).			


			//
			searched_index = max_index;
			if (max_index != -1){
				act_set.push_back(max_index);
				inside[max_index] = true;
			}
			stats->uni_search_time += get_current_time();
		}


		inline void subsolve(){
			if (act_set.size() == 0)
				return;
			stats->uni_subsolve_time -= get_current_time();
			Float* y_new = new Float[act_set.size()];
			int act_count = 0;

			Float* b = new Float[act_set.size()];
			memset(b, 0.0, sizeof(Float)*act_set.size());
			act_count = 0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k = *it;
				b[act_count] -= c[k]/rho;
				b[act_count] -= msg[k]-y[k];
				//cerr << b[act_count] << " ";
			}
			//cout << "solving simplex:" << endl;
			//cout << "\t";
			//for (int k = 0; k < act_set.size(); k++){
			//    cout << " " << b[k];
			//}
			//cout << endl;
			if (tight){
				solve_simplex(act_set.size(), y_new, b);
			} else {
				solve_simplex2(act_set.size(), y_new, b);
			}
			//cout << "\t";
			//for (int k = 0; k < act_set.size(); k++){
			//    cout << " " << y_new[k];
			//}
			//cout << endl;
			delete[] b;

			act_count = 0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k = *it;
				Float delta_y = y_new[act_count] - y[k];
				//stats->delta_Y_l1 += fabs(delta_y);
				msg[k] += delta_y;
				if (msg[k] != 0){
					add_ever_nnz(k);
				}
			}

			vector<int> next_act_set;
			next_act_set.clear();
			act_count = 0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k = *it;
				y[k] = y_new[act_count];
				//shrink
				if (!shrink || y[k] >= nnz_tol ){
					//this index is justed added by search()
					//Only if this index is active after subsolve, then it's added to ever_act_set
					/*if (k == searched_index){
					  adding_ever_act(k);
					  }*/
					next_act_set.push_back(k);
				} else {
					inside[k] = false;
				}
			}
			act_set = next_act_set;

			delete[] y_new;
			stats->uni_subsolve_time += get_current_time();
		}

		int recent_pred = -1;
		//goal: minimize score
		inline Float score(){
			/*Float score = 0.0;
			  for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
			  int k = *it;
			  score += c[k]*y[k];
			  }
			  return score;
			 */
			Float max_y = -1;
			recent_pred = -1;
			//randomly select when there is a tie
			random_shuffle(act_set.begin(), act_set.end());
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				if (y[k] > max_y){
					recent_pred = k;
					max_y = y[k];
				}
			}
			//cerr << "recent_pred=" << recent_pred << ", c=" << c[recent_pred] << endl;
			return c[recent_pred];
		}

		inline Float rel_score(){
			Float score = 0.0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				score += c[k]*y[k];
			}
			return score;
		}

		inline void display(){

			//cerr << grad[0] << " " << grad[1] << endl;
			cerr << endl;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				cerr << k << ":" << y[k] << ":" << c[k] << " ";
			}
			cerr << endl;
			/*for (int k = 0; k < K; k++)
			  cerr << y[k] << " ";
			  cerr << endl;*/
		}

};

#endif
