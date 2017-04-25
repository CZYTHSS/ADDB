#include <iostream>
#include <queue>
#include <string>
#include<cstring>

using namespace std;


/** Max Heap of pair<double, int>
 *  where the second value has range [T]
 *  support frequent update of values
 *  rev_index links from i to 'pointer to <value_i, i>'
 */ 
class IndexedHeap{
    public:
        IndexedHeap(int _T){
            T = _T;
            rev_index = new int[T];
            memset(rev_index, -1, sizeof(int)*T);
        }

        ~IndexedHeap(){
            queue.clear();
            delete rev_index;
        }

        inline pair<double, int> pop(){
            //swap first and last element
            pair<double, int> ans = queue[0];
            queue[0] = queue[queue.size()-1];
            //maintain reverse indices 
            rev_index[queue[0].second] = 0;
            rev_index[ans.second] = -1;
            //remove last element
            queue.pop_back();
            //sift down first element
            if (queue.size() > 0){
                siftDown(0);
            }
            return ans;
        }

        inline pair<double, int> top(){
            return queue[0];
        }

        // add (key, value) to heap
        inline void push(int key, double value){
            rev_index[key] = queue.size();
            queue.push_back(make_pair(value, key));
            siftUp(queue.size()-1);
        }

        inline void push(pair<double, int> p){
            rev_index[p.second] = queue.size();
            queue.push_back(p);
            siftUp(queue.size()-1);
        }

        inline vector<pair<double, int>>::iterator begin(){
            return queue.begin();
        }
        
        inline vector<pair<double, int>>::iterator end(){
            return queue.end();
        }

        inline int size(){
            return queue.size();
        }

        inline bool hasKey(int idx){
            return (rev_index[idx] != -1);
        }

        inline double get_value(int key){
            return queue[rev_index[key]].first;
        }

        inline void dump(){
            for (auto it = queue.begin(); it != queue.end(); it++){
                cout << " (" << it->second << "," << it->first << ")";
            }
            cout << endl;
        }

        // update value of the entry associated with key
        inline void update(int key, double value){
            int idx = rev_index[key];
            if (idx == -1){
                push(key, value);
            } else {
                queue[idx].first = value;
                siftUp(idx);
                siftDown(idx);
            }
        }
        
        //shift up, maintain reverse index
        inline void siftUp(int idx){
            pair<double, int> cur = queue[idx];
            while (idx > 0){
                int parent = (idx-1) >> 1;
                if (cur > queue[parent]){
                    queue[idx] = queue[parent];
                    rev_index[queue[parent].second] = idx;
                    idx = parent;
                } else {
                    break;
                }
            }
            rev_index[cur.second] = idx;
            queue[idx] = cur;
        }

        //shift down, maintain reverse index
        inline void siftDown(int idx){
            pair<double, int> cur = queue[idx];
            int lchild = idx * 2 +1;
            int rchild = lchild+1;
            int size_queue = queue.size();
            while (lchild < size_queue){
                int next_idx = idx;
                if (queue[lchild] > queue[idx]){
                    next_idx = lchild;
                }
                if (rchild < size_queue && queue[rchild] > queue[next_idx]){
                    next_idx = rchild;
                }
                if (idx == next_idx) 
                    break;
                queue[idx] = queue[next_idx];
                rev_index[queue[idx].second] = idx;
                queue[next_idx] = cur;
                idx = next_idx;
                lchild = idx*2+1; 
                rchild = lchild+1;
            }
            rev_index[cur.second] = idx;
        }

    private:
        vector<pair<double, int>> queue;
        int* rev_index;
        int T;
        
};
