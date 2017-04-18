
class Stats{
	public:
		int num_bi;
		Float area1;
		int area23;
		int area4;
		Float bi_act_size;

		int num_uni; 
		Float uni_act_size;
		Float ever_nnz_msg_size;
		Float delta_Y_l1;
		Float weight_b;		

		double uni_search_time = 0.0;
		double uni_subsolve_time = 0.0;
		double bi_search_time = 0.0;
		double bi_subsolve_time = 0.0;
		double maintain_time = 0.0;
		double construct_time = 0.0;


		Stats(){
			clear();
			uni_search_time = 0.0;
			uni_subsolve_time = 0.0;
			bi_search_time = 0.0;
			bi_subsolve_time = 0.0;
			maintain_time = 0.0;
			construct_time = 0.0;
		}

		void display(){
			cerr << ", uni_act_size=" << (double)uni_act_size/num_uni;

			cerr << ", area1=" << (double)area1/num_bi << ", area23=" << (double)area23/num_bi
				<< ", area4=" << (double)area4/num_bi
				<< ", bi_act_size=" << (double)bi_act_size/num_bi 
				<< ", bi_ever_nnz_msg=" << (double)ever_nnz_msg_size/num_bi;
		}

		void clear(){
			num_bi = 0; 
			area1 = 0; area23 = 0; area4 = 0; bi_act_size = 0;

			num_uni = 0;
			uni_act_size = 0; ever_nnz_msg_size = 0; 
		}

		void display_time(){
			cerr << ", uni_search=" << uni_search_time
				<< ", uni_subsolve=" << uni_subsolve_time
				<< ", bi_search=" << bi_search_time
				<< ", bi_subsolve=" << bi_subsolve_time 
				<< ", maintain=" << maintain_time 
				<< ", construct=" << construct_time;
		}
};
