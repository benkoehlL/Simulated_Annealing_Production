/* ############################################################################
###############################################################################
This program uses simulated annealing to determine low-tardiness states of a 
production system
###############################################################################
#############################################################################*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <gsl/gsl_rng.h>
#include "parameter.h"

using namespace std;
		
float delay_scaling(float delay, float n){
	if(delay<0.0){
		return(delay/n);
	}
	else{
		return(n*delay);
	}
}			

float cut_true_tardiness(float z){
	if(z<0){
		return 0;
	}
	else{
		return z;
	}
}

float cut(float z){
	if(z<0){
		return z/10;
	}
	else{
		return 10*z;
	}
}

class job{
	public:
		int id;
		int type;
		float due_date;
		float t_smd;
		void set_job_state(int i, int t, float dd, float tsmd){
			id = i;
			type = t;
			due_date = dd;
			t_smd = tsmd;
		}
};

class machine_state{
	public:
		vector<job> jobs;
		void set_machine_state(int ji, job j){
			jobs[ji].set_job_state(j.id, j.type, j.due_date, j.t_smd);
		}
        void print_jobs() {
            for (int i = 0; i < jobs.size(); i++) {
                job j = jobs[i];
                cout << "Job " << j.id << " (type " << j.type << "):" << endl;
                cout << "Due date: " << j.due_date << endl;
                cout << "Time to complete: " << j.t_smd << endl;
            }
        }
};

class production_state{
	public:
		machine_state  machine_states[num_machines];
		void set_production_state(int mi, int ji, job j){
			machine_states[mi].set_machine_state(ji, j);
		}
		void switch_production_states(int mi1, int ji1,
									  int mi2, int ji2){
			job help_job = {0,0,0.0,0.0};
			if(ji1 < machine_states[mi1].jobs.size() && ji2 < machine_states[mi2].jobs.size()){
				help_job.set_job_state(machine_states[mi1].jobs[ji1].id,
									   machine_states[mi1].jobs[ji1].type, 
									   machine_states[mi1].jobs[ji1].due_date,
									   machine_states[mi1].jobs[ji1].t_smd);
				machine_states[mi1].jobs[ji1].set_job_state(machine_states[mi2].jobs[ji2].id, 
                                                                            machine_states[mi2].jobs[ji2].type,
                                                                            machine_states[mi2].jobs[ji2].due_date,
                                                                            machine_states[mi2].jobs[ji2].t_smd);
				machine_states[mi2].jobs[ji2].set_job_state(help_job.id, 
                                                                            help_job.type,
                                                                            help_job.due_date,
                                                                            help_job.t_smd);
			}
			else if(ji1 == machine_states[mi1].jobs.size() && ji2 < machine_states[mi2].jobs.size()){
				machine_states[mi1].jobs.push_back(machine_states[mi2].jobs[ji2]);
				machine_states[mi2].jobs.erase(
					machine_states[mi2].jobs.begin() + ji2);
			}
			
			else if(ji2 == machine_states[mi2].jobs.size() && ji1 < machine_states[mi1].jobs.size() ){
				machine_states[mi2].jobs.push_back(machine_states[mi1].jobs[ji1]);
				machine_states[mi1].jobs.erase(
					machine_states[mi1].jobs.begin() + ji1);
			}
			
		}
		
		// Ising approach to dH
		
		float calculate_dH(int mi1, int ji1,
						   int mi2, int ji2, float n){
			float delay1 = 0.0;
			float delay2 = 0.0;
			int i = 0;
			float sum = 0.0;
			
			while(i<ji1){
				delay1 += machine_states[mi1].jobs[i].t_smd;
				i++;
			}
			i = 0;
			while(i<ji2){
				delay2 += machine_states[mi2].jobs[i].t_smd;
				i++;
			}
			sum += delay_scaling(delay1 + machine_states[mi2].jobs[ji2].t_smd
					- machine_states[mi2].jobs[ji2].due_date,n)
				+ delay_scaling(delay2 + machine_states[mi1].jobs[ji1].t_smd
					- machine_states[mi1].jobs[ji1].due_date,n)
				+ (machine_states[mi1].jobs[ji1].t_smd
					- machine_states[mi2].jobs[ji2].t_smd)
					*(machine_states[mi1].jobs.size()- ji1)
				+ (machine_states[mi2].jobs[ji2].t_smd
					- machine_states[mi1].jobs[ji1].t_smd)
					*(machine_states[mi2].jobs.size()- ji2);
			if(machine_states[mi1].jobs[ji1].type != machine_states[mi2].jobs[ji2].type){
				if(ji2-1>=0 && machine_states[mi1].jobs[ji1].type == machine_states[mi2].jobs[ji2-1].type 
					&& machine_states[mi2].jobs[ji2].type != machine_states[mi2].jobs[ji2-1].type){
					sum -= t_large_setup;
				}
				if(ji2-1>=0 && machine_states[mi1].jobs[ji1].type != machine_states[mi2].jobs[ji2-1].type 
					&& machine_states[mi2].jobs[ji2].type == machine_states[mi2].jobs[ji2-1].type){
					sum += t_large_setup;
				}
				if(ji1-1>=0 && machine_states[mi2].jobs[ji2].type == machine_states[mi1].jobs[ji1-1].type 
					&& machine_states[mi1].jobs[ji1].type != machine_states[mi1].jobs[ji1-1].type){
					sum -= t_large_setup;
				}
				if(ji1-1>=0 && machine_states[mi2].jobs[ji2].type != machine_states[mi1].jobs[ji1-1].type 
					&& machine_states[mi1].jobs[ji1].type == machine_states[mi1].jobs[ji1-1].type){
					sum += t_large_setup;
				}
				
				if(ji2+1<machine_states[mi2].jobs.size() && machine_states[mi1].jobs[ji1].type == machine_states[mi2].jobs[ji2+1].type 
					&& machine_states[mi2].jobs[ji2].type != machine_states[mi2].jobs[ji2+1].type){
					sum -= t_large_setup;
				}
				if(ji2+1<machine_states[mi2].jobs.size() && machine_states[mi1].jobs[ji1].type != machine_states[mi2].jobs[ji2+1].type 
					&& machine_states[mi2].jobs[ji2].type == machine_states[mi2].jobs[ji2+1].type){
					sum += t_large_setup;
				}
				if(ji1+1<machine_states[mi1].jobs.size() && machine_states[mi2].jobs[ji2].type == machine_states[mi1].jobs[ji1+1].type 
					&& machine_states[mi1].jobs[ji1].type != machine_states[mi1].jobs[ji1+1].type){
					sum -= t_large_setup;
				}
				if(ji1+1<machine_states[mi1].jobs.size() && machine_states[mi2].jobs[ji2].type != machine_states[mi1].jobs[ji1+1].type 
					&& machine_states[mi1].jobs[ji1].type == machine_states[mi1].jobs[ji1+1].type){
					sum += t_large_setup;
				}
			}
			return sum;
		}
		
		
		bool decide_reschedule(float dH, float T, double r){
			if(exp(-dH/T)>=r){
				return true;
			}
			else{
				return false;
			}
		}
		
		float calc_tardiness(){
			float sum = 0.0;
			float delay;
			for(int n = 0; n<num_machines; n++){
				delay = 0.0;
				for(int i = 0; i<machine_states[n].jobs.size(); i++){
					sum += cut(machine_states[n].jobs[i].t_smd + delay - machine_states[n].jobs[i].due_date);
					delay += machine_states[n].jobs[i].t_smd;
					if(i+1<machine_states[n].jobs.size()){
						if(machine_states[n].jobs[i].type != machine_states[n].jobs[i+1].type){
							//sum += t_large_setup + t_small_setup;
							delay += t_large_setup;
						}
						else{
							//sum += t_small_setup;
							delay += t_small_setup;
						}
					}
				}
			}
			return sum;
		}
		
		float calc_true_tardiness(){
			float sum = 0.0;
			float delay;
			for(int n = 0; n<num_machines; n++){
				delay = 0.0;
				for(int i = 0; i<machine_states[n].jobs.size(); i++){
					sum += cut_true_tardiness(machine_states[n].jobs[i].t_smd + delay - machine_states[n].jobs[i].due_date);
					delay += machine_states[n].jobs[i].t_smd;
					if(i+1<machine_states[n].jobs.size()){
						if(machine_states[n].jobs[i].type != machine_states[n].jobs[i+1].type){
							//sum += t_large_setup + t_small_setup;
							delay += t_large_setup;
						}
						else{
							//sum += t_small_setup;
							delay += t_small_setup;
						}
					}
				}
			}
			return sum;
		}
		
		float calc_number_large_setups(){
            int sum = num_machines;
            for(int n = 0; n<num_machines; n++){
                for(int i = 1; i<machine_states[n].jobs.size(); i++){
                    if(machine_states[n].jobs[i].type != machine_states[n].jobs[i-1].type){
                        sum += 1;
                    }
                }
            }
            return sum;
        }
        
        float calc_makespan(){
            float makespan = 0.0;
            for(int n = 0; n<num_machines; n++){
                float sum = 0.0;
                for(int i = 0; i<machine_states[n].jobs.size(); i++){
                    sum += machine_states[n].jobs[i].t_smd;
					if(i+1<machine_states[n].jobs.size()){
						if(machine_states[n].jobs[i].type != machine_states[n].jobs[i+1].type){
							sum += t_large_setup;
						}
						else{
							sum += t_small_setup;
						}
					}
				}
				if(sum>makespan){
                    makespan = sum;
                }
                    
            }
            return makespan;
        }
        
        float calc_diff_makespan(){
            float max_makespan = 0.0;
            float min_makespan = -1.0;
            
            for(int n = 0; n<num_machines; n++){
                float sum = 0.0;
                for(int i = 0; i<machine_states[n].jobs.size(); i++){
                    sum += machine_states[n].jobs[i].t_smd;
					if(i+1<machine_states[n].jobs.size()){
						if(machine_states[n].jobs[i].type != machine_states[n].jobs[i+1].type){
							sum += t_large_setup;
						}
						else{
							sum += t_small_setup;
						}
					}
				}
				if(sum>max_makespan){
                    max_makespan = sum;
                }
                if(1.0/sum > 1.0/min_makespan){
                    min_makespan = sum;
                }
                //cout << '\n' << sum << '\n';    
            }
            return (max_makespan-min_makespan);
        }
        
        float calc_late_jobs(){
            int sum = 0;
			float delay;
			for(int n = 0; n<num_machines; n++){
				delay = 0.0;
				for(int i = 0; i<machine_states[n].jobs.size(); i++){
					if(machine_states[n].jobs[i].t_smd + delay - machine_states[n].jobs[i].due_date > 0.0){
                        sum += 1;
                    }
                    delay += machine_states[n].jobs[i].t_smd;
					if(i+1<machine_states[n].jobs.size()){
						if(machine_states[n].jobs[i].type != machine_states[n].jobs[i+1].type){
							delay += t_large_setup;
						}
						else{
							delay += t_small_setup;
						}
					}
				}
			}
			return sum;
		}
};

int main(){
    const gsl_rng_type * rtype;
    gsl_rng * r;
    gsl_rng_env_setup();
    rtype = gsl_rng_default;
    r = gsl_rng_alloc (rtype);
    double T = 0;
    production_state p, p_opt, p_init, p_no_transient, p_opt_global;
    job j;
    int ji1, ji2, mi1, mi2;
    float T_start = 10000;
    float T_end = T_start/2;
    float T_incr= (T_start-T_end)/500;
    float T_opt, min;
    int help_id, help_machine;

    // read job_list from file and assign jobs one after another onto machines
    ostringstream fin0;
    fin0 << "data/J100S10.dat";
    cout << "File: " << fin0.str() << '\n';
    ifstream job_list_in(fin0.str().c_str());
    job_list_in >> j.id >> j.due_date >> j.type >> j.t_smd;
    int count = 0;
    while(!job_list_in.eof()){
            p.machine_states[count%num_machines].jobs.push_back(j);
            count++;
            job_list_in >> j.id >> j.due_date >> j.type >> j.t_smd;
    }
    job_list_in.close();
    
    /*for(int i=0; i<num_machines; i++){
        p.machine_states[i].print_jobs();
    }*/

    // remember initial state for reconstruction of genetic list
    p_init = p;
    
    // get rid of transient effects
    for(int t=0; t<10*N_transient; t++){
        // determine jobs to be possibly switched
        mi1 = num_machines*gsl_rng_uniform(r);
        mi2 = num_machines*gsl_rng_uniform(r);
        ji1 = (p.machine_states[mi1].jobs.size()+1)*gsl_rng_uniform(r);
        ji2 = (p.machine_states[mi2].jobs.size()+1)*gsl_rng_uniform(r);
        p.switch_production_states(mi1,ji1,mi2,ji2);
    }
    p_opt = p;
    p_no_transient = p;
    p_opt_global = p;
    
    // optimise the production list
    cout << "T = " << T << '\t' << "tardiness = " << p.calc_tardiness() << " t.u." << '\n';
    ostringstream fout1;
    fout1 << "results/dependence_on_scale_factor.dat";
    ofstream result(fout1.str().c_str());
    result << "scaling_factor" << '\t' << "Tardiness" << '\t' << "makespan" << '\t'
            << "# late jobs" << '\t' << "# large set-ups" << '\t' << "delta makespan" << '\n';
    float factor = 2.0;
    float global_min = p.calc_tardiness();
    //for(float factor=-100.0; factor<1000; factor = factor+ 0.1){
    int iter = 0;
    p = p_no_transient;
    p_opt = p;
    T_start = 10000;
    T_end = T_start/2;
    T_incr= (T_start-T_end)/500;
    min = p.calc_tardiness();
    while(iter<10){
        T = T_start;
        while(T > T_end){
            for(int t=0; t<N_transient; t++){
                // determine jobs to be possibly switched
                mi1 = num_machines*gsl_rng_uniform(r);
                mi2 = num_machines*gsl_rng_uniform(r);
                ji1 = (p.machine_states[mi1].jobs.size()+1)*gsl_rng_uniform(r);
                ji2 = (p.machine_states[mi2].jobs.size()+1)*gsl_rng_uniform(r);
                
                // determine whether to change states
                if(exp(-p.calculate_dH(mi1,ji1,mi2,ji2,factor)/T)>gsl_rng_uniform(r)){
                    p.switch_production_states(mi1,ji1,mi2,ji2);
                }
                
            }

            if(min>p.calc_tardiness()){
                p_opt = p;
                T_opt = T;
                cout << "T = " << T << '\t' << "tardiness = " << p.calc_tardiness() << " t.u." << '\n';
                min = p.calc_tardiness();
                
                // further optimise the state
                for(int t=0; t<int(1*N_transient); t++){
                    // determine jobs to be possibly switched
                    mi1 = num_machines*gsl_rng_uniform(r);
                    mi2 = num_machines*gsl_rng_uniform(r);
                    ji1 = (p.machine_states[mi1].jobs.size()+1)*gsl_rng_uniform(r);
                    ji2 = (p.machine_states[mi2].jobs.size()+1)*gsl_rng_uniform(r);
                    
                    // determine whether to change states
                    if(exp(-p.calculate_dH(mi1,ji1,mi2,ji2,factor)/T)>gsl_rng_uniform(r)){
                        p.switch_production_states(mi1,ji1,mi2,ji2);
                        if(min>p.calc_tardiness()){
                            p_opt = p;
                            min = p.calc_tardiness();
                            cout << "T = " << T << '\t' << "tardiness = " << p.calc_tardiness() << " t.u." << '\n';
                        }
                    }
                }
            }
            T -= T_incr;
            p = p_opt;
        }
        T_start = T_end;
        T_end = T_start/2;
        T_incr = (T_start-T_end)/500;
        iter++;
        p = p_opt;
        if(global_min > p_opt.calc_tardiness()){
            global_min = p_opt.calc_tardiness();
            p_opt_global = p_opt;
        }
    }
    result << factor << '\t' << p_opt.calc_true_tardiness() << '\t' << p_opt.calc_makespan() << '\t'
        << p_opt.calc_late_jobs() << '\t' << p_opt.calc_number_large_setups() << '\t' 
        << p_opt.calc_diff_makespan() << '\n';
    std::cerr << std::fixed;
    std::cerr << std::setprecision(5);
    cerr << (factor+100)/(1000+100) << '\r';
    //}
    result.close();

    // further optimise the state
    T = T_opt;
    p_opt = p_opt_global;
    p = p_opt;
    //float factor = 2;
    float min_makespan = p.calc_makespan();
    bool zero_tardy = false;
    
    for(int t=0; t<int(1000*N_transient); t++){
        // determine jobs to be possibly switched
        p = p_opt;
        mi1 = num_machines*gsl_rng_uniform(r);
        mi2 = num_machines*gsl_rng_uniform(r);
        ji1 = (p.machine_states[mi1].jobs.size()+1)*gsl_rng_uniform(r);
        ji2 = (p.machine_states[mi2].jobs.size()+1)*gsl_rng_uniform(r);
            
        // determine whether to change states
        if(exp(-p.calculate_dH(mi1,ji1,mi2,ji2,factor)/T)>gsl_rng_uniform(r)){
            p.switch_production_states(mi1,ji1,mi2,ji2);
            if(min>p.calc_tardiness()){
                if(!zero_tardy && p.calc_true_tardiness() == 0.0){
                    p_opt = p;
                    zero_tardy = true;
                    min = p.calc_tardiness();
                    min_makespan = p.calc_makespan();
                    cout << "T = " << T << '\t' << "tardiness = "
                        << p.calc_true_tardiness() << " t.u." << '\t'
                        << "makespan = " << p.calc_makespan() << " t.u." << '\n';

                }
                
                else if(zero_tardy && p.calc_true_tardiness() == 0.0){
                    if(min_makespan > p.calc_makespan()){
                        p_opt = p;
                        min = p.calc_tardiness();
                        min_makespan = p.calc_makespan();
                        //cout << "T = " << T << '\t' << "tardiness = " 
                        //     << p.calc_true_tardiness() << " t.u." << '\t'
                        //     << "makespan = " << p.calc_makespan() << " t.u." << '\n';

                    }
                }
                
                else if(!zero_tardy){
                    p_opt = p;
                    min = p.calc_tardiness();
                    min_makespan = p.calc_makespan();
                    //cout << "T = " << T << '\t' << "tardiness = " 
                    //     << p.calc_true_tardiness() << " t.u." << '\t'
                    //     << "makespan = " << p.calc_makespan() << " t.u." << '\n';
                }
            }
        }
    }
    //cout << '\n';
    p = p_opt;
    
    // write minimised production list into a file
    ostringstream fout2;
    fout2 << "results/optimised_job_list_sim_anneal.dat";
    ofstream opt_job(fout2.str().c_str());
    opt_job << "ID" << '\t' << "due_date" << '\t' << "family" << '\t' 
            << "t_smd" << '\t' << "t_aoi" << '\t' << "smd_machine" << '\n';

    for(int n =0; n<num_machines; n++){
        for(int i=0; i<p_opt.machine_states[n].jobs.size(); i++){
                //opt_job << n+1 << '\t' << p_opt.machine_states[n].jobs[i].id << '\n';
                j = p_opt.machine_states[n].jobs[i];
                opt_job << j.id << '\t' << j.due_date << '\t' << j.type << '\t' << j.t_smd << '\t' 
                                << 0.0000000000001 << '\t' << n+1 << '\n';
        }
    }
    opt_job.close();

    cout << "tardiness_simulated_annealing =  "
                << p.calc_true_tardiness() << '\n';
    cout << "large set-ups_simulated_annealing =  "
                << p.calc_number_large_setups() << '\n';
    cout << "makespan_simulated_annealing =  "
                << p.calc_makespan() << '\n';
    cout << "late jobs_simulated_annealing =  "
		 << p.calc_late_jobs() << '\n';
    cout << "diff makespan simulated_annealing =  "
                << p.calc_diff_makespan() << '\n';
    return 0;
}
