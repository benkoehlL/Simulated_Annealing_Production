from tqdm import tqdm
import numpy as np
import docplex.cp.utils_visu as visu
from docplex.cp.model import *
import collections
from io import StringIO

#import docplex.cp.utils_visu as visu
#import matplotlib.pyplot as plt
#%matplotlib inline
#from pylab import rcParams


def calc_obj(x,proc_t,return_max=True):
    #calc value of objective function
    if return_max:
        return max(np.array(np.transpose(x)).dot(np.array(proc_t)))
    return np.array(np.transpose(x)).dot(np.array(proc_t))


def solve(proc_t,due_dates,rel_dates,nb_m,setup_types,return_model,return_vars):
    """
    REF: https://ibmdecisionoptimization.github.io/tutorials/html/Scheduling_Tutorial.html
    
    Inspired by House Building Problem, hence the following terms are defined:
    
    Worker => Machines
    Tasks => Jobs
    Houses => 1 (does not fulfill a purpose here)
    Skills => each machine has the skill to process each job
    Deadline => a day
    """
    NbHouses = 1
    Deadline =  24*60
   
    Workers = ["M"+str(i) for i in range( nb_m)]
    
    Tasks = ["T"+str(i) for i in range( proc_t.shape[0])]

    Durations = proc_t
    ReleaseDate = rel_dates
    DueDate = due_dates

    Skills=[]
    for w in Workers:
        for t in Tasks:
            Skills.append((w,t,1))
            
    nbWorkers = len(Workers)
    Houses = range(NbHouses)
    mdl5 = CpoModel()
    tasks = {}
    wtasks = {}
    wseq = {}
    transitionTimes = transition_matrix(len(Tasks))
    for h in Houses:
        for i,t in enumerate(Tasks):
            # add interval decision var for each job, range from 0 to Deadline, and fixed length of PT
            # thus each task has to fit with its pt in the fictional deadline (max time)
            tasks[(h,t)] = mdl5.interval_var(start=[0,Deadline], size=Durations[i])
            
            # Add transition times between tasks, which do NOT share the same setup time
            for j,t2 in enumerate(Tasks):
                if np.dot(setup_types[i], setup_types[j]) == 0:
                    transitionTimes.set_value(i, j, 10)
                else:
                    transitionTimes.set_value(i, j, 0)
            
        for i,s in enumerate(Skills):
            # looping over each possible combi of machine and job (skill)
            # add interval decision var for each combi, range from 0 to DD for each job.
            # Thus each job on each machine must be processed within a range of 0 upto its DD.
            # this is optional, thus do not have to be fulfilled?
            wtasks[(h,s)] = mdl5.interval_var(start=[0,DueDate[i%len(Tasks)]],optional=True)
        for w in Workers:
#             print([int(s[1][1:]) for s in Skills if s[0] == w] )
            wseq[w] = mdl5.sequence_var([wtasks[(h,s)] for s in Skills if s[0] == w],
                                        types=[int(s[1][1:]) for s in Skills if s[0] == w ])
#         for i,w in enumerate(Workers):
#             # create a sequence of workers task, which is ordered by the solver
#             wseq[(h,w,t)] = mdl5.sequence_var([wtasks[(h,s)] for s in Skills if s[0] == w])
    for h in Houses:
        for t in Tasks:
            # add constraint such that if j is in the solution space, then there is exactly one job on a machine.
            mdl5.add( mdl5.alternative(tasks[h,t], [wtasks[h,s] for s in Skills if s[1]==t]) )
    for w in Workers:
        # add constraint which assumes no overlapping of two interval variables
        # loop over machines and thus enforce that the same jobs are not assinged to multiple machines.
#         mdl5.add( mdl5.no_overlap([wtasks[h,s] for h in Houses for s in Skills if s[0]==w]) )
        
        # add overlap constraint to enforce transitions is required
        mdl5.add( mdl5.no_overlap(wseq[w], transitionTimes))
        
#     # This will sort the workers tasks by their end time (dict first key elem)
#     # Hence, we will know in which order the tasks are processed on each worker
#     ordered_wtasks = []
#     for w in Workers:
#         worker_tasks = {k: wtasks.get(k) for k in wtasks.keys() if w == k[1][0]}
#         worker_tasks = collections.OrderedDict(sorted(worker_tasks.items(), key=lambda item: item[0][0]))
#         ordered_wtasks.append(worker_tasks)
    
#     # Now lets compute the number of setup changes by computing the dot product of the current
#     # task on the worker and its previous, if its 1, then the same setup applies, otherwise 0.
#     # Since we are interessed in the occurancce of changes, we will substract it from one.
#     setup_changes = np.zeros(len(Workers))
#     for w, worker in enumerate(Workers):
#         for t, task in enumerate(ordered_wtasks[w]):
#             setup_task = setup_types[t]
#             setup_worker = setup_types[t-1] if t > 1 else np.zeros_like(setup_types[t])
#             setup_changes[w] += 1 - np.dot(setup_task, setup_worker)
            
    # finally add the main objective in form of a minimization of the maximal end times of all job machine combos
    # thus optimizing towards the makespan minimization 
#     weight = 0.7
    print(transitionTimes)
    print('\n')
    print(wseq)
    print('\n')
    print(wtasks[(h,s)])
    mdl5.add(
        mdl5.minimize( 
            mdl5.max(mdl5.end_of(wtasks[h,s]) for h in Houses for s in Skills)
#             weight * mdl5.max(mdl5.end_of(wtasks[h,s]) for h in Houses for s in Skills) \
#             + (1 - weight) * mdl5.sum(setup_changes)
        )
    )
    
    # Solve the model
    print("\nSolving model....")
    solver_log_stream = StringIO()
    msol5 = mdl5.solve(log_output=solver_log_stream)
    print("done")
    
    # transform model solution to a format, which can be handled afterwards
    if msol5 is not None:
        print("Cost will be "+str( msol5.get_objective_values()[0] ))
    worker_idx = {w : i for i,w in enumerate(Workers)}
    worker_tasks = [[] for w in range(nbWorkers)]  # Tasks assigned to a given worker
    for h in Houses:
        for s in Skills:
            worker = s[0]
            wt = wtasks[(h,s)]
            worker_tasks[worker_idx[worker]].append(wt)
    sol_dict = {k: [] for k in range(nb_m)}
    
    for i,w in enumerate(Workers):
        visu.sequence(name=w)
        for k,t in enumerate(worker_tasks[worker_idx[w]]):
            wt = msol5.get_var_solution(t)
            #print(wt)
            if wt.is_present():
                sol_dict[i].append((k,wt.start,wt.end))
    for i,w in enumerate(Workers):
        sol_dict[i].sort(key=lambda tup: tup[1])
    
    return_list = [sol_dict,msol5.get_objective_values()[0]]
    if return_model:
        return_list.append(msol5)
    if return_vars:
        return_list.append(transitionTimes)
    return return_list

def gen_due_date(max_proc_t,nb_m,nb_t,pt_int=True):
    proc_l=[]
    fac_vec=np.arange(np.ceil(nb_t/2),dtype=np.int64)+1
    fac_vec=np.repeat(fac_vec, 2)
    np.random.shuffle(fac_vec)
    print(fac_vec)
    
    for i in range(nb_t):
        t=np.random.normal(loc=(fac_vec[i]*max_proc_t[i]+max_proc_t[i]/2),scale=max_proc_t[i]/6)
        if pt_int:
            proc_l.append(int(t))
        else:
            proc_l.append(t)
    return np.array(proc_l)

def gen_setup_types(nb_t, nb_s):
    # nb_s is the intended number of different setup tasks
    setup_types = [np.random.randint(0, nb_s - 1) for _ in range(nb_t)]
    setup_types = np.array(setup_types).reshape(-1)
    setup_types = np.eye(nb_s)[setup_types]
    return setup_types
        

def simulation(pt_mean,
               pt_std,
               nb_t_range,
               nb_m_range,
               nb_data,
               nb_s,
               pt_int=False,
               return_dur=False,
               return_pts=False):
    # init parameters
    # pt_mean: mean of processtime (duration of task on every machine)
    # pt_std:std of processtime 
    # number of task range f.e. [2,10] means uniform distribution between 2 and 10 tasks
    # number of machine range f.e. [2,10] means uniform distribution between 2 and 10 machines
    # nb_data = number of datasamples which should get generated
    # pt_int = (boolean) if process time is int or float (default: False)
    # nb_s = number of setup types 
    # return_dur = process time get's return (for statistics, default:False)
    x=[]
    y=[]
    if return_dur:
        dur=[]
    for i in range(nb_data):
        
        nb_m=np.random.randint(nb_m_range[0],nb_m_range[1])
        nb_t=np.random.randint(nb_t_range[0],nb_t_range[1])
        
        print('it:',i,'\t#m:',nb_m,'\t#t:',nb_t)
        if pt_int:
            #proc_t=np.random.normal(loc=pt_mean,scale=pt_std,size=(nb_m,nb_t))
            proc_t=np.random.normal(loc=pt_mean,scale=pt_std,size=(nb_t))
            proc_t=proc_t.astype(np.int64)
        else:
            #proc_t=np.random.normal(loc=pt_mean,scale=pt_std,size=(nb_m,nb_t))
            proc_t=np.random.normal(loc=pt_mean,scale=pt_std,size=(nb_t))
        print(proc_t)
        
        rel_dates = np.zeros(nb_t,dtype=np.int32)
        #due_dates=gen_due_date(np.amax(proc_t, axis=0),nb_m,nb_t)
        due_dates = gen_due_date(proc_t,nb_m,nb_t)
        setup_types = gen_setup_types(nb_t, nb_s)
        
        x.append([{
            'pt':proc_t,
            'dd':due_dates,
            'st':setup_types
        }])
        
        print(due_dates)
        print(setup_types)
        
        sol=solve(proc_t, due_dates, rel_dates, nb_m, setup_types,
                  return_model=True,return_vars=True)
        sol, model, transitionTimes = sol[:-2], sol[-2], sol[-1]
        
        y.append([sol])
        if return_dur:
            dur.append(sol[-1])
    if return_dur:
        return x,y,dur
    if return_pts:
        return x,y,proc_t,model,transitionTimes
    return x,y
    

# pt_mean = 30
# pt_std = 7
# nb_t_range = [10,11]#[16,17]
# nb_m_range = [2,3]#[4,5]
# nb_data = 1
# nb_s = 4
# pt_int = False,
# return_dur = False
# return_pts = True
# return_model = True
# return_trans = True
# x,y,pts,model,transT = simulation(
#                         pt_mean,
#                         pt_std,
#                         nb_t_range,
#                         nb_m_range,
#                         nb_data,
#                         nb_s,
#                         pt_int,
#                         return_dur,
#                         return_pts)

NB_EXAMPLES = 1000
for i in range(NB_EXAMPLES):
    print('{0} / {1}'.format(i, NB_EXAMPLES))
    x,y=simulation(30,7,[4,13],[3,5],1,5,True)
    with open('../../../../../res/ds/processed/setup_types/x_data.csv', mode="ab+") as f:
        data = [x[0][0].get('dd'), x[0][0].get('pt')]
        np.savetxt(f, data, delimiter=",", fmt='%.1e')
        np.savetxt(f, x[0][0].get('st'), delimiter=",", fmt='%.1e')
    with open('../../../../../res/ds/processed/setup_types/y_data.csv', mode="ab+") as f:
        for m in dict(y[0][0][0]).keys():
            np.savetxt(f, [m, ], delimiter=",", fmt='%.1e')
            np.savetxt(f, np.array(y[0][0][0][m]), delimiter=",", fmt='%.1e')
print('done')
