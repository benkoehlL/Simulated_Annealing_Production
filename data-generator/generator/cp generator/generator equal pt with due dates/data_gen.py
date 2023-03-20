import numpy as np
import docplex.cp.utils_visu as visu
from docplex.cp.model import *
from io import StringIO
from tqdm import tqdm


NB_EXAMPLES = 1000


def calc_obj(x,proc_t,return_max=True):
    #calc value of objective function
    if return_max:
        return max(np.array(np.transpose(x)).dot(np.array(proc_t)))
    return np.array(np.transpose(x)).dot(np.array(proc_t))


def solve(proc_t,due_dates,rel_dates,nb_m):
    NbHouses = 1
    Deadline =  24*60
   
    Workers = ["M"+str(i) for i in range( nb_m)]
    
    Tasks = ["T"+str(i) for i in range( proc_t.shape[0])]

    Durations = proc_t
    ReleaseDate = rel_dates
    DueDate     = due_dates

    Skills=[]
    for w in Workers:
        for t in Tasks:
            Skills.append((w,t,1))
    """
    Precedences = [("masonry","carpentry"),("masonry","plumbing"),("masonry","ceiling"),
                   ("carpentry","roofing"),("ceiling","painting"),("roofing","windows"),
                   ("roofing","facade"),("plumbing","facade"),("roofing","garden"),
                   ("plumbing","garden"),("windows","moving"),("facade","moving"),
                   ("garden","moving"),("painting","moving")
                  ]
    """
    nbWorkers = len(Workers)
    Houses = range(NbHouses)
    mdl5 = CpoModel()
    tasks = {}
    wtasks = {}
    for h in Houses:
        for i,t in enumerate(Tasks):
            tasks[(h,t)] = mdl5.interval_var(start=[0,Deadline], size=Durations[i])
        for i,s in enumerate(Skills):
            #print(DueDate[i%len(Tasks)])
            wtasks[(h,s)] = mdl5.interval_var(start=[0,DueDate[i%len(Tasks)]],optional=True)
    for h in Houses:
        for t in Tasks:
            mdl5.add( mdl5.alternative(tasks[h,t], [wtasks[h,s] for s in Skills if s[1]==t]) )
    for w in Workers:
        mdl5.add( mdl5.no_overlap([wtasks[h,s] for h in Houses for s in Skills if s[0]==w]) )
    mdl5.add(
        mdl5.minimize( 
            mdl5.max( mdl5.end_of(wtasks[h,s]) for h in Houses for s in Skills)
        )
    )
    # Solve the model
    print("\nSolving model....")
    #f = open("solver_log.txt", "r", encoding="utf-8")
    solver_log_stream = StringIO()
    msol5 = mdl5.solve(log_output=solver_log_stream)
    print("done")
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
    return sol_dict,msol5.get_objective_values()[0]

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
        
    
def simulation(pt_mean,pt_std,nb_t_range,nb_m_range,nb_data,pt_int=False,return_dur=False):
    # init parameters
    # pt_mean: mean of processtime (duration of task on every machine)
    # pt_std:std of processtime 
    # number of task range f.e. [2,10] means uniform distribution between 2 and 10 tasks
    # number of machine range f.e. [2,10] means uniform distribution between 2 and 10 machines
    # nb_data = number of datasamples which should get generated
    # pt_int = (boolean) if process time is int or float (default: False)
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
        due_dates=gen_due_date(proc_t,nb_m,nb_t)
        x.append([{'pt':proc_t,'dd':due_dates}])
       
        print(due_dates)
        sol=solve(proc_t,due_dates,rel_dates,nb_m)
        y.append([sol])
        if return_dur:
            dur.append(sol[-1])
    if return_dur:
        return x,y,dur
    return x,y
   

for i in tqdm(range(NB_EXAMPLES)):
    x,y=simulation(30,7,[16,17],[4,5],1,True)
    with open('../../../../../res/ds/processed/setup_types/x_data.csv', mode="ab+") as f:
        np.savetxt(f, x[0][0].get('pt'), delimiter=",", fmt='%.1e')
        np.savetxt(f, x[0][0].get('dd'), delimiter=",", fmt='%.1e')
    with open('../../../../../res/ds/processed/setup_types/y_data.csv', mode="ab+") as f:
        #print(dict(y[0][0][0])[0])
        for m in dict(y[0][0][0]).keys():
            np.savetxt(f, np.array(y[0][0][0][m]), delimiter=",", fmt='%.1e')
print('done')
