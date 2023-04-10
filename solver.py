#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
from docplex.cp.model import *
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import math


# In[23]:


LARGE_SETUP_PENALTY = 65
SMALL_SETUP_PENALTY = 20
N_MACHINES = 4
N_JOBS = 15
NAME = 'J{}S10'.format(N_JOBS)
DATA_PATH = './data/{}.pkl'.format(NAME) # "./data/tectron1.dat"


# In[24]:


def cut_true_tardiness(z: float) -> float:
    if z < 0:
        return 0
    else:
        return z


def calc_tardiness(vars, jobs):
    """ Calculates the tardiness.
    :param sequences: list of lists of vars
    :param jobs: contains the properties of jobs
    :return:
    """
    tardiness = 0.0
    for m in range(N_MACHINES):
        # as last_setup is initialized with -1 it will always add the large penalty to the first job
        # to counteract this, we initialize the delay with the negative of that penalty
        delay, last_setup = -LARGE_SETUP_PENALTY, -1
        for j, job in enumerate(jobs):
            # setup_equal = 1 if both setups are equal, else 0
            # setup_equal = (job['type'] / last_setup) % 1
            setup_equal = job['type'] == last_setup
            # only add setup penalty if (setup_equal=0 - vars[j][m]=1)
            delay += vars[j][m] * (job['pt'] + SMALL_SETUP_PENALTY)
            delay += (vars[j][m] - setup_equal * vars[j][m]) * (LARGE_SETUP_PENALTY - SMALL_SETUP_PENALTY)
            tardiness += vars[j][m] * max(0, delay - job['dd'])
            # if vars[j][m]=1 --> last_setup= job['type'] else -1
            last_setup = job['type'] * vars[j][m] + vars[j][m] - 1
    return tardiness


def calc_number_large_setups(sequences):
    """ Calculates the number of large setups.
    :param sequences: list of lists of job properties
    :return:
    """
    n_setups = len(sequences)
    for m in range(len(sequences)):
        for i in range(1, len(sequences)[m]):
            if sequences[m][i].type != sequences[m][i - 1].type:
                n_setups += 1
    return n_setups


def calc_makespan(vars, jobs):
    """ Calculates the maximum makespan.
    :param sequences: list of lists of vars
    :param jobs: contains the properties of jobs
    :return:
    """
    makespan = [0 for _ in range(N_MACHINES)]
    for m in range(N_MACHINES):
        last_setup = -1
        for j, job in enumerate(jobs):
            # setup_equal = 1 if both setups are equal, else 0
            # setup_equal = (job['type'] / last_setup) % 1
            setup_equal = job['type'] == last_setup
            # only add setup penalty if (setup_equal=0 - vars[j][m]=1)
            makespan[m] += vars[j][m] * (job['pt'] + SMALL_SETUP_PENALTY)
            makespan[m] += (vars[j][m] - setup_equal * vars[j][m]) * (LARGE_SETUP_PENALTY - SMALL_SETUP_PENALTY)
            # if vars[j][m]=1 --> last_setup= job['type'] else -1
            last_setup = job['type'] * vars[j][m] + vars[j][m] - 1
    return max(makespan)


def evaluate(sequences):
    goodness = calc_tardiness(sequences)
    goodness += calc_makespan(sequences)
    goodness += calc_number_large_setups(sequences)
    return goodness


def load_data(path="tectron1.dat"):
    if 'tectron' in path:
        jobs = []
        with open(path) as job_list_in:
            for count, line in enumerate(job_list_in.readlines()):
                id, due_date, type, p_time = line.split()
                jobs.append({
                    'id': int(id), 'dd': float(due_date),
                    'type': int(type), 'pt': float(p_time)
                })
    else:
        jobs = pd.read_pickle(path).to_dict('records')
    return jobs


def init_solver(jobs):
    mdl = CpoModel()
    # add bin decision variables
    stack = []
    for j in range(len(jobs)):
        row = [mdl.integer_var(0, 1, '{}:{}'.format(j, m)) for m in range(N_MACHINES)]
        # every job needs to be mapped to exactly one machine constraint
        mdl.add(mdl.sum(row) == 1)
        stack.append(row)
    return mdl, stack


def gantt(sequences, name='gantt', title=''):
    """
    :param sequences: list of lists of job properties
    :return:
    """
    df = []
    cumulated_pt_prior = [0. for _ in range(N_MACHINES)]
    for m in range(len(sequences)):
        for j, job in enumerate(sequences[m]):
            df.append({
                'Task': m,
                'Start': cumulated_pt_prior[m],
                'Finish': cumulated_pt_prior[m] + job['pt'],
                'Delay': max(0, cumulated_pt_prior[m] + job['pt'] - job['dd']),
                'Delta': job['pt'],
                'Name': job['id']
            })
            cumulated_pt_prior[m] += job['pt'] + SMALL_SETUP_PENALTY
            if j < len(sequences[m]) - 1 and sequences[m][j+1]['type'] != job['type']:
                cumulated_pt_prior[m] += LARGE_SETUP_PENALTY - SMALL_SETUP_PENALTY

    df = pd.DataFrame(df)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Delay", #text="Name",
                     color_continuous_scale='Hot')#, range_color=[0, 20000], range_x=[0, 20000])
    fig.update_layout(xaxis_title="Timesteps", yaxis_title="Machine")
    # fig.update_traces(showlegend=False)
    fig.update_annotations(align='center')
    fig.layout.xaxis.type = 'linear'
    fig.layout.yaxis.tickvals = [m for m in range(N_MACHINES)]
    fig.update_layout(template="plotly_white")
    fig.update_layout(title=title)
    # fig.update_coloraxes(colorbar_tickmode='array')
    # fig.update_coloraxes(colorbar_tickvals=[0,100])

    # this is necessary for a int/float x axis
    for i, d in enumerate(fig.data):
        fig.data[0].x = df['Delta'].tolist()

    return fig


# In[ ]:


jobs = load_data(path=DATA_PATH)[:N_JOBS]
solver, vars = init_solver(jobs)

#calculate sum of all job durations for every machine
# obj_var = [sum(vars[j][m] * job['pt']
#                for j, job in enumerate(jobs)) for m in range(N_MACHINES)]
obj_var = calc_tardiness(vars, jobs) #+ calc_makespan(vars, jobs)

# minimize the max duration of all machines
solver.minimize(solver.min(obj_var))

msol = solver.solve()

sequences = [[] for _ in range(N_MACHINES)]
solution = msol.solution.var_solutions_dict
for i, key in enumerate(solution):
    if i % 2:
        continue
    element = msol.solution.var_solutions_dict[key]
    if element.value == 1:
        job, machine = element.expr.name.split(":")
        sequences[int(machine)].append(jobs[int(job)])


# In[ ]:


pd.DataFrame(sequences).to_pickle('./solutions/solver/{}.pkl'.format(NAME))


# In[ ]:


def calc_true_tardiness(sequence):
    sum = 0.0
    for m in range(N_MACHINES):
        delay = 0.0
        for j in range(len(sequence[m])):
            sum += max(0, sequence[m][j]['pt'] + delay - sequence[m][j]['dd'])
            delay += sequence[m][j]['pt'] + SMALL_SETUP_PENALTY
            if j > 0 and sequence[m][j]['type'] != sequence[m][j-1]['type']:
                delay += LARGE_SETUP_PENALTY - SMALL_SETUP_PENALTY
    return sum


def calc_true_makespan(sequence):
    makespan = [0. for _ in range(N_MACHINES)]
    for m in range(N_MACHINES):
        for j in range(len(sequence[m])):
            makespan[m] += sequence[m][j]['pt'] + SMALL_SETUP_PENALTY
            if j > 0 and sequence[m][j]['type'] != sequence[m][j-1]['type']:
                makespan[m] += LARGE_SETUP_PENALTY - SMALL_SETUP_PENALTY
    return max(makespan)


# In[ ]:


tardiness = calc_true_tardiness(sequences)
makespan = calc_true_makespan(sequences)
title = 'Makespan: {:.2f} | Tardiness {:.2f}'.format(makespan, tardiness)

fig = gantt(sequences, name='solver/gantt_{}'.format(NAME), title=title)

fig.write_image('figures/solver/gantt_{}.pdf'.format(NAME))
