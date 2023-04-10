import numpy as np
from docplex.cp.model import *
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import time
import math

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
                     color_continuous_scale='Hot', range_color=[0, 20000], range_x=[0, 20000])
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


N_MACHINES = 4
SMALL_SETUP_PENALTY = 25
LARGE_SETUP_PENALTY = 65

problem_path = './data/J100S10.dat'
solution_paths = [
    # './results/optimised_job_list_EDD_full.dat',
    # './results/optimised_job_list_ant_colony_full.dat',
    # './results/optimised_job_list_genetic_full.dat',
    # './results/optimised_job_list_A3C_full.dat',
    './results/optimised_job_list_sim_anneal.dat'
]

for ID, name in enumerate(['SA']): # 'EDD', 'Ant', 'Gen', 'A3C',

    jobs = []
    with open(problem_path) as job_list_in:
        for count, line in enumerate(job_list_in.readlines()):
            id, due_date, type, p_time = line.split()
            jobs.append({
                'id': int(id), 'dd': float(due_date),
                'type': int(type), 'pt': float(p_time)
            })


    solution_path = solution_paths[ID]
    sequences = [[] for _ in range(N_MACHINES)]
    with open(solution_path) as job_list_in:
        for count, line in enumerate(job_list_in.readlines()):
            if count == 0:
                continue
            j, _, _, _, _, m = line.split()
            sequences[int(m) - 1].append(jobs[int(j) - 1])

    tardiness = calc_true_tardiness(sequences)
    makespan = calc_true_makespan(sequences)
    title = 'Makespan: {:.2f} | Tardiness {:.2f}'.format(makespan, tardiness)
    fig = gantt(sequences, name='sim_anneal', title=title)

    fig.write_image('./figures/{}.pdf'.format(name))
    fig.show()
