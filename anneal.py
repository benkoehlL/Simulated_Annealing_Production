#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
from tqdm import tqdm

class Job:
    def __init__(self, id: int, type: int, due_date: float, p_time: float):
        self.id = int(id)
        self.type = int(type)
        self.due_date = float(due_date)
        self.p_time = float(p_time)

    def set_job_state(self, i: int, t: int, dd: float, p_time: float) -> None:
        self.id = int(i)
        self.type = int(t)
        self.due_date = float(dd)
        self.p_time = float(p_time)

    def __str__(self):
        return 'ID {} | Setup {} | PT {:.1f} | DD {:.1f}'.format(
            self.id, self.type, self.p_time, self.due_date)

class MachineState:
    def __init__(self):
        self._jobs = []

    def add(self, job):
        self._jobs.append(job)

    def remove(self, job):
        self._jobs.pop(job)

    def size(self):
        return len(self._jobs)

    def set_machine_state(self, ji: int, j: Job) -> None:
        self._jobs[ji].set_job_state(j.id, j.type, j.due_date, j.p_time)

    def get_jobs(self, hot_end=True):
        # add an empty job to the end to allow dH computation 
        # with sampled job positions at the end of queue
        jobs_copy = self._jobs.copy()
        if hot_end:
            jobs_copy.append(Job(0, 0, 0., 0.))
        return jobs_copy

    def __str__(self):
        print_statement = '----- Machine Queue -----'
        for job in self._jobs:
            print_statement += '\n' + job.__str__()
        return print_statement

def delay_scaling(delay: float, n: float) -> float:
    if delay < 0.0:
        return delay / n
    else:
        return n * delay

def cut_true_tardiness(z: float) -> float:
    if z < 0:
        return 0
    else:
        return z

def cut(z: float) -> float:
    if z < 0:
        return z / 10
    else:
        return 10 * z


# In[2]:


class ProductionState:
    def __init__(self, num_machines, t_small_setup=20, t_large_setup=65):
        self.num_machines = num_machines
        self.machine_states = [MachineState() for _ in range(num_machines)]
        self.t_large_setup = t_large_setup
        self.t_small_setup = t_small_setup

    def set_production_state(self, mi, ji, j):
        self.machine_states[mi].set_machine_state(ji, j)

    def switch_production_states(self, mi1, ji1, mi2, ji2):
        help_job = Job(0, 0, 0.0, 0.0)
        if ji1 < self.machine_states[mi1].size() and ji2 < self.machine_states[mi2].size():
            help_job.set_job_state(
                self.machine_states[mi1].get_jobs()[ji1].id,
                self.machine_states[mi1].get_jobs()[ji1].type,
                self.machine_states[mi1].get_jobs()[ji1].due_date,
                self.machine_states[mi1].get_jobs()[ji1].p_time
            )
            self.machine_states[mi1].get_jobs()[ji1].set_job_state(
                self.machine_states[mi2].get_jobs()[ji2].id,
                self.machine_states[mi2].get_jobs()[ji2].type,
                self.machine_states[mi2].get_jobs()[ji2].due_date,
                self.machine_states[mi2].get_jobs()[ji2].p_time
            )
            self.machine_states[mi2].get_jobs()[ji2].set_job_state(
                help_job.id,
                help_job.type,
                help_job.due_date,
                help_job.p_time
            )
        elif ji1 == self.machine_states[mi1].size() and ji2 < self.machine_states[mi2].size():
            self.machine_states[mi1].add(self.machine_states[mi2].get_jobs()[ji2])
            self.machine_states[mi2].remove(ji2)
        elif ji2 == self.machine_states[mi2].size() and ji1 < self.machine_states[mi1].size():
            self.machine_states[mi2].add(self.machine_states[mi1].get_jobs()[ji1])
            self.machine_states[mi1].remove(ji1)

    def calculate_dH(self, mi1, ji1, mi2, ji2, n):

        sum = 0.

        # delay1 = np.array([self.machine_states[mi1].get_jobs()[i].p_time for i in range(ji1)]).sum()
        # delay2 = np.array([self.machine_states[mi2].get_jobs()[i].p_time for i in range(ji2)]).sum()
        # sum += delay_scaling(delay1 + self.machine_states[mi2].get_jobs()[ji2].p_time - self.machine_states[mi2].get_jobs()[ji2].due_date, n)
        # sum += delay_scaling(delay2 + self.machine_states[mi1].get_jobs()[ji1].p_time - self.machine_states[mi1].get_jobs()[ji1].due_date, n)

        N1 = max(self.machine_states[mi1].size() - ji1, 1)
        N2 = max(self.machine_states[mi2].size() - ji2, 1)
        sum += delay_scaling((self.machine_states[mi1].get_jobs()[ji1].p_time - self.machine_states[mi2].get_jobs()[ji2].p_time) * N1, n)
        sum += delay_scaling((self.machine_states[mi2].get_jobs()[ji2].p_time - self.machine_states[mi1].get_jobs()[ji1].p_time) * N2, n)

        if self.machine_states[mi1].get_jobs()[ji1].type != self.machine_states[mi2].get_jobs()[ji2].type:

            if ji2-1 >= 0 and self.machine_states[mi1].get_jobs()[ji1].type == self.machine_states[mi2].get_jobs()[ji2-1].type and self.machine_states[mi2].get_jobs()[ji2].type != self.machine_states[mi2].get_jobs()[ji2-1].type:
                sum -= self.t_large_setup / n
            if ji2-1 >= 0 and self.machine_states[mi1].get_jobs()[ji1].type != self.machine_states[mi2].get_jobs()[ji2-1].type and self.machine_states[mi2].get_jobs()[ji2].type == self.machine_states[mi2].get_jobs()[ji2-1].type:
                sum += self.t_large_setup * n
            if ji1-1 >= 0 and self.machine_states[mi2].get_jobs()[ji2].type == self.machine_states[mi1].get_jobs()[ji1-1].type and self.machine_states[mi1].get_jobs()[ji1].type != self.machine_states[mi1].get_jobs()[ji1-1].type:
                sum -= self.t_large_setup / n
            if ji1-1 >= 0 and self.machine_states[mi2].get_jobs()[ji2].type != self.machine_states[mi1].get_jobs()[ji1-1].type and self.machine_states[mi1].get_jobs()[ji1].type == self.machine_states[mi1].get_jobs()[ji1-1].type:
                sum += self.t_large_setup * n

            if ji2+1 < self.machine_states[mi2].size() and self.machine_states[mi1].get_jobs()[ji1].type == self.machine_states[mi2].get_jobs()[ji2+1].type and self.machine_states[mi2].get_jobs()[ji2].type != self.machine_states[mi2].get_jobs()[ji2+1].type:
                sum -= self.t_large_setup / n
            if ji2+1 < self.machine_states[mi2].size() and self.machine_states[mi1].get_jobs()[ji1].type != self.machine_states[mi2].get_jobs()[ji2+1].type and self.machine_states[mi2].get_jobs()[ji2].type == self.machine_states[mi2].get_jobs()[ji2+1].type:
                sum += self.t_large_setup * n
            if ji1+1 < self.machine_states[mi1].size() and self.machine_states[mi2].get_jobs()[ji2].type == self.machine_states[mi1].get_jobs()[ji1+1].type and self.machine_states[mi1].get_jobs()[ji1].type != self.machine_states[mi1].get_jobs()[ji1+1].type:
                sum -= self.t_large_setup / n
            if ji1+1 < self.machine_states[mi1].size() and self.machine_states[mi2].get_jobs()[ji2].type != self.machine_states[mi1].get_jobs()[ji1+1].type and self.machine_states[mi1].get_jobs()[ji1].type == self.machine_states[mi1].get_jobs()[ji1+1].type:
                sum += self.t_large_setup * n
        return sum / 100000. # todo for overflow errs due to low temperature

    @staticmethod
    def decide_reschedule(dH, T, r):
        if (math.exp(-dH/T) >= r):
            return True
        else:
            return False

    def calc_tardiness(self):
        sum = 0.0
        for n in range(self.num_machines):
            delay = 0.0
            for i in range(self.machine_states[n].size()):
                sum += cut(self.machine_states[n].get_jobs()[i].p_time + delay - self.machine_states[n].get_jobs()[i].due_date)
                delay += self.machine_states[n].get_jobs()[i].p_time
                if i + 1 < self.machine_states[n].size():
                    if self.machine_states[n].get_jobs()[i].type != self.machine_states[n].get_jobs()[i + 1].type:
                        delay += self.t_large_setup
                    else:
                        delay += self.t_small_setup
        return sum

    def calc_true_tardiness(self):
        sum = 0.0
        for n in range(self.num_machines):
            delay = 0.0
            for i in range(self.machine_states[n].size()):
                sum += cut_true_tardiness(self.machine_states[n].get_jobs()[i].p_time + delay - self.machine_states[n].get_jobs()[i].due_date)
                delay += self.machine_states[n].get_jobs()[i].p_time
                if i + 1 < self.machine_states[n].size():
                    if self.machine_states[n].get_jobs()[i].type != self.machine_states[n].get_jobs()[i + 1].type:
                        delay += self.t_large_setup
                    else:
                        delay += self.t_small_setup
        return sum

    def calc_number_large_setups(self):
        sum = self.num_machines
        for n in range(self.num_machines):
            for i in range(1, self.machine_states[n].size()):
                if self.machine_states[n].get_jobs()[i].type != self.machine_states[n].get_jobs()[i - 1].type:
                    sum += 1
        return sum

    def calc_makespan(self):
        makespan = 0.0
        for n in range(self.num_machines):
            sum = 0.0
            for i in range(self.machine_states[n].size()):
                sum += self.machine_states[n].get_jobs()[i].p_time
                if i + 1 < self.machine_states[n].size():
                    if self.machine_states[n].get_jobs()[i].type != self.machine_states[n].get_jobs()[i+1].type:
                        sum += self.t_large_setup
                    else:
                        sum += self.t_small_setup
            if sum > makespan:
                makespan = sum
        return makespan

    def calc_diff_makespan(self):
        max_makespan = 0.0
        min_makespan = -1.0

        for n in range(self.num_machines):
            sum = 0.0
            for i in range(self.machine_states[n].size()):
                sum += self.machine_states[n].get_jobs()[i].p_time
                if i + 1 < self.machine_states[n].size():
                    if self.machine_states[n].get_jobs()[i].type != self.machine_states[n].get_jobs()[i+1].type:
                        sum += self.t_large_setup
                    else:
                        sum += self.t_small_setup
            if sum > max_makespan:
                max_makespan = sum
            if 1.0 / sum > 1.0 / min_makespan:
                min_makespan = sum
        return max_makespan - min_makespan

    def calc_late_jobs(self):
        sum = 0
        for n in range(self.num_machines):
            delay = 0.0
            for i in range(self.machine_states[n].size()):
                if self.machine_states[n].get_jobs()[i].p_time + delay - self.machine_states[n].get_jobs()[i].due_date > 0.0:
                    sum += 1
                delay += self.machine_states[n].get_jobs()[i].p_time
                if i + 1 < self.machine_states[n].size():
                    if self.machine_states[n].get_jobs()[i].type != self.machine_states[n].get_jobs()[i+1].type:
                        delay += self.t_large_setup
                    else:
                        delay += self.t_small_setup
        return sum


# In[3]:


def load_and_assign(path="tectron1.dat", num_machines=4):
    # read job_list from file and assign jobs one after another onto machines
    p = ProductionState(num_machines)
    with open(path) as job_list_in:
        for count, line in enumerate(job_list_in.readlines()):
            id, due_date, type, p_time = line.split()
            job = Job(id, type, due_date, p_time)
            p.machine_states[count % num_machines].add(job)
    return p


def sample_job_machine(p, n_samples=2):
    m_samples = np.random.randint(0, len(p.machine_states), n_samples)
    # allow to over-index the job array to allow placing jobs at the end of queue
    j_samples = [np.random.randint(0, p.machine_states[m].size() + 1) for m in m_samples]
    return m_samples, j_samples


def warmup(p, n_transient=1000):
    # todo speed up by random permutation
    for _ in range(n_transient):
        # determine jobs to be possibly switched
        [mi1, mi2], [ji1, ji2] = sample_job_machine(p)
        p.switch_production_states(mi1, ji1, mi2, ji2)
    return p


def flip_gate(p, mi1, ji1, mi2, ji2, t, factor=2.):
    energy = p.calculate_dH(mi1, ji1, mi2, ji2, factor)
    energy = math.exp(-energy / t) if energy > 0. else 1.
    return energy > random.uniform(0, 1)


def transition_phase(p, t, n_transient=1000, factor=2.0,
                     tardiness_primal=None, p_incumbent=None):
    """ Performes a transition phase.
    :param p:
    :param t:
    :param n_transient:
    :param factor:
    :param tardiness_primal: (optional) adds incumbent plan update
    :param p_incumbent: (optional) adds incumbent plan update
    :return:
    """
    for _ in range(n_transient):
        [mi1, mi2], [ji1, ji2] = sample_job_machine(p)
        if flip_gate(p, mi1, ji1, mi2, ji2, t, factor=factor):
            p.switch_production_states(mi1, ji1, mi2, ji2)
            if tardiness_primal is not None:
                tardiness = p.calc_tardiness()
                if tardiness_primal > tardiness:
                    p_incumbent, tardiness_primal = p, tardiness
    # if the input parameters min_val and p_opt are None, then the second return value is None, too
    return p, p_incumbent, tardiness_primal


def simulated_annealing(p, t_start=10000, t_end=5000, incremental_steps=500, factor=2.):
    p_incumbent = p
    tardiness_primal = p.calc_tardiness()
    # the tempering scheme is piece-wise linear, this loops over the linear pieces
    for c in range(n_cylces):
        tardiness_update = 0
        # loop over the linear interpolation steps
        p_bar = tqdm(np.linspace(t_start, t_end, incremental_steps))
        for t in p_bar:
            p = transition_phase(p, t, n_transient=n_transient, factor=factor)[0]
            # update the primal and incumbent
            tardiness = p.calc_tardiness()
            if tardiness_primal > tardiness:
                tardiness_update += 1
                p_incumbent, tardiness_primal = p, tardiness
                # further optimise the state
                p, p_incumbent, tardiness_primal = transition_phase(p, t, n_transient=n_transient, factor=factor,
                                                                    tardiness_primal=tardiness_primal, p_incumbent=p_incumbent)
            # update vars for next iteration
            p_bar.set_description('{}/{} | H: {:.1f} | Q: {} | U: {}'.format(c + 1, n_cylces, tardiness_primal, [p.machine_states[m].size() for m in range(n_machines)], tardiness_update))
            p = p_incumbent
        t_start, t_end = t_end, t_end / 2
    return p_incumbent


def print_kpis(p):
    print('----- KPIs -----')
    print('Tardiness: ' + str(p.calc_true_tardiness()))
    print('Makespan: ' + str(p.calc_makespan()))
    print('Late Jobs: ' + str(p.calc_late_jobs()))
    print('Large Setups: ' + str(p.calc_number_large_setups()))
    print('Diff Makespan: ' + str(p.calc_diff_makespan()))


# In[4]:


n_machines = 4
n_transient=1000
t_large_setup=65
t_small_setup=20
t_start = 10000
t_end = t_start/2
incremental_steps = 500
n_cylces = 10
factor = 2.0

p = load_and_assign(path="data/tectron1.dat", num_machines=n_machines)
# remember initial state for reconstruction of genetic list
p_init = p
# get rid of transient effects
p = warmup(p, n_transient=10*n_transient)
# simulated annealing loop with piece-wise linear tempering scheme
p = simulated_annealing(p, t_start=t_start, t_end=t_end,
                        incremental_steps=incremental_steps, factor=factor)
#
print_kpis(p)

