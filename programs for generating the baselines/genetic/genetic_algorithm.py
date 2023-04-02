import numpy as np
import matplotlib.pyplot as plt
import timeit
from lib.env.hfs_env_genetic import HFS_Env

def initialization(size, schedulers_list_smd, schedulers_list_aoi, max_switch, num_switch):
    '''initializes a population of random chromosomes
    
    Parameters:
    -----------
    size:
        size of the population, number of chromosomes to be created
    schedulers_list_smd:
        list of implemented schedulers for SMD stage
    schedulers_list_aoi:
        list of implemented schedulers for AOI stage
    max_switch:
        maximum switching point
    num_switch:
        initial number of dispatching rule switches
    '''

    population = []

    # create chromosomes
    for _ in range(size):
        schedulers_smd = np.random.choice(schedulers_list_smd, (num_switch,1))
        schedulers_aoi = np.random.choice(schedulers_list_aoi, (num_switch,1))
        periods = np.random.randint(100, max_switch, (num_switch,1))
        chromosome = np.concatenate((schedulers_smd, schedulers_aoi, periods), axis=1)
        population.append(chromosome)

    return population

def selection(pop, scores, k=3):
    '''performs a tournament selection with k chromosomes and returns the best one
    
    Parameters
    ----------
    pop:
        population of chromosomes
    scores:
        fitness of the chromosmes
    k:
        torunament size
    '''

    # first random selection
    selection_idx = np.random.randint(0, len(pop))
    for idx in np.random.randint(0, len(pop), size=k-1):
        # check if better (perform a tournament)
        if scores[idx] < scores[selection_idx]:
            selection_idx = idx

    return pop[selection_idx]


def crossover(p1, p2, r_cross):
    '''recombines two chromosomes and creates two new chromosomes
    
    Parameters:
    -----------
    p1, p2:
        parent chromosomes
    r_cross:
        probability for crossover of the chromosome
    '''

    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()

    # check for recombination
    if np.random.random_sample() < r_cross and len(c1) > 2 and len(c2) > 2:
        # select a crossover point
        pt = np.random.randint(1, len(p1)-1)
        # perform crossover
        c1 = np.concatenate((p1[:pt,:], p2[pt:,:]), axis=0)
        c2 = np.concatenate((p2[:pt,:], p1[pt:,:]), axis=0)

    return [c1, c2]

def mutation_drop(chromosome, r_mut):
    '''deletes a random number of orders of the chromosome
    
    Parameters:
    -----------
    chromosome:
        chromosome for mutation
    r_mut:
        probability of mutation per gene
    '''

    # iterate over orders
    for _ in range(len(chromosome)):
        # check for mutation
        chromosome_mut = [i for i in chromosome if np.random.random_sample() > r_mut]
        chromosome = np.stack(chromosome_mut)

        return chromosome

def mutation_add(chromosome, r_mut, schedulers_list_smd, schedulers_list_aoi, max_switch):
    '''adds a random number of random orders to the chromosome
    
    Parameters:
    -----------
    chromosome:
        chromosome for mutation
    r_mut:
        probability of mutation per gene
    '''

    # initialize lists with switches and indices
    switch = []
    index = []

    for i in range(len(chromosome)):
        # determine indices where new switches will be inserted
        if np.random.random_sample() < r_mut:
            schedulers_smd = np.random.choice(schedulers_list_smd)
            schedulers_aoi = np.random.choice(schedulers_list_aoi)
            periods = np.random.randint(100, max_switch)
            new_switch = np.array((schedulers_smd, schedulers_aoi, periods))

            switch.append(new_switch)
            index.append(i)

    # insert new switches
    if switch:
        switch = np.stack(switch).reshape(len(switch),3)
        chromosome = np.insert(chromosome, index, switch, axis=0)

    return chromosome

def plot_evolution(evolution, n_iter):
    '''creates plot for genetic algorithm evolution
    
    Parameters:
    -----------
    evolution:
        list with best fitness value of each generation
    n_iter:
        number of generations
    '''

    plt.plot(np.arange(1, n_iter), evolution)
    plt.title('Genetic Algorithm Evolution')
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.xticks(np.arange(1, n_iter, step=5))
    plt.savefig('ga_history', bbox_inches='tight', dpi=400)
    plt.close()

def genetic_algorithm(problem, schedulers_list_smd, schedulers_list_aoi, n_iter, n_pop, r_cross, r_mut, num_switch, max_switch):
    '''evolution process of the genetic algorithm
    
    Parameters:
    -----------
    problem:
        string that contains the name of the problem instance
    schedulers_list_smd:
        list of implemented schedulers for smd stage
    schedulers_list_aoi:
        list of implemented schedulers for aoi stage
    n_iter:
        number of generations
    n_pop:
        population size
    r_cross:
        crossover probability
    r_mut:
        mutation probability
    num_switch:
        number of dispatching rule switches
    max_switch:
        maximum switching point
    '''

    # get new random seed
    np.random.seed()

    # initial population
    pop = initialization(
        schedulers_list_smd=schedulers_list_smd,
        schedulers_list_aoi=schedulers_list_aoi,
        size=n_pop,
        max_switch=max_switch,
        num_switch=num_switch
    )

    evolution = []
    best_eval = {}

    # initial solution
    best = pop[0].copy()
    best_eval['score'] = 100000

    for gen in range(1, n_iter): 
        # start time measure
        start = timeit.default_timer()

        print("\n### Generation {} #############################################".format(gen))

        scores = []
        # evaluate all candidates in the population
        for individual in pop:

            # simulation run
            hfs_env = HFS_Env(
                problem=problem,
                schedulers_list_smd=individual[:,0].tolist(),
                schedulers_list_aoi=individual[:,1].tolist(),
                switches=individual[:,2].tolist()
            )

            fitness = hfs_env.makespan + hfs_env.total_tardiness
            scores.append(fitness)

            #check for new best solution
            if fitness < best_eval['score']:
                eval = {
                    'score': hfs_env.makespan + hfs_env.total_tardiness,
                    'makespan': hfs_env.makespan,
                    'total_tardiness': hfs_env.total_tardiness,
                    'major_setups': hfs_env.num_major_setups 
                }
                best, best_eval = individual, eval
                #print('New best: ', individual)
                print('Found new best: ', eval)

        evolution.append(best_eval['score'])
        print('Average Fitness: ', np.mean(scores))

        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
    
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                c = mutation_drop(
                    chromosome=c,
                    r_mut=r_mut
                )

                c = mutation_add(
                    chromosome=c,
                    r_mut=r_mut,
                    schedulers_list_smd=schedulers_list_smd,
                    schedulers_list_aoi=schedulers_list_aoi,
                    max_switch=max_switch
                )

                # store for next generation
                children.append(c)
            # replace population
            pop = children

        print('Computation Time: ', timeit.default_timer() - start, " seconds")
    
    plot_evolution(evolution, n_iter)
    f = open('/home/benjamin/Documents/Projects/Seneca/Simulated Annealing/results/optimised_job_list_genetic.dat', "w")
    for i, smd in enumerate(hfs_env.smds):
        for entry in smd.production_plan:
            f.write(str(i)+'\t'+str(entry[0])+'\n')
    f.close()
    return best, best_eval
