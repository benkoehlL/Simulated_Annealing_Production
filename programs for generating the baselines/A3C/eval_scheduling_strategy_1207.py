
#  Benjamin Köhler, Hagen Borstell, 2021 (Thorsis Technologies)
# Code adapted from Sebastian Lang (OvGU)

from lib.a3c import models
from lib.env.hfs_env import HFS_Env
from lib.env.importer import input_data, SCALING_MINMAX1
from lib.env import schedulers

import torch
import os
import timeit

if __name__ == "__main__":
    # start time measure
    start = timeit.default_timer()

    ## set problem 
    # day one
    problem_1 = input_data(
        filename = os.path.join(os.getcwd(), "data/input_noAOI.xlsx"),
        sheet_name = "dataset 2",
        scaling=SCALING_MINMAX1,
        num_jobs=None
    )

    problems = {
        "problem 1": problem_1, 
    }

    # set model
    my_actor_critic_model = models.TwoLayerContinuousMLP(num_inputs=12, num_hidden=200)
    my_actor_critic_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "trained_models", "2021-02-09_17.15.26_a3c/smd_agent_checkpoint")))
    # start evalution
    for problem in problems:
        hfs_env = HFS_Env(
            problem=problems[problem]['problem'],
            scaling=problems[problem]['scaler'],
            smd_scheduling_class=schedulers.SMD_12Inputs_DRL_Scheduler,
            aoi_scheduling_class=schedulers.EDD_Scheduler,
            smd_scheduling_model=my_actor_critic_model,
            animation=False,
            freeze_window_after_initsim=False,
            freeze_window_at_endsim=False,
            tracing=False,
        )

        print(
            "\n### Performance on {} #############################################".format(
                problem
            )
        )
        print("Makespan: ", hfs_env.makespan, " minutes")
        print("Total Tardiness: ", hfs_env.total_tardiness, " minutes")
        print("Major Setups: ", hfs_env.num_major_setups)
        print("Computational Time: ", timeit.default_timer() - start, " seconds")
        print("Forward Propagations: {}\n ".format(hfs_env.num_forward))
        
        f = open('/home/benjamin/Documents/Projects/Seneca/Simulated Annealing/results/optimised_job_list_A3C.dat', "w")
        for i, smd in enumerate(hfs_env.smds):
            for entry in smd.production_plan:
                f.write(str(i)+'\t'+str(entry[0])+'\n')
        f.close()
        
        for aoi in hfs_env.aois:
            print("\nProduction Plan of", aoi.name())
            print("JOB ID \t START TIME \t FINISH TIME")
            for entry in aoi.production_plan:
                print("{:6.0f} \t {:10.0f} \t {:11.0f}".format(entry[0], entry[1], entry[2]))
        
