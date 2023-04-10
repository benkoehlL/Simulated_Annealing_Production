import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import DataConversionWarning
import warnings

# --- Super Classes ------------------------------------------------------------------------------------------------

class Scheduler:

    """
    Base class for selecting jobs from a queue 
    according to a certain strategy.
    
    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    hfs_env: hybrid flow shop environment where the Scheduler shall
             be embedded.
    
    stage: The production stage for which the Scheduler is utilized.
           The following string values are valid:
            - 'aoi'
            - 'smd'
    ------------------------------------------------------------------
    """

    def __init__(self, hfs_env, stage, *args, **kwargs):

        self.hfs_env = hfs_env
        self.stage = stage
        self.machines = self.hfs_env.aois if self.stage == "aoi" else self.hfs_env.smds

    def construct_sequence(self, problem, *args, **kwargs):

        # return sequence
        return [job for job in problem.values()]

    def select_job(self, *args, **kwargs):

        pass

    def reallocate_and_reselect(self, job, machine, queue, priorities):

        # create a copy of the list of priorities, which can be manipulated
        copy_priorities = priorities
        # check if the family of the selected job is currently produced on another SMD
        while (
            job.family
            in [
                smd.setuptype if smd.status != "major setup" else smd.setup_to
                for smd in self.machines
            ]
            and job.family != machine.setuptype
        ):
            # if so: reallocate job to local queue of SMD that currently produces the family of the selected job
            smd_index = list(
                map(
                    lambda smd: smd.setuptype == job.family
                    if smd.status != "major setup"
                    else smd.setup_to == job.family,
                    self.machines,
                )
            ).index(True)
            job.enter(self.machines[smd_index].local_queue)
            # ...and reduce copy_priorities by the priority of the job that has been reallocated
            copy_priorities = np.delete(copy_priorities, copy_priorities.argmax())
            # check if there are still jobs in the global smd queue
            if copy_priorities.size > 0:
                # if so: identify index of the next job with maximum priority
                job_index = int(copy_priorities.argmax())
                # ...and take the job from the queue
                job = queue.pop(job_index)
            else:
                # set job to None otherwise
                job = None
                # ...and leave while-loop
                break

        return job


class DRL_Scheduler(Scheduler):

    """
    Base class for selecting jobs from a queue according to the
    decisions of an agent, which is trained with a deep reinforcement 
    learning algorithm.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    hfs_env: hybrid flow shop environment where the Scheduler shall
             be embedded.
    
    stage: The production stage for which the Scheduler is utilized.
           The following string values are valid:
            - 'aoi'
            - 'smd'
    ------------------------------------------------------------------
    """

    def __init__(self, hfs_env, stage, *args, **kwargs):

        super(DRL_Scheduler, self).__init__(hfs_env, stage, *args, **kwargs)
        self.model = (
            self.hfs_env.aoi_scheduling_model
            if self.stage == "aoi"
            else (self.hfs_env.smd_scheduling_model)
        )
        self.max_earliness = self.calc_max_earliness()
        self.max_tardiness = self.calc_max_tardiness()
        if self.stage == "smd":
            self.padding_tracker = []
            self.padding_machine_tracker = []
            self.current_padding_list = []
            self.current_padding_machine_list = []

    def select_job(self, machine, *args, **kwargs):

        # determine queue
        queue = machine.global_queue
        # compute priorities of all jobs in queue
        states, actions = self.compute_priorities(machine, queue)
        # compute rewards
        rewards = self.compute_rewards(machine, queue, actions)
        # identify index of job with maximum priority
        job_index = int(actions.argmax())
        # take job from queue
        job = queue.pop(job_index)
        # if smd scheduler
        if self.stage == "smd":
            # save content of padding tracker as padding list for the current time step
            self.current_padding_list = self.padding_tracker[:]
            self.current_padding_machine_list = self.padding_machine_tracker[:]
            # check if selected job must be reallocated and another job must be selected
            job = self.reallocate_and_reselect(job, machine, queue, actions[:])
        # save selected job as active job of machine
        machine.job = job
        # pad states, actions and rewards with zeros for each job that has been removed in the previous time step
        if self.current_padding_list:
            for position in sorted(self.current_padding_list):
                states = np.insert(
                    states, position, np.zeros(shape=states.shape[-1]), axis=0
                )
                actions = np.insert(actions, position, 0)
                rewards = np.insert(rewards, position, 0)
        # reshape states, actions and rewards
        states = states.reshape((states.shape[0], -1, states.shape[-1]))
        actions = actions.reshape((-1, *actions.shape))
        rewards = rewards.reshape((-1, *rewards.shape))

        # check if there are still jobs in the global smd queue
        if(queue._length==1):
            print(self.padding_tracker,'\n')
            print(self.padding_machine_tracker,'\n')
            print(set(zip(self.padding_tracker,self.padding_machine_tracker)),'\n')


        # return actions, rewards, index of selected job and selected job
        return {"states": states, "actions": actions, "rewards": rewards}

    def compute_priorities(self, machine, queue, *args, **kwargs):

        # initialize states
        states = np.stack([self.create_features(job, machine) for job in queue], axis=0)
        # compute actions
        actions = self.model.compute_actions(states, num_jobs=len(queue))

        return states, actions

    def compute_rewards(self, machine, queue, actions):

        # create reward list
        rewards = []

        # sorted queue - step 1: create indexes for each job in queue
        indexed_queue = [(index, job) for index, job in enumerate(queue)]
        # sorted queue - step 2: sort queue according to descending job priorities
        sorted_queue = [
            pair
            for (pair, action) in sorted(
                zip(indexed_queue, actions), key=lambda pair: pair[1], reverse=True
            )
        ]
        # initialize workloads
        workloads = [machine.calc_workload() for machine in self.machines]
        # if aoi scheduler
        if self.stage == "aoi":
            # determine completion time for all jobs in queue by cumulating the workloads of all machines
            for _, job in sorted_queue:
                index = workloads.index(min(workloads))
                workloads[index] += 25 + job.t_aoi
                # calculate tardiness of each job
                tardiness = job.due_date - (self.hfs_env.env.now() + workloads[index])
                # the reward is either zero or the positive tardiness
                rewards.append((-tardiness / 2) if tardiness > 0 else tardiness)
        # if smd scheduler
        else:
            # determine current setups of all machines
            setups = [
                machine.setuptype
                if machine.status != "major setup"
                else machine.setup_to
                for machine in self.machines
            ]
            # determine completion time for all jobs in queue by cumulating the workloads of all machines
            for _, job in sorted_queue:
                # consider minor setup time
                if job.family in setups:
                    index = setups.index(job.family)
                    workloads[index] += 25 + job.t_smd
                # consider major setup time
                else:
                    index = workloads.index(min(workloads))
                    workloads[index] += 65 + job.t_smd
                    setups[index] = job.family
                # calculate tardiness of each job
                tardiness = (job.due_date * (job.t_smd / (job.t_smd + job.t_aoi))) - (
                    self.hfs_env.env.now() + workloads[index]
                )
                # the reward is either -1 (if job is too late) or zero
                scaled_tardiness = (
                    (
                        ((tardiness - self.max_tardiness) / (-1 - self.max_tardiness))
                        * (-30 + 40)
                        - 40
                    )
                    if tardiness < 0
                    else ((tardiness / self.max_earliness) * (-10))
                )
                rewards.append(scaled_tardiness)
        # sort rewards according to job sequence in queue and cast list to numpy array
        rewards = np.array(
            [
                reward
                for reward, _ in sorted(
                    zip(rewards, sorted_queue), key=lambda pair: pair[1][0]
                )
            ]
        )

        # return rewards
        return rewards

    def reallocate_and_reselect(self, job, machine, queue, priorities):

        # disable error code E1101 in pylint to surpress error message: "Module 'numpy' has no 'NINF' member; maybe 'PINF'?"
        # pylint: disable=E1101

        # create a copy of the list of priorities, which can be manipulated
        copy_priorities = priorities[:]
        # if pcurrent padding list is not empty
        if self.current_padding_list:
            # pad copy of priority vector
            for position in sorted(self.current_padding_list):
                copy_priorities = np.insert(copy_priorities, position, np.NINF)
        # check if the family of the first selected job is currently produced on another SMD
        while (
            job.family
            in [
                smd.setuptype if smd.status != "major setup" else smd.setup_to
                for smd in self.machines
            ]
            and job.family != machine.setuptype
        ):
            # if so: reallocate job to local queue of SMD that currently produces the family of the selected job
            smd_index = list(
                map(
                    lambda smd: smd.setuptype == job.family
                    if smd.status != "major setup"
                    else smd.setup_to == job.family,
                    self.machines,
                )
            ).index(True)
            job.enter(self.machines[smd_index].local_queue)
            # remove priority of the job that has been reallocated from original priority list
            priorities = np.delete(priorities, priorities.argmax())
            # save index of job that has been reallocated
            self.padding_tracker.append(copy_priorities.argmax())
            
            # write number of the machine the job was assigned to into a list
            self.padding_machine_tracker.append(machine)
            
            # and set priority of the job that has been reallocated to zero
            copy_priorities[copy_priorities.argmax()] = np.NINF
            # check if there are still jobs in the global smd queue
            if priorities.size:
                # if so: identify index of the next job with maximum priority
                job_index = int(priorities.argmax())
                # take the job from the queue
                job = queue.pop(job_index)
            else:
                # set job to None otherwise
                job = None
                print(self.padding_tracker,'\n')
                print(self.padding_machine_tracker,'\n')
                print(set(zip(self.padding_tracker,self.padding_machine_tracker)),'\n')
                # and leave while-loop
                break
        # if the family of the first selected job is not currently produced on another SMD
        else:
            # update padding tracker with index of the first selected job
            self.padding_tracker.append(copy_priorities.argmax())
            
            # write number of the machine the job was assigned to into a list
            self.padding_machine_tracker.append(machine)

        # pylint: enable=E1101

        return job

    def calc_max_earliness(self):

        # identify job with maximum due date
        job = max(self.hfs_env.problem.values(), key=lambda job: job["due date"])
        # calculate estimated maximum earliness
        return job["due date"] - (job["t_smd"] + 65)

    def calc_max_tardiness(self):

        # sort jobs according to descending due dates
        sorted_jobs = sorted(
            self.hfs_env.problem.items(),
            key=lambda job: job[1]["due date"],
            reverse=True,
        )
        # initialize workloads for all SMDs
        workloads = [0 for i in range(4)]
        # initialize setups for all SMDs
        setups = [0 for i in range(4)]

        for i, (_, job) in enumerate(sorted_jobs):

            if job["family"] in setups:
                index = setups.index(job["family"])
                workloads[index] += 25 + job["t_smd"]
            else:
                index = workloads.index(min(workloads))
                workloads[index] += 65 + job["t_smd"]
                setups[index] = job["family"]

            if i == len(sorted_jobs) - 1:
                return job["due date"] - workloads[index]

    def create_features(self, job, machine, *args, **kwargs):

        return np.array([])  # placeholder

# ------------------------------------------------------------------------------------------------------------------

class FIFO_Scheduler(Scheduler):

    """
    Selects jobs from a queue of a machine according to the
    first-in-first-out dispatching strategy.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    hfs_env: hybrid flow shop environment where the Scheduler shall
             be embedded.
    
    stage: The production stage for which the Scheduler is utilized.
           The following string values are valid:
            - 'aoi'
            - 'smd'
    ------------------------------------------------------------------
    """

    def __init__(self, hfs_env, stage, *args, **kwargs):

        super(FIFO_Scheduler, self).__init__(hfs_env, stage, *args, **kwargs)

    def select_job(self, machine):

        # determine queue
        queue = machine.global_queue
        # identify index of job
        job_index = 0
        # take job from queue
        job = queue.pop(job_index)
        # if smd scheduler
        if self.stage == "smd":
            # check if selected job must be reallocated and another job must be selected
            job = self.reallocate_and_reselect(
                job, machine, queue, priorities=np.array([i for i in range(len(queue))])
            )
        # save selected job as active job of machine
        machine.job = job


class LIFO_Scheduler(Scheduler):

    """
    Selects jobs from a queue of a machine according to the
    last-in-first-out dispatching strategy.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    hfs_env: hybrid flow shop environment where the Scheduler shall
             be embedded.
    
    stage: The production stage for which the Scheduler is utilized.
           The following string values are valid:
            - 'aoi'
            - 'smd'
    ------------------------------------------------------------------
    """

    def __init__(self, hfs_env, stage, *args, **kwargs):

        super(LIFO_Scheduler, self).__init__(hfs_env, stage, *args, **kwargs)

    def select_job(self, machine):

        # determine queue
        queue = machine.global_queue
        # identify index of job
        job_index = len(queue) - 1
        # take job from queue
        job = queue.pop(job_index)
        # if smd scheduler
        if self.stage == "smd":
            # check if selected job must be reallocated and another job must be selected
            job = self.reallocate_and_reselect(
                job,
                machine,
                queue,
                priorities=np.array([i for i in range(len(queue) + 1)]),
            )
        # save selected job as active job of machine
        machine.job = job


class EDD_Scheduler(Scheduler):

    """
    Selects jobs from a queue of a machine according to the
    earliest-due-date dispatching strategy.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    hfs_env: hybrid flow shop environment where the Scheduler shall
             be embedded.
    
    stage: The production stage for which the Scheduler is utilized.
           The following string values are valid:
            - 'aoi'
            - 'smd'
    ------------------------------------------------------------------
    """

    def __init__(self, hfs_env, stage, *args, **kwargs):

        super(EDD_Scheduler, self).__init__(hfs_env, stage, *args, **kwargs)

    def select_job(self, machine):

        # determine queue
        queue = machine.global_queue
        # identify index of job with lowest due date
        job_index = queue.index(min(queue, key=lambda job: job.due_date))
        # take job from queue
        job = queue.pop(job_index)
        # if smd scheduler
        if self.stage == "smd":
            # check if selected job must be reallocated and another job must be selected
            job = self.reallocate_and_reselect(
                job,
                machine,
                queue,
                priorities=np.array([(-1) * job.due_date for job in queue]),
            )
        # save selected job as active job of machine
        machine.job = job


class SPT_Scheduler(Scheduler):

    """
    Selects jobs from a queue of a machine according to the
    shortest-processing-time dispatching strategy.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    hfs_env: hybrid flow shop environment where the Scheduler shall
             be embedded.
    
    stage: The production stage for which the Scheduler is utilized.
           The following string values are valid:
            - 'aoi'
            - 'smd'
    ------------------------------------------------------------------
    """

    def __init__(self, hfs_env, stage, *args, **kwargs):

        super(SPT_Scheduler, self).__init__(hfs_env, stage, *args, **kwargs)

    def select_job(self, machine):

        # determine queue
        queue = machine.global_queue
        # identify index of job with lowest SMD or AOI processing time
        if self.stage == "smd":
            job_index = queue.index(min(queue, key=lambda job: job.t_smd))
        else:
            job_index = queue.index(min(queue, key=lambda job: job.t_aoi))
        # take job from queue
        job = queue.pop(job_index)
        # if smd scheduler
        if self.stage == "smd":
            # check if selected job must be reallocated and another job must be selected
            job = self.reallocate_and_reselect(
                job,
                machine,
                queue,
                priorities=np.array([(-1) * job.t_smd for job in queue]),
            )
        # save selected job as active job of machine
        machine.job = job


class EDDxSPT_Scheduler(Scheduler):

    """
    Selects jobs from a queue of a machine according
    to a weighted combination of the earliest-due-date
    and the shortest-processing-time dispatching strategy.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    hfs_env: hybrid flow shop environment where the Scheduler shall
             be embedded.
    
    stage: The production stage for which the Scheduler is utilized.
           The following string values are valid:
            - 'aoi'
            - 'smd'

    w_edd: weights the influence of the job's due date.

    w_spt: weights the influence of the job's smd or aoi process time.
    ------------------------------------------------------------------
    """

    def __init__(self, hfs_env, stage, w_edd=0.5, w_spt=0.5, *args, **kwargs):

        super(EDDxSPT_Scheduler, self).__init__(hfs_env, stage, *args, **kwargs)
        self.w_edd = w_edd
        self.w_spt = w_spt

    def select_job(self, machine):

        # determine queue
        queue = machine.global_queue
        # rank jobs according to ascending due dates
        due_date_ranks = stats.rankdata([job.due_date for job in queue])
        # rank jobs according to ascending SMD or AOI processing times
        if self.stage == "smd":
            process_time_ranks = stats.rankdata([job.t_smd for job in queue])
        else:
            process_time_ranks = stats.rankdata([job.t_aoi for job in queue])
        # create weighted ranking vector
        weighted_ranks = self.w_edd * due_date_ranks + self.w_spt * process_time_ranks
        # identify index of job with lowest weighted rank
        job_index = list(weighted_ranks).index(min(weighted_ranks))
        # take job from queue
        job = queue.pop(job_index)
        # if smd scheduler
        if self.stage == "smd":
            # check if selected job must be reallocated and another job must be selected
            job = self.reallocate_and_reselect(
                job, machine, queue, priorities=(-1) * weighted_ranks
            )
        # save selected job as active job of machine
        machine.job = job


class SMD_12Inputs_DRL_Scheduler(DRL_Scheduler):

    """
    Agent-based selection of jobs from a global smd queue in 
    Deep Reinforcement Learning scenarios.
    The model expects the following 12 inputs as state:
     - due date of job
     - family of job
     - smd processing time of job
     - aoi processing time of job
     - setup type of smd 1, ..., 4 (4 neurons)
     - workload of smd 1, ..., 4 (4 neurons)

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    hfs_env: hybrid flow shop environment where the Scheduler shall
             be embedded.
    ------------------------------------------------------------------
    """

    def __init__(self, hfs_env, *args, **kwargs):

        self.stage = "smd"
        super(SMD_12Inputs_DRL_Scheduler, self).__init__(hfs_env, *args, **kwargs)

    def create_features(self, job, machine, *args, **kwargs):

        # ignore warnings from scaler ("Data with input dtype int32 was converted to float64")
        warnings.filterwarnings(action="ignore", category=DataConversionWarning)
        # extract job features
        job_features = [
            job.scaled_due_date,
            job.scaled_family,
            job.scaled_t_smd,
            job.scaled_t_aoi,
        ]
        # check whether job family corresponds to setup of machine
        same_setups = [
            1 if job.family == machine.setuptype else 0 for machine in self.machines
        ]
        # extract machine workloads
        workload_features = np.array(
            [machine.calc_workload() for machine in self.machines]
        )
        # scale machine workloads according to the preference of the user (default setting = 'minmax (0,1)')
        if self.hfs_env.scaling == "minmax (-1,1)":
            workload_features = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
                workload_features.reshape(-1, 1)
            )
        elif self.hfs_env.scaling == "zscore":
            workload_features = StandardScaler().fit_transform(
                workload_features.reshape(-1, 1)
            )
        else:
            workload_features = MinMaxScaler().fit_transform(
                workload_features.reshape(-1, 1)
            )
        # cast machine features to list
        workload_features = list(np.squeeze(workload_features))
        # concat job and machine features to numpy array
        features = np.array(job_features + same_setups + workload_features).astype(
            np.float32
        )
        # return features
        return features

#############################################################################################################################################################################################################
