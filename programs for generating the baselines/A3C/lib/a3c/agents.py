"""
Note: There seem to be several bugs in PyTorch that are responsible for
some false error messages in the PROBLEMS console, although the code will
run as expected. Some error messages indicate that methods, such as from_numpy,
are not existing (pylint(no-member)). The problem is reported here:
https://github.com/pytorch/pytorch/issues/701. The following comment lines are
therefore sometimes integrated to disable and enable the tracking of no-member errors:

# pylint: disable=E1101
# pylint: enable=E1101
"""

import torch
import torch.multiprocessing as mp
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import deque
import pickle
import timeit


class A3CAgent_SMD(mp.Process):

    """
    Creates an agent which performs the A3C training routine 
    for SMD scheduling. The agent is designed to allow a parallelized
    training on several CPU cores. The agent reports the gradients of 
    its local agent to a global agent. 
    
    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    name: Unique identifier for the agent.

    hfs_env_class: Class of the hybrid flow shop 
                   simulation environment.
    
    problem: Problem instance to train training.
    
    scaling: Scaling method for input features. 
             The following string values are valid:
                - 'minmax (0,1)'
                - 'minmax (-1,1)'
                - 'zscore'
    
    schedulers: Dictionary of classes for the scheduling of jobs. 
                The dictionary can contain the following keys:
                    {'smd': ,
                     'aoi': }

    smd_actor_critic: Global actor-critic model based on which 
                      local actor-critic model will be derrived.
    
    optimizer: Optimizer to update parameters of the actor-critic 
               model.
   
    max_episodes: Total number of episodes executed by all A3C agents.
                  Determines when the A3C algorithm finishes.
    
    global_episode: Loop variable counting all performed episodes
                    over all A3C agents.

    global_episode_reward: Global episode reward of actor-critic.
    
    global_episode_reward_queue: Torch.mp.Queue for storing the global 
                                 episode reward over all A3C agents.
    
    global_makespan_queue: Torch.mp.Queue to track the episode 
                           makespan of all A3C agents.
    
    global_total_tardiness_queue: Torch.mp.Queue to track episode 
                                  total tardiness of all A3C agents.
    
    global_agent_queue: Torch.mp.Queue to track all A3C agents that
                        are still running the training.
    
    update_rate: Defines after how many set of actions the
                 actor-critic should be updated.
               
    gamma: Discount factor for values. Float between 0 and 1.
    
    save_dir: Directory for saving checkpoints of current 
              actor-critic parameters.
    ------------------------------------------------------------------
    """

    def __init__(
        self,
        name,
        hfs_env_class,
        problem,
        scaling,
        schedulers,
        smd_actor_critic,
        optimizer,
        max_episodes,
        global_episode,
        global_episode_reward,
        global_episode_reward_queue,
        global_makespan_queue,
        global_total_tardiness_queue,
        global_agent_queue,
        update_rate,
        gamma,
        save_dir,
    ):

        super(A3CAgent_SMD, self).__init__()

        # name
        self.name = name

        # environment class
        self.hfs_env_class = hfs_env_class

        # training problems
        self.problem = problem
        self.scaling = scaling

        # scheduling classes
        self.schedulers = schedulers

        # global and local actor-critic models
        self.global_actor_critic = smd_actor_critic
        if "LSTM" in smd_actor_critic.__class__.__name__:
            self.local_actor_critic = smd_actor_critic.__class__(
                (smd_actor_critic.num_job_inputs, smd_actor_critic.num_system_inputs),
                smd_actor_critic.num_hidden_lstm,
                smd_actor_critic.num_hidden,
                smd_actor_critic.beta,
            )
        else:
            self.local_actor_critic = smd_actor_critic.__class__(
                smd_actor_critic.num_inputs,
                smd_actor_critic.num_hidden,
                smd_actor_critic.beta,
            )

        # optimizers
        self.optimizer = optimizer

        # global variables
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.global_episode_reward_queue = global_episode_reward_queue
        self.global_makespan_queue = global_makespan_queue
        self.global_total_tardiness_queue = global_total_tardiness_queue
        self.global_agent_queue = global_agent_queue

        # run control
        self.max_episodes = max_episodes
        self.update_rate = update_rate

        # discount factor for Q-values
        self.gamma = gamma

        # directory for saving agent parameters
        self.save_dir = save_dir

    def run(self):

        # print empty line, if this is the first reported episode
        if self.name == "AGENT 001":
            print()

        # print info
        print("--- {} started ---".format(self.name))

        # while termination criteria not reached
        while self.global_episode.value < self.max_episodes:
            self.create_env()
            self.step_through_env()

        # Add None to reward queue (acts as a stop symbol, when looting reward queues in a3c_training.py)
        self.global_episode_reward_queue.put(None)

        # Add None to makespan queue (acts as a stop symbol, when looting makespan queue in a3c_training.py)
        self.global_makespan_queue.put(None)

        # Add None to total tardiness queue (acts as a stop size, when looting total tardiness queue in a3c_training.py)
        self.global_total_tardiness_queue.put(None)

        print("--- {} finished ---".format(self.name))

        self.global_agent_queue.get()

        # plot and save train metrics, if this is the last agent that terminates
        if self.global_agent_queue.qsize() == 0:
            self.plot_and_save_train_metrics()

    def create_env(self):

        # create new environment (resets the environment)
        self.hfs_env = self.hfs_env_class(
            problem=self.problem,
            scaling=self.scaling,
            smd_scheduling_class=self.schedulers["smd"],
            aoi_scheduling_class=self.schedulers["aoi"],
            smd_scheduling_model=self.local_actor_critic,
            step_execution=True,
        )

        # initialize step counter
        self.step = 0

        # initialize state buffer
        self.states = []

        # initialize action buffer
        self.actions = []

        # initialize reward buffer
        self.rewards = []

        # initialize episode reward
        self.episode_reward = 0

    def step_through_env(self):

        # start time measure
        self.start = timeit.default_timer()

        # while simulation has not reached final state
        while True:

            # step to next state requiring a scheduling decision
            self.hfs_env.step()

            if self.hfs_env.state == "smd scheduling":

                # deploy smd_scheduler together with smd_scheduling_agent
                transition = self.hfs_env.smd_scheduler.select_job(
                    self.hfs_env.observed_machine
                )

                # add list of states, actions, rewards to corresponding buffer
                self.states = (
                    np.concatenate((self.states, transition["states"]), axis=1)
                    if len(self.states)
                    else transition["states"]
                )
                self.actions = (
                    np.concatenate((self.actions, transition["actions"]))
                    if len(self.actions)
                    else transition["actions"]
                )
                self.rewards = (
                    np.concatenate((self.rewards, transition["rewards"]))
                    if len(self.rewards)
                    else transition["rewards"]
                )

                # update episode reward
                self.episode_reward += sum(transition["rewards"].squeeze())

                # increase step counter
                self.step += 1

                if self.step == self.update_rate:

                    # determine future state
                    future_state = self.states[:, -1, :]

                    # update parameters
                    self.update_agent(future_state)

                    # save current agent as checkpoint
                    torch.save(
                        self.global_actor_critic.state_dict(),
                        os.path.join(self.save_dir, "smd_agent_checkpoint"),
                    )

                    # keep only last (future) state action and reward within buffers
                    self.step = 0
                    self.states = self.states[:, -1:, :]
                    self.actions = self.actions[-1:]
                    self.rewards = self.rewards[-1:]

            elif self.hfs_env.state == "end of simulation":

                # update parameters
                self.update_agent()

                # save current agent as checkpoint
                torch.save(
                    self.global_actor_critic.state_dict(),
                    os.path.join(self.save_dir, "smd_agent_checkpoint"),
                )

                # leave while-loop if simulation environment has terminated
                break

        # update global variables and print interim report in console
        self.update_global_variables()

        # if agent has a LSTM input layer, reset the hidden state of the agent
        if "LSTM" in self.local_actor_critic.__class__.__name__:

            # pylint: disable=E1101

            self.local_actor_critic.hidden_state_list = [
                torch.zeros(2, 1, self.local_actor_critic.num_hidden_lstm),
                torch.zeros(2, 1, self.local_actor_critic.num_hidden_lstm),
            ]

            # pylint: enable=E1101

        # delete model object
        del self.hfs_env

    def update_agent(self, future_state=None):

        if future_state is None:

            # create zero-vector for initial values
            value = np.zeros(self.states.shape[0])

        else:

            # pylint: disable=E1101

            # cast state to tensor
            future_state = torch.from_numpy(future_state).float()

            # pylint: enable=E1101

            # initialize values with output of value network
            if "LSTM" in self.local_actor_critic.__class__.__name__:
                value = (
                    self.local_actor_critic.feed_forward(
                        future_state, future_state=True
                    )[-1]
                    .data.numpy()
                    .squeeze()
                )
            else:
                value = (
                    self.local_actor_critic.feed_forward(future_state)[-1]
                    .data.numpy()
                    .squeeze()
                )

        # create labels for value network output
        target_values = []
        for reward in reversed(
            self.rewards[:] if future_state is None else self.rewards[:-1]
        ):
            value = reward + self.gamma * value
            target_values.append(value)
        target_values.reverse()

        # calculate loss
        loss = self.local_actor_critic.calc_loss(
            states=self.states[:, :, :]
            if future_state is None
            else self.states[:, :-1, :],
            actions=self.actions[:, :]
            if future_state is None
            else self.actions[:-1, :],
            target_values=np.array(target_values),
        )

        # reset gradients from last update
        self.optimizer.zero_grad()

        # backpropagate loss to compute gradients
        if (
            "LSTM" in self.local_actor_critic.__class__.__name__
            and self.global_episode.value < self.max_episodes
            and self.update_rate < len(self.hfs_env.global_smd_queue)
        ):
            # Set retain_graph to True if agent has a LSTM input layer and if this was not the very last training step
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        # update gradients of global network with gradients of local network
        for global_param, local_param in zip(
            self.global_actor_critic.parameters(), self.local_actor_critic.parameters()
        ):
            global_param._grad = local_param.grad

        # perform update
        self.optimizer.step()

        # update weights of local network with weights of global network
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

    def update_global_variables(self):

        # increment global number of performed episodes
        with self.global_episode.get_lock():
            self.global_episode.value += 1

        # update individual global episode rewards
        with self.global_episode_reward.get_lock():
            if self.global_episode_reward.value == 0:
                self.global_episode_reward.value = self.episode_reward
            else:
                self.global_episode_reward.value = (
                    self.global_episode_reward.value * 0.99 + self.episode_reward * 0.01
                )
            # add current global episode reward of agent to corresponding global queue
            self.global_episode_reward_queue.put(self.global_episode_reward.value)

        # add current makespan and total tardiness to corresponding global queues
        self.global_makespan_queue.put(self.hfs_env.makespan)
        self.global_total_tardiness_queue.put(self.hfs_env.total_tardiness)

        # print empty line, if this is the first reported episode
        if self.global_episode.value == 1:
            print()

        # report statistics of finished episode
        print(
            "{}: \t Episode: {:7.0f} | Makespan: {:7.0f} | Total Tardiness: {:7.0f} | Reward: {} | Computational Time: {} seconds".format(
                self.name,
                self.global_episode.value,
                self.hfs_env.makespan,
                self.hfs_env.total_tardiness,
                int(self.global_episode_reward.value),
                timeit.default_timer() - self.start,
            )
        )

    def plot_and_save_train_metrics(self):

        # initialize plot lists for visualizing later the final results
        episode_reward_list = []
        best_episode_reward_list = []
        last_episode_reward_list = deque(maxlen=100)
        mov_avg_episode_reward_list = []
        makespan_list = []
        best_makespan_list = []
        last_makespan_list = deque(maxlen=100)
        mov_avg_makespan_list = []
        total_tardiness_list = []
        best_total_tardiness_list = []
        last_total_tardiness_list = deque(maxlen=100)
        mov_avg_total_tardiness_list = []

        # collect makespan statistics
        best_makespan = 1e7
        while True:
            makespan = self.global_makespan_queue.get()
            if makespan is not None:
                makespan_list.append(makespan)
                last_makespan_list.append(makespan)
                mov_avg_makespan_list.append(np.mean(last_makespan_list))
                if makespan < best_makespan:
                    best_makespan = makespan
                best_makespan_list.append(best_makespan)
            else:
                break

        # collect total tardiness statistics
        best_total_tardiness = 1e7
        while True:
            total_tardiness = self.global_total_tardiness_queue.get()
            if total_tardiness is not None:
                total_tardiness_list.append(total_tardiness)
                last_total_tardiness_list.append(total_tardiness)
                mov_avg_total_tardiness_list.append(np.mean(last_total_tardiness_list))
                if total_tardiness < best_total_tardiness:
                    best_total_tardiness = total_tardiness
                best_total_tardiness_list.append(best_total_tardiness)
            else:
                break

        # collect reward statistics
        best_episode_reward = -1e7
        while True:
            episode_reward = self.global_episode_reward_queue.get()
            if episode_reward is not None:
                episode_reward_list.append(episode_reward)
                last_episode_reward_list.append(episode_reward)
                mov_avg_episode_reward_list.append(np.mean(last_episode_reward_list))
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                best_episode_reward_list.append(best_episode_reward)
            else:
                break

        # create plots
        fig = plt.figure(figsize=(30, 10))
        spec = gridspec.GridSpec(1, 3)
        ax0 = plt.subplot(spec[0])
        ax0.plot(
            [i for i in range(len(total_tardiness_list))],
            total_tardiness_list,
            color="salmon",
        )
        ax0.plot(
            [i for i in range(len(mov_avg_total_tardiness_list))],
            mov_avg_total_tardiness_list,
            color="darkred",
        )
        ax0.plot(
            [i for i in range(len(best_total_tardiness_list))],
            best_total_tardiness_list,
            color="gold",
        )
        ax0.set_xlim(left=0, right=len(total_tardiness_list))
        ax0.set_ylabel(
            "total tardiness", bbox=dict(facecolor="None", edgecolor="None", pad=7)
        )
        ax0.set_facecolor("whitesmoke")
        ax0.grid(which="both")
        ax0.legend(("all", "moving average", "best"), loc="upper right")
        ax1 = plt.subplot(spec[1])
        ax1.plot(
            [i for i in range(len(makespan_list))], makespan_list, color="springgreen"
        )
        ax1.plot(
            [i for i in range(len(mov_avg_makespan_list))],
            mov_avg_makespan_list,
            color="green",
        )
        ax1.plot(
            [i for i in range(len(best_makespan_list))],
            best_makespan_list,
            color="gold",
        )
        ax1.set_xlim(left=0, right=len(makespan_list))
        ax1.set_ylabel("makespan", bbox=dict(facecolor="None", edgecolor="None", pad=7))
        ax1.set_facecolor("whitesmoke")
        ax1.grid(which="both")
        ax1.legend(("all", "moving average", "best"), loc="upper right")
        ax2 = plt.subplot(spec[2])
        ax2.plot(
            [i for i in range(len(episode_reward_list))],
            episode_reward_list,
            color="cornflowerblue",
        )
        ax2.plot(
            [i for i in range(len(mov_avg_episode_reward_list))],
            mov_avg_episode_reward_list,
            color="darkblue",
        )
        ax2.plot(
            [i for i in range(len(best_episode_reward_list))],
            best_episode_reward_list,
            color="gold",
        )
        ax2.set_xlim(left=0, right=len(episode_reward_list))
        ax2.set_ylabel(
            "episode reward\n(smd scheduling)",
            bbox=dict(facecolor="None", edgecolor="None", pad=7),
        )
        ax2.set_facecolor("whitesmoke")
        ax2.grid(which="both")
        ax2.legend(("all", "moving average", "best"), loc="upper left")
        fig.tight_layout()
        fig.savefig(
            fname="plots/trainmetrics_a3c_m{}_t{}.png".format(
                int(best_makespan), int(best_total_tardiness)
            )
        )

        # store plot data
        train_metrics = {
            "makespan_list": makespan_list,
            "best_makespan_list": best_makespan_list,
            "last_makespan_list": last_makespan_list,
            "mov_avg_makespan_list": mov_avg_makespan_list,
            "total_tardiness_list": total_tardiness_list,
            "best_total_tardiness_list": best_total_tardiness_list,
            "last_total_tardiness_list": last_total_tardiness_list,
            "mov_avg_total_tardiness_list": mov_avg_total_tardiness_list,
            "episode_reward_list": episode_reward_list,
            "best_episode_reward_list": best_episode_reward_list,
            "last_episode_reward_list": last_episode_reward_list,
            "mov_avg_episode_reward_list": mov_avg_episode_reward_list,
        }

        # save x-y-data of plots to file
        with open(
            os.path.join(
                os.getcwd(),
                "plots/trainmetrics_a3c_m{}_t{}".format(
                    int(best_makespan), int(best_total_tardiness)
                ),
            ),
            "wb",
        ) as f:
            pickle.dump(train_metrics, f)

        # save global agent
        torch.save(
            self.global_actor_critic.state_dict(),
            os.path.join(self.save_dir, "smd_agent_final"),
        )
