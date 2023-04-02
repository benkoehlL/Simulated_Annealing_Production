"""
Note: There seem to be several bugs in PyTorch that are responsible for 
some false error messages in the PROBLEMS console, although the code will 
run as expected. Some error messages indicate that methods, such as 'from_numpy',
are not existing (pylint(no-member)). The problem is reported here:
https://github.com/pytorch/pytorch/issues/701. The following comment lines are 
therefore sometimes integrated to disable and enable the tracking of no-member 
errors:

# pylint: disable=E1101
# pylint: enable=E1101
"""

import torch
import numpy as np
from collections import deque
import random

class DDPG_TD3():
    
    """
    ------------------------------------------------------------------
    Creates an agent which trains a PyTorch model with the Deep
    Deterministic Policy Gradient (DDPG) or Twin Delayed DDPG (TD3)
    algorithm. Depending on the number of critic models passed, the
    agent performs the DDPG (1x critc model) or TD3 algorithm
    (2x critic models).
    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    actor_model: PyTorch model that maps job and system states to
                 scheduling actions.
    
    critic_model: PyTorch model that evaluates (state, action) tuples.
                  The critic must be either a single PyTorch model 
                  (DDPG) or a 2-tuple of PyTorch models (TD3).
    
    actor_optimizer: PyTorch optimizer object for updating the model
                     parameters of the actor.

    critic_optimizer: PyTorch optimizer object for updating the model
                      parameters of the critic.
    
    gamma: Discount factor for Bellman equation.

    replay_memory_size: Capacity of replay memory.

    replay_memory_min: Minimum number of samples in replay memory to
                       perform a training step.

    minibatch_size: Number of samples from replay memory  that are
                    processed within a single training step
    ------------------------------------------------------------------
    """

    def __init__(
        self,
        actor_model,
        critic_model,
        actor_optimizer,
        critic_optimizer,
        gamma=0.99,
        replay_memory_size=200,
        min_replay_memory=20,
        minibatch_size=10,
    ):

        # check if critic contains a PyTorch model or a twintuple of PyTorch models
        assert (
            issubclass(type(critic_model), torch.nn.modules.module.Module)
            or (
                isinstance(critic_model, tuple) 
                and len(critic_model) == 2 
                and all(issubclass(
                    type(model), 
                    torch.nn.modules.module.Module
                ) for model in critic_model)
            )
        ), "Parameter 'critic' must be a PyTorch model or a 2-tuple of PyTorch models"

        # actor model
        self.actor = actor_model
        self.target_actor = self.actor.__class__(
            self.actor.num_inputs,
            self.actor.num_hidden,
            self.actor.is_actor
        )
        self.copy_model_params(self.actor, self.target_actor)
        
        # critic model(s)
        if not isinstance(critic_model, tuple):
            # DDPG
            self.critic = critic_model
            self.target_critic = self.critic.__class__(
                self.critic.num_inputs,
                self.critic.num_hidden,
                self.critic.is_actor
            )
            self.copy_model_params(self.critic, self.target_critic)
        else:
            # TD3
            self.critic_1, self.critic_2 = critic_model
            self.target_critic_1 = self.critic_1.__class__(
                self.critic_1.num_inputs,
                self.critic_1.num_hidden,
                self.critic_1.is_actor
            )
            self.copy_model_params(self.critic_1, self.target_critic_1)
            self.target_critic_2 = self.critic_2.__class__(
                self.critic_2.num_inputs,
                self.critic_2.num_hidden,
                self.critic_2.is_actor
            )
            self.copy_model_params(self.critic_2, self.target_critic_2)
        
        # replay memory
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.min_replay_memory = min_replay_memory

        # learning parameters
        self.gamma = gamma
        self.minibatch_size = minibatch_size
    
    def train(self, update_target_models):

        """
        Training process:
        1) Create minibatch from replay memory
        2) Feed state to critic model to determine value
        3) Feed future state to critic model to determine future value
        4) ....
        5) Update value network
        6) Update target network, if simulation has terminated
        7) Save target network, if objective has been improved
        """

        # training does not start, until a minimum number of training samples is collected
        if len(self.replay_memory) < self.min_replay_memory:
            return
        
        # create state, action, reward and future state minibatches
        batch = random.sample(
            self.replay_memory, 
            self.minibatch_size
        )

        states = np.vstack(transition[0] for transition in batch)
        actions = np.vstack(transition[1] for transition in batch)
        rewards = np.vstack(transition[2] for transition in batch)
        future_states = np.vstack(transition[3] for transition in batch)
        dones = np.array([transition[4] for transition in batch])

        # HIER WEITER 

        # Critic loss
        qvalues = self.critic.feed_forward(
            torch.cat([states, actions.transpose(0, 1).unsqueeze(-1)], dim=-1)
        )
        future_actions = self.target_actor(future_states)
        future_qvalues = self.target_critic.feed_forward(
             torch.cat([future_states, future_actions.transpose(0, 1).unsqueeze(-1)], dim=-1)
        )
        target_qvalues

    
    def copy_model_params(self, source, target):

        """
        Copies all parameters of a source model and transfers them to a 
        target model.
        """

        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

        



        
