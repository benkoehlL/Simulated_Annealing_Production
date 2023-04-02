"""
The implementation of the models is inspired by: 
https://github.com/MorvanZhou/pytorch-A3C/

The following article provides a detailed explaination of the 
different design of A3C models for discrete and continuous action spaces: 
https://arxiv.org/pdf/1602.01783.pdf

Note: There seem to be several bugs in PyTorch that are responsible for 
some false error messages in the PROBLEMS console, although the code will 
run as expected. Some error messages indicate that methods, such as 'from_numpy' 
or 'relu', are not existing (pylint(no-member)). The problem is reported here:
https://github.com/pytorch/pytorch/issues/701. Another errormessage indicates 
that torch.tensor is not callable (pylint(not-callable)). The problem is 
reported here: https://github.com/pytorch/pytorch/issues/24807. The folling 
comment line are therefore sometimes integrated to disable and enable the
tracking of no-member and not-callable errors:

# pylint: disable=E1101
# pylint: enable=E1101
(for no-member errors)

# pylint: disable=E1102
# pylint: enable=E1102
(for not-callable errors)
"""

import torch
import numpy as np

### Torch models For Continuous Action Spaces ###########################################################################################################################################################

# --- Super Classes ----------------------------------------------------------------------------------------------

class ContinuousModel(torch.nn.Module):

    """
    Base class for all Torch models with continuous action spaces.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    num_inputs: state space.

    num_hidden: number of neurons in each hidden layer.

    beta: magnitude of entropy regularization.
    ------------------------------------------------------------------
    """

    def __init__(self, num_inputs, num_hidden, beta=0.005):

        """
        Construction of the actor and the critic model.
        """

        super(ContinuousModel, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.beta = beta

    def feed_forward(self, states):

        """
        Computes the mean and standard deviations of a normal 
        distribution (actor) and the values (critic) by forward 
        propagating a numpy array. The shape of the numpy array 
        must be (batchsize, num_inputs).
        """

        # pylint: disable=E1102

        mu = torch.tensor(0)  # placeholder
        sigma = torch.tensor(0)  # placeholder
        values = torch.tensor(0)  # placeholder

        # pylint: enable=E1102

        return mu, sigma, values

    def compute_actions(self, states, num_jobs):

        """
        Creates a normal distribution based on a given mean and standard
        deviation (actor output) to sample a list of actions.
        """

        self.eval()

        # pylint: disable=E1101

        # cast states to tensor
        states = torch.from_numpy(states)

        # pylint: enable=E1101

        # feed forward
        mu, sigma, _ = self.feed_forward(states)
        # create normal distribution
        policy = torch.distributions.Normal(mu.view(num_jobs,), sigma.view(num_jobs,))
        # sample actions
        actions = policy.sample().numpy()

        return actions

    def calc_loss(self, states, actions, target_values):

        """
        Calculates the loss of the actor and critic model.
        Returns the total loss by taking the average of the 
        summed loss of the actor and critic model.
        """

        # pylint: disable=E1102

        loss = torch.tensor(0)  # placeholder

        # pylint: enable=E1102

        return loss

# ----------------------------------------------------------------------------------------------------------------

class TwoLayerContinuousMLP(ContinuousModel):

    """
    Constructs a Torch multilayer perceptron (MLP) with continuous
    action space. The MLP incorporates an actor and a critic model.
    Both models contain two dense hidden layers, each of size
    num_hidden.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    num_inputs: state space.

    num_hidden: number of neurons in each hidden layer.

    beta: magnitude of entropy regularization.
    ------------------------------------------------------------------
    """

    def __init__(self, num_inputs, num_hidden, beta=0.005):

        """
        Construction of the actor and the critic model.
        """

        super(TwoLayerContinuousMLP, self).__init__(num_inputs, num_hidden, beta)

        self.actor_input_weights = torch.nn.Linear(num_inputs, num_hidden)
        self.actor_hidden_weights = torch.nn.Linear(num_hidden, num_hidden)
        self.actor_output_mu_weights = torch.nn.Linear(num_hidden, 1)
        self.actor_output_sigma_weights = torch.nn.Linear(num_hidden, 1)

        self.critic_input_weights = torch.nn.Linear(num_inputs, num_hidden)
        self.critic_hidden_weights = torch.nn.Linear(num_hidden, num_hidden)
        self.critic_output_weights = torch.nn.Linear(num_hidden, 1)

    def feed_forward(self, states):

        """
        Computes the mean and standard deviations of a normal
        distribution (actor) and the values (critic) by forward
        propagating a numpy array. The shape of the numpy array
        must be (batchsize, num_inputs).
        """

        # pylint: disable=E1101

        actor_input = torch.relu(self.actor_input_weights(states))
        actor_hidden = torch.relu(self.actor_hidden_weights(actor_input))
        mu = 2 * torch.tanh(self.actor_output_mu_weights(actor_hidden))
        sigma = (
            torch.nn.functional.softplus(self.actor_output_sigma_weights(actor_hidden))
            + 0.001
        )

        critic_input = torch.relu(self.critic_input_weights(states))
        critic_hidden = torch.relu(self.critic_hidden_weights(critic_input))
        values = self.critic_output_weights(critic_hidden)

        # pylint: enable=E1101

        return mu, sigma, values

    def calc_loss(self, states, actions, target_values):

        """
        Calculates the loss of the actor and critic model.
        Returns the total loss by taking the average of the
        summed loss of the actor and critic model.
        """

        # pylint: disable=E1101
        # pylint: disable=E1102

        self.train()

        # cast states to tensor
        states = torch.from_numpy(states)
        # feed forward
        mu, sigma, values = self.feed_forward(states)
        # squeeze & transpose mu, sigma, values
        mu = mu.squeeze(-1).transpose(0, 1)
        sigma = sigma.squeeze(-1).transpose(0, 1)
        values = values.squeeze(-1).transpose(0, 1)
        # cast target values to tensor
        target_values = torch.from_numpy(target_values)
        # calculate advantage
        advantage = target_values - values
        # calculate critic loss
        critic_loss = advantage.pow(2)
        # create normal distribution
        policy = torch.distributions.Normal(mu, sigma)
        # calculate entropy (for exploration)
        entropy = 0.5 + 0.5 * np.log(2 * np.pi) + torch.log(policy.scale)
        # cast actions to tensor
        actions = torch.from_numpy(actions.astype(np.float32))
        # calculate actor loss
        actor_loss = (-1) * (
            policy.log_prob(actions) * advantage.detach() + self.beta * entropy
        )
        # calculate total loss
        total_loss = (actor_loss + critic_loss).mean()

        # pylint: enable=E1101
        # pylint: enable=E1102

        return total_loss


class BiLSTMContinuousModel(ContinuousModel):

    """
    Constructs a Torch model with continuous action space. The model
    incorporates a bidirectional LSTM layer for preprocessing the
    input sequence of jobs as well as an actor and a critic model.
    Both, the actor and the critic model, contain two dense hidden
    layers, each of size num_hidden_dense.

    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    num_inputs: tuple containing the number of job states and the
                number of systems states.

    num_hidden_lstm: size of the hidden state of the lstm layer.

    num_hidden_dense: number of neurons in the dense hidden layers.

    beta: magnitude of entropy regularization.
    ------------------------------------------------------------------
    """

    def __init__(self, num_inputs, num_hidden_lstm, num_hidden_dense, beta=0.005):

        """
        Construction of the actor and the critic model.
        """

        super(BiLSTMContinuousModel, self).__init__(
            sum(num_inputs), num_hidden_dense, beta
        )
        self.num_job_inputs, self.num_system_inputs = num_inputs
        self.num_hidden_lstm = num_hidden_lstm

        self.hidden_state = None

        # pylint: disable=E1101

        self.hidden_state_list = [
            torch.zeros(2, 1, self.num_hidden_lstm),
            torch.zeros(2, 1, self.num_hidden_lstm),
        ]

        # pylint: enable=E1101

        self.lstm_layer = torch.nn.LSTM(
            self.num_job_inputs, num_hidden_lstm, bidirectional=True
        )

        self.actor_hidden_layer = torch.nn.Linear(
            num_hidden_lstm + self.num_system_inputs, num_hidden_dense
        )
        self.actor_output_layer_mu = torch.nn.Linear(num_hidden_dense, 1)
        self.actor_output_layer_sigma = torch.nn.Linear(num_hidden_dense, 1)

        self.critic_hidden_layer = torch.nn.Linear(
            num_hidden_lstm + self.num_system_inputs, num_hidden_dense
        )
        self.critic_output_layer = torch.nn.Linear(num_hidden_dense, 1)

    def feed_forward(self, states, future_state=False, calc_loss=False):

        """
        Computes the mean and standard deviations of a normal 
        distribution (actor) and the values (critic) by forward 
        propagating a numpy array. The shape of the numpy array 
        must be (batchsize, num_inputs).
        """

        # pylint: disable=E1101

        # reshape states if not in required shape
        if len(states.size()) != 3:
            states = self.reshape_states(states)

        # split states into job features and system features
        job_features = states[:, :, : self.num_job_inputs]
        system_features = states[:, :, self.num_job_inputs :].squeeze()

        # LSTM pre-processing
        if future_state:
            hidden_state = (
                self.reshape_states(self.hidden_state_list[0][:, -2, :]),
                self.reshape_states(self.hidden_state_list[1][:, -2, :]),
            )
            lstm_out, _ = self.lstm_layer(job_features, hidden_state)
        elif calc_loss:
            hidden_state_batch = (
                self.hidden_state_list[0][:, : states.shape[1], :].detach(),
                self.hidden_state_list[1][:, : states.shape[1], :].detach(),
            )
            lstm_out, _ = self.lstm_layer(job_features, hidden_state_batch)
            self.hidden_state_list = [self.hidden_state[0], self.hidden_state[1]]
        else:
            lstm_out, self.hidden_state = self.lstm_layer(
                job_features, self.hidden_state
            )
            self.hidden_state_list[0] = torch.cat(
                tensors=(self.hidden_state_list[0], self.hidden_state[0]), dim=1
            )
            self.hidden_state_list[1] = torch.cat(
                tensors=(self.hidden_state_list[1], self.hidden_state[1]), dim=1
            )
        lstm_out = torch.add(
            lstm_out[:, :, : self.num_hidden_lstm],
            lstm_out[:, :, self.num_hidden_lstm :],
        )
        lstm_out = torch.div(lstm_out, 2).squeeze()

        # concatenate lstm_out with system_features
        states = torch.cat(tensors=(lstm_out, system_features), dim=(-1))

        # Actor activation
        actor_hidden_activation = torch.relu(self.actor_hidden_layer(states))
        mu = 2 * torch.tanh(self.actor_output_layer_mu(actor_hidden_activation))
        sigma = (
            torch.nn.functional.softplus(
                self.actor_output_layer_sigma(actor_hidden_activation)
            )
            + 0.001
        )

        # Critic activation
        critic_hidden_activation = torch.relu(self.critic_hidden_layer(states))
        values = self.critic_output_layer(critic_hidden_activation)

        # pylint: enable=E1101

        return mu, sigma, values

    def calc_loss(self, states, actions, target_values):

        """
        Calculates the loss of the actor and critic model.
        Returns the total loss by taking the average of the 
        summed loss of the actor and critic model.
        """

        # pylint: disable=E1101
        # pylint: disable=E1102

        self.train()

        # cast states to tensor
        states = torch.from_numpy(states)
        # feed forward
        mu, sigma, values = self.feed_forward(states, calc_loss=True)
        # cast target values to tensor
        target_values = torch.from_numpy(target_values)
        # calculate advantage
        advantage = target_values - values
        # calculate critic loss
        critic_loss = advantage.pow(2)
        # create normal distribution
        policy = torch.distributions.Normal(mu, sigma)
        # calculate entropy (for exploration)
        entropy = 0.5 + 0.5 * np.log(2 * np.pi) + torch.log(policy.scale)
        # cast actions to tensor
        actions = torch.from_numpy(actions.astype(np.float32))
        # calculate actor loss
        actor_loss = (-1) * (
            policy.log_prob(actions) * advantage.detach() + self.beta * entropy
        )
        # calculate total loss
        total_loss = (actor_loss + critic_loss).mean()

        # pylint: enable=E1101
        # pylint: enable=E1102

        return total_loss

    def reshape_states(self, states):

        states = states.reshape(states.shape[0], -1, states.shape[1])

        return states

###########################################################################################################################################################################################################
