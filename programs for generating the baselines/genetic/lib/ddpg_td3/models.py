"""
Note: There seem to be several bugs in PyTorch that are responsible for 
some false error messages in the PROBLEMS console, although the code will 
run as expected. Some error messages indicate that methods, such as'sigmoid',
are not existing (pylint(no-member)). The problem is reported here:
https://github.com/pytorch/pytorch/issues/701. Another error message indicates 
that torch.tensor is not callable (pylint(not-callable)). The problem is 
reported here: https://github.com/pytorch/pytorch/issues/24807. The following 
comment lines are therefore sometimes integrated to disable and enable the
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

class MLP_Model(torch.nn.Module):

    """
    Multilayer Perceptron (MLP) with ELU activation in the hidden layers
    and Sigmoid or Linear activation in the output layer. The MLP can 
    represent an actor (is_actor=True) or a critic model.

    --------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------
    num_inputs: Number of states (is_actor=True) or number of states
                and actions (is_actor=False).

    num_hidden: List of non-negative integer values. len(num_hidden)
                indicates the number of hidden layers, while 
                num_hidden[:] indicates the number of neurons in 
                each hidden layer.
                
    is_actor: If set to True, the agent will represent an actor 
              network, by which means the output layer underlies 
              sigmoid activation function. Otherwise, the agent will 
              represent a critic network, by which means the output 
              layer underlies linear activation. Furthermore, 
              num_inputs is increased by one to also consider the 
              action as input value.
    --------------------------------------------------------------------
    """

    def __init__(self, num_inputs, num_hidden=[100,100], is_actor=True):

        """
        Construction of the MLP
        """

        assert (
            isinstance(num_hidden, list)
            and all(isinstance(i, int) and i > 0 for i in num_hidden)
        ), "Parameter 'num_hidden' must be a list with non-negative integer values."

        super(MLP_Model, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.is_actor = is_actor

        if self.is_actor:
            self.mask_with = -1e10

        self.layers = torch.nn.Sequential()

        if num_hidden:
            self.layers.add_module("input", torch.nn.Linear(self.num_inputs, num_hidden[0]))
            self.layers.add_module("input activation", torch.nn.ELU())
            for i in range(len(num_hidden)-1):
                self.layers.add_module(
                    "hidden_{}".format(i),
                    torch.nn.Linear(num_hidden[i], num_hidden[i+1])
                )
                self.layers.add_module("hidden_{} activation".format(i), torch.nn.ELU())
        
        self.layers.add_module(
            "output", 
            torch.nn.Linear(num_hidden[-1] if num_hidden else self.num_inputs, 1)
        )

        if is_actor:
            self.layers.add_module("output activation", torch.nn.Sigmoid())

    def __call__(self, inputs, mask_zeros=True, no_grad=True):
        
        """
        Forward propagation
        
        --------------------------------------------------------------------
        Parameters:
        --------------------------------------------------------------------
        inputs: Nested list, numpy array or tensor of shape (batch size X
                sequence length X number of states if is_actor else state 
                number of states + number of actions).
        
        mask_zeros: If True, the model computes a mask over all zero-states.
                    If the model is an Actor (is_actor == True), the mask
                    ensures that actions resulting from zero-states are not
                    taken into account for scheduling decisions. Otherwise,
                    the mask ensures that zero-state-action relations are
                    valued with zero.
        
        no_grad: Set to True if the model collects experience in the
                 environment or during deployment. Set to False when
                 performing a training step.
        --------------------------------------------------------------------
        """
        
        # pylint: disable=E1101
        # pylint: disable=E1102

        # Cast states to tensor if necessary
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # Compute mask over zero-states
        if mask_zeros:
            mask = inputs.abs().sum(dim=-1)
            mask = mask==0 if self.is_actor else mask!=0
        else:
            mask = torch.tensor(False)

        # pylint: enable=E1101
        # pylint: enable=E1102

        # Forward propagation
        if no_grad:
            self.eval()
            with torch.no_grad():
                out = self.__forward(inputs, mask)
        else:
            out = self.__forward(inputs, mask)
        
        return out
    
    def __forward(self, tensor, mask=None):
        
        """
        Private method. Instead, call the model itself to perform 
        forward propagation.
        """
        
        # pylint: disable=E1101

        out = self.layers(tensor).squeeze()

        if mask.any():
            out = out + mask * self.mask_with if self.is_actor else out * mask
            
        return out


class LSTM_MLP_Actor(torch.nn.Module):

    """
    Long Short-Term Memory (LSTM) model for pre-processing job sequences
    and Multilayer Perception (MLP) for pre-processing system states.
    The output of both networks is merged and post-processed by a 
    subsequent MLP model. The hidden layers of both MLP models underlie
    ELU activation, while the output layer underlies Sigmoid activation.
    This model can be only deployed as an Actor model.
    
    --------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------
    num_inputs: Tuple containing the number of job states and the
                number of systems states.
    
    num_hidden_job: Non-empty list of tuples. Each tuple represents an 
                    LSTM layer for processing job states. The first
                    element of each tuple must be a non-negative integer
                    value indicating the number of LSTM cells. The
                    second element must be True or False indicating
                    whether hidden LSTM layer is bidirectional (True) or
                    unidirectional (False).
    
    num_hidden_system: Non-empty list of non-negative integer values.
                       len(num_hidden_system_states) indicates the
                       number of hidden dense layers for processing 
                       system states, while num_hidden[:] indicates the
                       number of neurons in each hidden dense layer.
    
    num_hidden_joined: Non-empty list of non-negative integer values.
                       len(num_hidden_joined) indicates the number of
                       hidden dense layers for processing joined job and
                       system states, while num_hidden[:] indicates the 
                       number of neurons in each  hidden dense layer.
    --------------------------------------------------------------------
    """

    def __init__(
        self, 
        num_inputs, 
        num_hidden_job=[(100, True), (100, True)],
        num_hidden_system=[100, 100],
        num_hidden_joined=[100, 100]
    ):

        """
        Construction of the LSTM-MLP model
        """

        assert (
            isinstance(num_inputs, tuple)
            and len(num_inputs) == 2
            and all(isinstance(i, int) and i > 0 for i in num_inputs)
        ), "Parameter 'num_inputs' must be a non-empty 2-tuple with non-negative integer values."

        assert (
            isinstance(num_hidden_job, list)
            and len(num_hidden_job) > 0
            and all(
                isinstance(i, tuple)
                and len(i) == 2
                and (isinstance(i[0], int) and i[0] > 0)
                and isinstance(i[1], bool) for i in num_hidden_job
            )
        ), """Parameter 'num_hidden_job' must be a non-empty list of 2-tuples each with a non-negative
              integer value as first element and True or False as second element."""

        assert (
            isinstance(num_hidden_system, list)
            and num_hidden_system
            and all(isinstance(i, int) and i > 0 for i in num_hidden_system)
        ), "Parameter 'num_hidden_system' must be a non-empty list with non-negative integer values."

        assert (
            isinstance(num_hidden_joined, list)
            and num_hidden_joined
            and all(isinstance(i, int) and i > 0 for i in num_hidden_joined)
        ), "Parameter 'num_hidden_joined' must be a non-empty list with non-negative integer values."

        super(LSTM_MLP_Actor, self).__init__()

        self.num_inputs = num_inputs
        self.num_job_inputs, self.num_system_inputs = num_inputs
        self.num_hidden_job_states = num_hidden_job
        self.num_hidden_system_states = num_hidden_system
        self.num_hidden_joined = num_hidden_joined
        self.mask_with = -1e10
        self.model ={}

        # LSTM model for pre-processing job sequences
        self.job_layers = torch.nn.Sequential()
        self.job_layers.add_module("job_layer_0",
            torch.nn.LSTM(
                self.num_job_inputs,
                num_hidden_job[0][0],
                batch_first=True,
                bidirectional=num_hidden_job[0][1]
            )
        )
        for i in range(len(num_hidden_job)-1):
            self.job_layers.add_module("job_layer_{}".format(i+1),
                torch.nn.LSTM(
                    num_hidden_job[i][0],
                    num_hidden_job[i+1][0],
                    batch_first=True,
                    bidirectional=num_hidden_job[i][1]
                )
            )
        self.model["job_layers"] = self.job_layers
        
        # MLP model for pre-processing system states
        self.system_layers = torch.nn.Sequential()
        self.system_layers.add_module("system_layer_0", 
            torch.nn.Linear(self.num_system_inputs, num_hidden_system[0])
        )
        self.system_layers.add_module("system_layer_0 activation", torch.nn.ELU())
        for i in range(len(num_hidden_system)-1):
            self.system_layers.add_module("system_layer_{}".format(i+1),
                torch.nn.Linear(num_hidden_system[i], num_hidden_system[i+1])
            )
            self.system_layers.add_module("system_layer_{} activation".format(i+1), torch.nn.ELU())
        # Add this layer in addition if the number of neurons of the last hidden layer
        # for processing system states does not correspond to the number of neurons of
        # the last hidden layer for processing job states.
        if num_hidden_system[-1] != num_hidden_job[-1][0]:
            self.system_layers.add_module("system_layer_{}".format(len(num_hidden_system)),
                torch.nn.Linear(num_hidden_system[-1], num_hidden_job[-1][0])
            )
            self.system_layers.add_module(
                "system_layer_{} activation".format(len(num_hidden_system)), torch.nn.ELU()
            )
        self.model["system_layers"] = self.system_layers
        
        # MLP model for post-processing joined job states and system states
        self.joined_layers = torch.nn.Sequential()
        self.joined_layers.add_module("joined_layer_0", 
            torch.nn.Linear(num_hidden_job[-1][0], num_hidden_joined[0])
        )
        self.joined_layers.add_module("joined_layer_0 activation", torch.nn.ELU())
        for i in range(len(num_hidden_joined)-1):
            self.joined_layers.add_module("joined_layer_{}".format(i+1),
                torch.nn.Linear(num_hidden_joined[i], num_hidden_joined[i+1])
            )
            self.joined_layers.add_module("joined_layer_{} activation".format(i+1), torch.nn.ELU())
        self.joined_layers.add_module("joined_layer_{}".format(len(num_hidden_joined)),
            torch.nn.Linear(num_hidden_joined[-1], 1)
        )
        self.joined_layers.add_module(
            "joined_layer_{} activation".format(len(num_hidden_joined)), torch.nn.Sigmoid()
        )
        self.model["joined_layers"] = self.joined_layers
    
    def __call__(self, states, mask_zeros=True, no_grad=True):
        
        """
        Forward propagation
        
        --------------------------------------------------------------------
        Parameters:
        --------------------------------------------------------------------
        states: Nested list, numpy array or tensor of shape (batch size X 
                sequence length X number of states).
        
        mask_zeros: If True, the model computes a mask over all zero-states,
                    which ensures that actions resulting from zero-states
                    are not taken into account for scheduling decisions.
        
        no_grad: Set to True if the model collects experience in the
                 environment or during deployment. Set to False when
                 performing a training step.
        --------------------------------------------------------------------
        """
        
        # pylint: disable=E1101
        # pylint: disable=E1102

        # Cast states to tensor if necessary
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        
       # Compute mask over zero-states
        if mask_zeros:
            mask = inputs.abs().sum(dim=-1)==0

        else:
            mask = torch.tensor(False)

        # pylint: enable=E1101
        # pylint: enable=E1102

        # Forward propagation
        if no_grad == True:
            self.eval()
            with torch.no_grad():
                return self.__forward(states, mask)
        else:
            return self.__forward(states, mask)
    
    def __forward(self, tensor, mask=None):
        
        """
        Private method. Instead, call the model itself to perform 
        forward propagation.
        """

        # split tensor into job and system states
        job_states = tensor[:, :, :self.num_job_inputs]
        system_states = tensor[:, :, self.num_job_inputs:]

        # pylint: disable=E1101

        # forward propagation of job states (LSTM)
        job_out, _ = self.job_layers[0](job_states)
        if self.job_layers[0].bidirectional:
            job_out = self.__mean_bidirect(job_out, self.job_layers[0])
        for i in range(1, len(self.num_hidden_job_states)):
            job_out, _ = self.job_layers[i](job_out)
            if self.job_layers[i].bidirectional:
                job_out = self.__mean_bidirect(job_out, self.job_layers[i])
        
        # forward propagation of system states (MLP)
        system_out = self.system_layers(system_states)
        
        # join pre-procecced job and system states
        joined_out = torch.add(job_out, system_out)
        
        # forward propagation of joined states (MLP)
        joined_out = self.joined_layers(joined_out).squeeze()
        
        # pylint: enable=E1101
        
        # apply mask (if given)
        if mask.any():
            joined_out = joined_out + mask * self.mask_with
        
        return joined_out
    
    def __mean_bidirect(self, tensor, layer):

        """
        Private method. Calculates the mean activation 
        of a tensor from a bidirectional LSTM layer.
        """

        # pylint: disable=E1101

        return torch.div(
            torch.add(
                tensor[:,:,:layer.hidden_size],
                tensor[:,:,layer.hidden_size:]
            ), 2
        )

        # pylint: enable=E1101