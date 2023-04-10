"""
The implementation of the shared optimizers is inspired by: 
https://github.com/MorvanZhou/pytorch-A3C/

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
"""

import torch


class SharedAdam(torch.optim.Adam):

    """
    Wrapper of Torch's Adam optimizer, in order to enable parameter
    sharing between a set of workers.
    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    params Iterable of parameters to optimize or dicts defining
           parameter groups

    lr: Learning rate (default: 1e-3)

    betas: Coefficients used for computing running averages of gradient 
           and its square (default: (0.9, 0.999))

    eps: Term added to the denominator to improve numerical stability
         (default: 1e-8)

    weight_decay: Weight decay (L2 penalty) (default: 0)

    amsgrad: whether to use the AMSGrad variant of this algorithm from 
             the paper "On the Convergence of Adam and Beyond"
             (default: False)
    ------------------------------------------------------------------
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):

        super(SharedAdam, self).__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        # initialize state
        for group in self.param_groups:

            for param in group["params"]:

                state = self.state[param]
                state["step"] = 0

                # pylint: disable=E1101

                state["exp_avg"] = torch.zeros_like(param.data)
                state["exp_avg_sq"] = torch.zeros_like(param.data)

                # pylint: enable=E1101

                # share in memory
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


class SharedRMSprop(torch.optim.RMSprop):

    """
    Wrapper of Torch's RMSprop optimizer, in order to enable parameter
    sharing between a set of workers.
    ------------------------------------------------------------------
    Parameters:
    ------------------------------------------------------------------
    params Iterable of parameters to optimize or dicts defining
           parameter groups

    lr: Learning rate (default: 1e-2)

    alpha: Smoothing constant (default: 0.99)

    eps: Term added to the denominator to improve numerical stability
         (default: 1e-8)

    weight_decay: Weight decay (L2 penalty) (default: 0)

    momentum: Momentum factor (default: 0)

    centered: If True, compute the centered RMSProp, the gradient 
              is normalized by an estimation of its variance 
              (default: False)
    ------------------------------------------------------------------
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):

        super(SharedRMSprop, self).__init__(
            params=params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )

        # initialize state
        for group in self.param_groups:

            for param in group["params"]:

                state = self.state[param]
                state["step"] = 0

                # pylint: disable=E1101
                state["exp_avg"] = torch.zeros_like(param.data)
                state["exp_avg_sq"] = torch.zeros_like(param.data)

                # pylint: enable=E1101

                # share in memory
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
