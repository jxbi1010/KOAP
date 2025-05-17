from __future__ import division
from __future__ import unicode_literals

from typing import Iterable, Optional
import copy
import contextlib

import torch


from typing import Iterable, Optional
import copy
import contextlib

import torch

class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float, use_num_updates: bool = True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.parameters = list(parameters)  # Store direct references to parameters
        self.shadow_params = [p.clone().detach() for p in self.parameters]
        self.collected_params = None

    def update(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None:
        """
        Update the EMA parameters.
        """
        parameters = self.parameters if parameters is None else list(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = (s_param - param)
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def copy_to(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None:
        """
        Copy the EMA parameters to another set of parameters.
        """
        parameters = self.parameters if parameters is None else list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None:
        """
        Store the current parameters for later restoration.
        """
        parameters = self.parameters if parameters is None else list(parameters)
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None:
        """
        Restore parameters previously stored.
        """
        if self.collected_params is None:
            raise RuntimeError("No stored parameters to restore.")
        parameters = self.parameters if parameters is None else list(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None):
        """
        Context manager for temporarily using EMA parameters.
        """
        parameters = self.parameters if parameters is None else list(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        """
        Move EMA parameters to specified device and dtype.
        """
        self.shadow_params = [p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device) for p in self.shadow_params]
        if self.collected_params is not None:
            self.collected_params = [p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device) for p in self.collected_params]

    def state_dict(self) -> dict:
        """
        Return the current state of the EMA as a dictionary.
        """
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state of the EMA from a dictionary.
        """
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]
        self.collected_params = state_dict["collected_params"]

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)




# # Partially based on:
# # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
# class ExponentialMovingAverage:
#     """
#     Maintains (exponential) moving average of a set of parameters.
#
#     Args:
#         parameters: Iterable of `torch.nn.Parameter` (typically from
#             `model.parameters()`).
#             Note that EMA is computed on *all* provided parameters,
#             regardless of whether or not they have `requires_grad = True`;
#             this allows a single EMA object to be consistantly used even
#             if which parameters are trainable changes step to step.
#
#             If you want to some parameters in the EMA, do not pass them
#             to the object in the first place. For example:
#
#                 ExponentialMovingAverage(
#                     parameters=[p for p in model.parameters() if p.requires_grad],
#                     decay=0.9
#                 )
#
#             will ignore parameters that do not require grad.
#
#         decay: The exponential decay.
#
#         use_num_updates: Whether to use number of updates when computing
#             averages.
#     """
#     def __init__(
#         self,
#         parameters: Iterable[torch.nn.Parameter],
#         decay: float,
#         use_num_updates: bool = True
#     ):
#         if decay < 0.0 or decay > 1.0:
#             raise ValueError('Decay must be between 0 and 1')
#         self.decay = decay
#         self.num_updates = 0 if use_num_updates else None
#         self.parameters = list(parameters)
#         self.shadow_params = [
#             p.clone().detach()
#             for p in parameters
#         ]
#         self.collected_params = None
#         # By maintaining only a weakref to each parameter,
#         # we maintain the old GC behaviour of ExponentialMovingAverage:
#         # if the model goes out of scope but the ExponentialMovingAverage
#         # is kept, no references to the model or its parameters will be
#         # maintained, and the model will be cleaned up.
#
#         # self._params_refs = [weakref.ref(p) for p in parameters]
#
#
#     def update(
#         self,
#         parameters: Optional[Iterable[torch.nn.Parameter]] = None
#     ) -> None:
#         """
#         Update currently maintained parameters.
#
#         Call this every time the parameters are updated, such as the result of
#         the `optimizer.step()` call.
#
#         Args:
#             parameters: Iterable of `torch.nn.Parameter`; usually the same set of
#                 parameters used to initialize this object. If `None`, the
#                 parameters with which this `ExponentialMovingAverage` was
#                 initialized will be used.
#         """
#         if parameters is None:
#             parameters = self.parameters
#         else:
#             parameters = list(parameters)
#             if len(parameters) != len(self.shadow_params):
#                 raise ValueError(
#                     "Number of parameters passed as argument is different "
#                     "from number of shadow parameters maintained by this "
#                     "ExponentialMovingAverage"
#                 )
#
#         decay = self.decay
#         if self.num_updates is not None:
#             self.num_updates += 1
#             decay = min(
#                 decay,
#                 (1 + self.num_updates) / (10 + self.num_updates)
#             )
#         one_minus_decay = 1.0 - decay
#         with torch.no_grad():
#             for s_param, param in zip(self.shadow_params, parameters):
#                 s_param.sub_(one_minus_decay * (s_param - param))
#
#
#     def copy_to(self,parameters):
#
#         for param, s_param in zip(parameters, self.shadow_params):
#             param.data.copy_(s_param.data)
#
#     def store(self,parameters):
#         self.collected_params = [param.clone() for param in parameters]
#
#     def restore(self,parameters):
#
#         for c_param, param in zip(self.collected_params, parameters):
#             param.data.copy_(c_param.data)
#
#     @contextlib.contextmanager
#     def average_parameters(
#         self,
#         parameters: Optional[Iterable[torch.nn.Parameter]] = None
#     ):
#         """
#         Args:
#             parameters: Iterable of `torch.nn.Parameter`; the parameters to be
#                 updated with the stored parameters. If `None`, the
#                 parameters with which this `ExponentialMovingAverage` was
#                 initialized will be used.
#         """
#         if parameters is None:
#             raise ValueError("Parameters must be explicitly provided for `average_parameters` context.")
#
#         self.store(parameters)
#         self.copy_to(parameters)
#         try:
#             yield
#         finally:
#             self.restore(parameters)
#
#     def to(self, device=None, dtype=None) -> None:
#         r"""Move internal buffers of the ExponentialMovingAverage to `device`.
#
#         Args:
#             device: like `device` argument to `torch.Tensor.to`
#         """
#         # .to() on the tensors handles None correctly
#         self.shadow_params = [
#             p.to(device=device, dtype=dtype)
#             if p.is_floating_point()
#             else p.to(device=device)
#             for p in self.shadow_params
#         ]
#         if self.collected_params is not None:
#             self.collected_params = [
#                 p.to(device=device, dtype=dtype)
#                 if p.is_floating_point()
#                 else p.to(device=device)
#                 for p in self.collected_params
#             ]
#         return
#
#     def state_dict(self) -> dict:
#         r"""Returns the state of the ExponentialMovingAverage as a dict."""
#         # Following PyTorch conventions, references to tensors are returned:
#         # "returns a reference to the state and not its copy!" -
#         # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
#         return {
#             "decay": self.decay,
#             "num_updates": self.num_updates,
#             "shadow_params": self.shadow_params,
#             "collected_params": self.collected_params
#         }
#
#     def load_state_dict(self, state_dict: dict) -> None:
#         r"""Loads the ExponentialMovingAverage state.
#
#         Args:
#             state_dict (dict): EMA state. Should be an object returned
#                 from a call to :meth:`state_dict`.
#         """
#         # deepcopy, to be consistent with module API
#         state_dict = copy.deepcopy(state_dict)
#         self.decay = state_dict["decay"]
#         if self.decay < 0.0 or self.decay > 1.0:
#             raise ValueError('Decay must be between 0 and 1')
#         self.num_updates = state_dict["num_updates"]
#         assert self.num_updates is None or isinstance(self.num_updates, int), \
#             "Invalid num_updates"
#
#         # Directly assign the tensors from the state dict to shadow_params
#         self.shadow_params = state_dict['shadow_params']
#         assert isinstance(self.shadow_params, list), \
#             "shadow_params must be a list"
#         assert all(
#             isinstance(p, torch.Tensor) for p in self.shadow_params
#         ), "shadow_params must all be Tensors"
#
#         self.collected_params = state_dict["collected_params"]
#         # if self.collected_params is not None:
#         #     assert isinstance(self.collected_params, list), \
#         #         "collected_params must be a list"
#         #     assert all(
#         #         isinstance(p, torch.Tensor) for p in self.collected_params
#         #     ), "collected_params must all be Tensors"
#         #     assert len(self.collected_params) == len(self.shadow_params), \
#         #         "collected_params and shadow_params had different lengths"
#         # else:
#         #     self.collected_params = None
#         #
#         # if len(self.shadow_params) == len(self._params_refs):
#         #     # Consistant with torch.optim.Optimizer, cast things to consistant
#         #     # device and dtype with the parameters
#         #     params = [p() for p in self._params_refs]
#         #     # If parameters have been garbage collected, just load the state
#         #     # we were given without change.
#         #     if not any(p is None for p in params):
#         #         # ^ parameter references are still good
#         #         for i, p in enumerate(params):
#         #             self.shadow_params[i] = self.shadow_params[i].to(
#         #                 device=p.device, dtype=p.dtype
#         #             )
#         #             if self.collected_params is not None:
#         #                 self.collected_params[i] = self.collected_params[i].to(
#         #                     device=p.device, dtype=p.dtype
#         #                 )
#         # else:
#         #     raise ValueError(
#         #         "Tried to `load_state_dict()` with the wrong number of "
#         #         "parameters in the saved state."
#         #     )
