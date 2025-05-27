import torch

from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_step_sparse,
    lif_feed_forward_step,
    lif_feed_forward_step_sparse,
)
from norse.torch.functional.adjoint.lif_adjoint import (
    lif_adjoint_step,
    lif_adjoint_step_sparse,
    lif_feed_forward_adjoint_step,
    lif_feed_forward_adjoint_step_sparse,
)
from norse.torch.module.snn import SNN, SNNCell, SNNRecurrent, SNNRecurrentCell
from norse.torch.utils.clone import clone_tensor

from typing import Any, Callable, List, Optional, Tuple
import torch

FeedforwardActivation = Callable[
    [torch.Tensor, torch.Tensor, torch.nn.Module, float],
    Tuple[torch.Tensor, torch.Tensor],
]

RecurrentActivation = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module, float],
    Tuple[torch.Tensor, torch.Tensor],
]

class CoupledSNNCell(SNNRecurrentCell):
    def __init__(
        self,
        activation: RecurrentActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        input_size: int,
        hidden_size: int,
        p: torch.nn.Module,
        input_weights: Optional[torch.Tensor] = None,
        recurrent_weights: Optional[torch.Tensor] = None,
        autapses: bool = False,
        dt: float = 0.001,
        activation_sparse: Optional[RecurrentActivation] = None,
        use_coupled: bool = False,
        use_movingAvg: bool = False,
        lam: float = 0.5, 
        eta: float = 0.5, 
    ):
        super().__init__(
            activation,
            state_fallback,
            input_size,
            hidden_size,
            p,
            input_weights,
            recurrent_weights,
            autapses,
            dt,
            activation_sparse,
        )
        self.use_coupled = use_coupled
        if not use_coupled:
            self.use_movingAvg = False
        else:
            self.use_movingAvg = use_movingAvg
        self.previous_M = None
        self.lam = lam
        self.eta = eta

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)
        # Pass lam and eta to the activation function
        z_new, new_state, M = self.activation(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            self.p,
            self.dt,
            use_coupled=self.use_coupled,
            use_movingAvg=self.use_movingAvg,
            previous_M=self.previous_M,
            lam=self.lam,
            eta=self.eta,
        )
        self.previous_M = M.detach()
        return z_new, new_state

