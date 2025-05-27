import torch

from norse.torch.functional.lif import (
    LIFState,
    LIFParameters,
    lif_step,
    lif_step_sparse,
)
from norse.torch.functional.adjoint.lif_adjoint import (
    lif_adjoint_step,
    lif_adjoint_step_sparse,
)

from coupled_neuron.coupled_snn import CoupledSNNCell
from coupled_neuron.norse_helper import CoupledLIFStep  # Using CoupledLIFStep

from typing import Any, Callable, List, Optional, Tuple
import torch

class CoupledLIFCell(CoupledSNNCell):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFParameters = LIFParameters(),
        lam: float = 0.5,  
        eta: float = 0.5,  
        **kwargs,
    ):
        super().__init__(
            activation=CoupledLIFStep,
            state_fallback=self.initial_state,
            p=LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            input_size=input_size,
            hidden_size=hidden_size,
            lam=lam,
            eta=eta,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = LIFState(
            z=(
                torch.zeros(
                    dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ).to_sparse()
                if input_tensor.is_sparse
                else torch.zeros(
                    dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                )
            ),
            v=torch.full(
                dims,
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                dims,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state

