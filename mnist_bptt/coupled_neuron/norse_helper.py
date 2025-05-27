from typing import NamedTuple, Tuple
import torch
import torch.jit

from norse.torch.functional.threshold import threshold
import norse.torch.utils.pytree as pytree
import torch
from norse.torch.functional.lif import (
    LIFState,
    LIFParameters,
)

def CoupledLIFStep(
    input_spikes: torch.Tensor,
    state: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParameters,
    dt: float = 0.001,
    use_coupled: bool = False,
    use_movingAvg: bool = False,
    previous_M: torch.Tensor = None,
    lam: float = 0.5,   # New parameter for previous_M weight
    eta: float = 0.5,   # New parameter for current M weight
) -> Tuple[torch.Tensor, LIFState]:
    M = torch.bmm(state.z.unsqueeze(2), state.z.unsqueeze(1))

    if use_movingAvg and previous_M is not None:
        if previous_M.shape == M.shape:
            M = (lam * previous_M) + (eta * M)

    M = M.to(input_spikes.device)
    coupled_weight = recurrent_weights * M

    if use_coupled:
        i_jump = (
            state.i
            + torch.nn.functional.linear(input_spikes, input_weights)
            + torch.bmm(state.z.unsqueeze(1), coupled_weight).squeeze(1)
        )
    else:
        i_jump = (
            state.i
            + torch.nn.functional.linear(input_spikes, input_weights)
            + torch.nn.functional.linear(state.z, recurrent_weights)
        )

    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_decayed = state.v + dv

    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    v_new = (1 - z_new.detach()) * v_decayed + z_new.detach() * p.v_reset

    return z_new, LIFState(z_new, v_new, i_decayed), M

