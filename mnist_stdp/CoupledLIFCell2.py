import jax
from jax import numpy as jnp, random, jit, nn
from functools import partial
from ngclearn.utils import tensorstats
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, step_euler, step_rk2
from ngclearn.utils.surrogate_fx import secant_lif_estimator

###############################################################################
# Helpers
###############################################################################
@jit
def _update_times(t, s, tols):
    """
    Updates time-of-last-spike (tols) variable.
    """
    _tols = (1. - s) * tols + (s * t)
    return _tols

@jit
def _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_decay=1.):
    """
    Raw voltage dynamics.
    """
    mask = (rfr >= refract_T).astype(jnp.float32)
    dv_dt = (v_rest - v) * v_decay + (j * mask)
    dv_dt = dv_dt * (1. / tau_m)
    return dv_dt

def _dfv(t, v, params):
    """
    Voltage dynamics wrapper.
    """
    j, rfr, tau_m, refract_T, v_rest, v_decay = params
    dv_dt = _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_decay)
    return dv_dt

def _run_cell(dt, j, v, v_thr, v_theta, rfr, skey, tau_m, v_rest, v_reset,
              v_decay, refract_T, integType=0):
    """
    Runs one integration step of the LIF cell dynamics.
    """
    _v_thr = v_theta + v_thr
    v_params = (j, rfr, tau_m, refract_T, v_rest, v_decay)

    if integType == 1:
        _, _v = step_rk2(0., v, _dfv, dt, v_params)
    else:
        _, _v = step_euler(0., v, _dfv, dt, v_params)

    # Determine spikes based on threshold crossing
    s = (_v > _v_thr).astype(jnp.float32)
    _rfr = (rfr + dt) * (1. - s)
    _v = _v * (1. - s) + s * v_reset

    raw_s = s + 0  # keep a copy of raw spikes
    if skey is not None:
        # Optionally enforce exactly one spike per batch if s>0
        m_switch = (jnp.sum(s) > 0.).astype(jnp.float32)
        rS = s * random.uniform(skey, s.shape)
        rS = nn.one_hot(jnp.argmax(rS, axis=1), num_classes=s.shape[1],
                        dtype=jnp.float32)
        s = s * (1. - m_switch) + rS * m_switch
    return _v, s, raw_s, _rfr

@partial(jit, static_argnums=[3, 4])
def _update_theta(dt, v_theta, s, tau_theta, theta_plus=0.05):
    """
    Updates the adaptive threshold variable using Euler integration.
    """
    theta_decay = jnp.exp(-dt / tau_theta)
    _v_theta = v_theta * theta_decay + s * theta_plus
    return _v_theta


class CoupledLIFCell(JaxComponent):
    """
    A LIF cell that includes an optional recurrent coupling term M = s_{t-1} * s_{t-1}^T.
    The cell dynamics are given by:
        tau_m * dv/dt = (v_rest - v) + (j + i_recurrent)*R_m
    """

    def __init__(
        self,
        name,
        n_units,
        tau_m,
        resist_m=1.,
        thr=-52.,
        v_rest=-65.,
        v_reset=-60.,
        v_decay=1.,
        tau_theta=1e7,
        theta_plus=0.05,
        refract_time=5.,
        thr_jitter=0.,
        one_spike=False,
        integration_type="euler",
        recurrent_weights=None,
        use_coupled=True,
        use_movingAvg=False,
        lam=0.5,
        eta=0.5,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        # Integration settings
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        # Membrane parameters
        self.tau_m = tau_m
        self.R_m = resist_m
        self.one_spike = one_spike
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_decay = v_decay
        assert self.R_m > 0.

        self.tau_theta = tau_theta
        self.theta_plus = theta_plus
        self.refract_T = refract_time
        self.thr = thr

        # Layer size
        self.batch_size = 1
        self.n_units = n_units

        # Surrogate function for spike emission
        self.spike_fx, self.d_spike_fx = secant_lif_estimator()

        # Initialize compartments
        restVals = jnp.zeros((self.batch_size, self.n_units))
        thr0 = 0.
        if thr_jitter > 0.:
            key, subkey = random.split(self.key.value)
            thr0 = random.uniform(
                subkey, (1, n_units),
                minval=-thr_jitter, maxval=thr_jitter, dtype=jnp.float32
            )

        self.j = Compartment(restVals)
        self.v = Compartment(restVals + self.v_rest)
        self.s = Compartment(restVals)
        self.s_raw = Compartment(restVals)
        self.rfr = Compartment(restVals + self.refract_T)
        self.thr_theta = Compartment(restVals + thr0)
        self.tols = Compartment(restVals)
        self.surrogate = Compartment(restVals + 1.)

        # Recurrent weight matrix for coupling
        if recurrent_weights is None:
            self.recurrent_weights = jnp.eye(n_units, dtype=jnp.float32)
        else:
            self.recurrent_weights = recurrent_weights

        # === Additional Coupling-Related Setup ===
        self.use_coupled = use_coupled
        self.use_movingAvg = use_movingAvg
        self.lam = lam
        self.eta = eta
        # We'll store the "previous M" in a new compartment
        M_init = jnp.zeros((self.batch_size, self.n_units, self.n_units),
                           dtype=jnp.float32)
        self.prev_M = Compartment(M_init)

    ###########################################################################
    # Advance State
    ###########################################################################
    @staticmethod
    def _advance_state(
        t, dt, tau_m, R_m, v_rest, v_reset, v_decay, refract_T,
        thr, tau_theta, theta_plus, one_spike, intgFlag, d_spike_fx,
        key, j, v, s, rfr, thr_theta, tols,
        recurrent_weights,
        # new compartments and flags
        prev_M, use_coupled, use_movingAvg, lam, eta
    ):
        # Handle one-spike mode if enabled.
        skey = None
        if one_spike:
            key, skey = random.split(key, 2)

        # --- Compute new M from outer product s*s^T ---
        M_new = jnp.einsum('bi,bj->bij', s, s)

        # Optional moving average with previous_M
        if use_movingAvg:
            # shapes must match (they do if batch_size, n_units are consistent)
            M_new = lam * prev_M + eta * M_new

        # Either use the newly formed matrix M_new or keep linear recurrence
        if use_coupled:
            # Coupled weight is rec_weights * M
            coupled_weight = recurrent_weights * M_new
            # Recurrent input is s * (coupled weight) for each batch
            i_recurrent = jnp.einsum('bi,bij->bj', s, coupled_weight)
        else:
            # Plain old linear recurrence: i_recurrent = s * W_rec
            # shape check: s is [B, N], rec is [N, N]
            i_recurrent = s @ recurrent_weights

        # Combine external input + recurrent input
        j_total = (j + i_recurrent) * R_m

        # Run the LIF cell dynamics
        v, s_new, raw_spikes, rfr = _run_cell(
            dt, j_total, v, thr, thr_theta, rfr, skey,
            tau_m, v_rest, v_reset, v_decay, refract_T,
            intgFlag
        )

        # Surrogate derivative for backprop
        surrogate = d_spike_fx(v, thr + thr_theta)
        if tau_theta > 0.:
            thr_theta = _update_theta(dt, thr_theta, raw_spikes, tau_theta, theta_plus)

        tols = _update_times(t, s_new, tols)
        return (v, s_new, raw_spikes, rfr, thr_theta, tols, key,
                surrogate, M_new)

    @resolver(_advance_state)
    def advance_state(self, v, s, s_raw, rfr, thr_theta, tols, key, surrogate, prev_M):
        """
        This resolver automatically unpacks/repacks the updated compartments.
        """
        self.v.set(v)
        self.s.set(s)
        self.s_raw.set(s_raw)
        self.rfr.set(rfr)
        self.thr_theta.set(thr_theta)
        self.tols.set(tols)
        self.key.set(key)
        self.surrogate.set(surrogate)
        self.prev_M.set(prev_M)

    @staticmethod
    def _reset(batch_size, n_units, v_rest, refract_T):
        restVals = jnp.zeros((batch_size, n_units))
        j = restVals
        v = restVals + v_rest
        s = restVals
        s_raw = restVals
        rfr = restVals + refract_T
        tols = restVals
        surrogate = restVals + 1.
        M_init = jnp.zeros((batch_size, n_units, n_units), dtype=jnp.float32)
        return j, v, s, s_raw, rfr, tols, surrogate, M_init

    @resolver(_reset)
    def reset(self, j, v, s, s_raw, rfr, tols, surrogate, prev_M):
        self.j.set(j)
        self.v.set(v)
        self.s.set(s)
        self.s_raw.set(s_raw)
        self.rfr.set(rfr)
        self.tols.set(tols)
        self.surrogate.set(surrogate)
        self.prev_M.set(prev_M)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name,
                  threshold_theta=self.thr_theta.value,
                  key=self.key.value,
                  prev_M=self.prev_M.value)

    def load(self, directory, seeded=False, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.thr_theta.set(data['threshold_theta'])
        self.prev_M.set(data['prev_M'])
        if seeded:
            self.key.set(data['key'])

    @classmethod
    def help(cls):
        properties = {
            "cell_type": (
                "LIFCell with optional recurrent coupling.\n"
                "Now supports either linear recurrence or an outer-product-based coupling.\n"
                "Optionally applies a moving average to the coupling matrix M."
            )
        }
        compartment_props = {
            "inputs": {"j": "External input current"},
            "states": {
                "v": "Membrane potential",
                "rfr": "Refractory state",
                "thr_theta": "Adaptive threshold",
                "key": "JAX PRNG key",
                "prev_M": "Previous outer-product coupling matrix"
            },
            "outputs": {"s": "Spikes", "tols": "Time-of-last-spike"},
        }
        hyperparams = {
            "n_units": "Number of neurons in this cell",
            "tau_m": "Membrane time constant",
            "resist_m": "Membrane resistance",
            "thr": "Base voltage threshold (mV)",
            "v_rest": "Resting membrane potential (mV)",
            "v_reset": "Reset potential after spike (mV)",
            "v_decay": "Voltage leak factor",
            "tau_theta": "Adaptive threshold time constant",
            "theta_plus": "Threshold increment per spike",
            "refract_time": "Refractory period (ms)",
            "thr_jitter": "Noise scale for initial threshold",
            "one_spike": "Enforce single spike per time step",
            "integration_type": "Integration method (euler or rk2)",
            "recurrent_weights": "Recurrent weight matrix for coupling",
            "use_coupled": "Toggle outer-product coupling vs. linear recurrence",
            "use_movingAvg": "If True, uses a weighted moving average of M",
            "lam": "Weight on previous M in the moving average",
            "eta": "Weight on new M in the moving average",
        }
        info = {
            cls.__name__: properties,
            "compartments": compartment_props,
            "dynamics": (
                "tau_m * dv/dt = (v_rest - v) + (j + i_recurrent)*R_m"
            ),
            "hyperparameters": hyperparams,
        }
        return info

    def __repr__(self):
        comps = [
            varname for varname in dir(self)
            if Compartment.is_compartment(getattr(self, varname))
        ]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = ", ".join(f"{k}: {v}" for k, v in stats.items())
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines

