# Import necessary libraries for neural simulation and visualization
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngcsimlib.operations import summation
from ngclearn.components.other.varTrace import VarTrace
from ngclearn.components.input_encoders.poissonCell import PoissonCell
from CoupledLIFCell2 import CoupledLIFCell
from ngclearn.components.neurons.spiking.LIFCell import LIFCell
from ngclearn.components.synapses import TraceSTDPSynapse, StaticSynapse
from ngclearn.utils.model_utils import normalize_matrix
import ngclearn.utils.weight_distribution as dist

class DC_SNN():
    """
    Structure for constructing the spiking neural model proposed in:

    Diehl, Peter U., and Matthew Cook. "Unsupervised learning of digit recognition
    using spike-timing-dependent plasticity." Frontiers in computational
    neuroscience 9 (2015): 99.

    Args:
        dkey: JAX seeding key
        in_dim: input dimensionality
        hid_dim: dimensionality of the representation layer of neuronal cells
        T: number of discrete time steps to simulate neuronal dynamics
        dt: integration time constant
        exp_dir: experimental directory to save model results
        model_name: unique model name to stamp the output files/dirs with
        loadDir: directory to load model from, overrides initialization/model
            object creation if non-None (Default: None)
    """
    
    def __init__(self, dkey, in_dim=1, hid_dim=100, T=200, dt=1., exp_dir="exp",
                 model_name="snn_stdp", loadDir=None, use_coupled=True,use_movingAvg=True, lam=0.5, eta=0.5, **kwargs):
        # Initialize experimental setup and directory structure
        self.exp_dir = exp_dir
        self.model_name = model_name
        makedir(exp_dir)  # Create main experiment directory
        makedir(exp_dir + "/filters")  # Directory for filter visualizations
        makedir(exp_dir + "/raster")   # Directory for raster plots

        # Set simulation parameters
        self.T = T    # Total simulation time steps (default: 200ms)
        self.dt = dt  # Integration time constant (default: 1ms)
        
        # Neuronal membrane and trace time constants
        tau_m_e = 100.500896468  # Excitatory membrane time constant (ms)
        tau_m_i = 100.500896468  # Inhibitory membrane time constant (ms)
        tau_tr = 20.             # Synaptic trace time constant (ms)

        # STDP (Spike-Timing-Dependent Plasticity) hyperparameters
        Aplus = 1e-2   # LTP (Long-Term Potentiation) learning rate
        Aminus = 1e-4  # LTD (Long-Term Depression) learning rate
        self.wNorm = 78.4  # Weight normalization constraint for synapses

        # Split random key for reproducible initialization
        dkey, *subkeys = random.split(dkey, 12)

        # Load existing model or create new one
        if loadDir is not None:
            self.load_from_disk(loadDir)
        else:
            # Build neural circuit from scratch
            with Context("Circuit") as self.circuit:
                # Input layer: Poisson spike encoder
                self.z0 = PoissonCell("z0", n_units=in_dim, max_freq=100., key=subkeys[0])
                
                # Input-to-excitatory synapses with STDP learning
                self.W1 = TraceSTDPSynapse("W1", shape=(in_dim, hid_dim),
                                           A_plus=Aplus, A_minus=Aminus, eta=1.,
                                           pretrace_target=0.,
                                           weight_init=dist.uniform(0.0, 0.3),
                                           key=subkeys[1])
                
                # Excitatory neuron layer - either coupled or standard LIF
                if use_coupled:
                    print("Using coupled with use_coupled", use_coupled, " - use_movingAvg", use_movingAvg, " - lam, eta", lam, eta)
                    self.z1e = CoupledLIFCell("z1e", n_units=hid_dim, tau_m=tau_m_e,
                                       resist_m=tau_m_e/dt, thr=-52., v_rest=-65.,
                                       v_reset=-60., tau_theta=1e7, theta_plus=0.05,
                                       refract_time=5., one_spike=True, key=subkeys[2],
                                       use_coupled=use_coupled,use_movingAvg=use_movingAvg,lam=lam,eta=eta)
                else:
                    print("not using coupled")
                    self.z1e = LIFCell("z1e", n_units=hid_dim, tau_m=tau_m_e,
                                       resist_m=tau_m_e/dt, thr=-52., v_rest=-65.,
                                       v_reset=-60., tau_theta=1e7, theta_plus=0.05,
                                       refract_time=5., one_spike=True, key=subkeys[2])

                # Inhibitory neuron layer (1/4 the size of excitatory layer)
                self.z1i = LIFCell("z1i", n_units=hid_dim//4, tau_m=tau_m_i,
                                   resist_m=tau_m_i/dt, thr=-40., v_rest=-60.,
                                   v_reset=-45., tau_theta=0., refract_time=5.,
                                   one_spike=False, key=subkeys[3])

                # Static synapses for excitatory-inhibitory interactions
                # Inhibitory-to-excitatory connections (lateral inhibition)
                self.W1ie = StaticSynapse("W1ie", shape=(hid_dim//4, hid_dim),
                                          weight_init=dist.constant(-120.),
                                          key=subkeys[4])
                
                # Excitatory-to-inhibitory connections
                self.W1ei = StaticSynapse("W1ei", shape=(hid_dim, hid_dim//4),
                                          weight_init=dist.constant(22.5),
                                          key=subkeys[5])

                # Recurrent excitatory connections with STDP learning
                self.W1re = TraceSTDPSynapse("W1re", shape=(hid_dim, hid_dim),
                             A_plus=Aplus, A_minus=Aminus, eta=1.0,
                             pretrace_target=0.0,
                             weight_init=dist.uniform(0.0, 0.3),
                             key=subkeys[8])

                # Trace parameters for STDP learning
                a_delta_pre = 0.1   # Pre-synaptic trace increment
                a_delta_post = 0.05 # Post-synaptic trace increment
                print("a_pre: ", a_delta_pre)
                print("a_post: ", a_delta_post)
                
                # Synaptic traces for input-to-excitatory connections (W1)
                self.tr0 = VarTrace("tr0", n_units=in_dim, tau_tr=tau_tr, decay_type="exp",
                                    a_delta=a_delta_pre, key=subkeys[6])
                self.tr1 = VarTrace("tr1", n_units=hid_dim, tau_tr=tau_tr, decay_type="exp",
                                    a_delta=a_delta_post, key=subkeys[7])

                # Additional traces for recurrent connections (W1re)
                self.tr_re_pre = VarTrace(
                    "tr_re_pre",
                    n_units=hid_dim,
                    tau_tr=tau_tr,
                    decay_type="exp",
                    a_delta=a_delta_pre,
                    key=subkeys[9]
                )

                self.tr_re_post = VarTrace(
                    "tr_re_post",
                    n_units=hid_dim,
                    tau_tr=tau_tr,
                    decay_type="exp",
                    a_delta=a_delta_post,
                    key=subkeys[10]
                )

                # Connect synaptic inputs to their respective sources
                self.W1.inputs << self.z0.outputs      # Input spikes to W1
                self.W1ie.inputs << self.z1i.s         # Inhibitory spikes to W1ie
                self.W1re.inputs << self.z1e.s         # Excitatory spikes to recurrent W1re

                # Connect synaptic outputs to neuronal current inputs
                if use_coupled:
                    # For coupled neurons, the recurrence is handled internally
                    self.z1e.j << summation(self.W1.outputs, self.W1ie.outputs)
                else:
                    # For standard neurons, include all connections
                    self.z1e.j << summation(self.W1.outputs, self.W1ie.outputs, self.W1re.outputs)

                # Connect excitatory-to-inhibitory pathway
                self.W1ei.inputs << self.z1e.s         # Excitatory spikes to W1ei
                self.z1i.j << self.W1ei.outputs        # W1ei output to inhibitory neurons

                # Connect traces to their respective spike sources
                self.tr0.inputs << self.z0.outputs     # Input layer traces
                self.tr1.inputs << self.z1e.s          # Excitatory layer traces
                self.tr_re_pre.inputs << self.z1e.s    # Recurrent pre-synaptic traces
                self.tr_re_post.inputs << self.z1e.s   # Recurrent post-synaptic traces

                # Connect STDP learning signals for input-to-excitatory synapses
                self.W1.preTrace << self.tr0.trace     # Pre-synaptic trace
                self.W1.preSpike << self.z0.outputs    # Pre-synaptic spikes
                self.W1.postTrace << self.tr1.trace    # Post-synaptic trace
                self.W1.postSpike << self.z1e.s        # Post-synaptic spikes

                # Connect STDP learning signals for recurrent synapses
                self.W1re.preTrace << self.tr_re_pre.trace   # Recurrent pre-synaptic trace
                self.W1re.preSpike << self.z1e.s             # Recurrent pre-synaptic spikes
                self.W1re.postTrace << self.tr_re_post.trace # Recurrent post-synaptic trace
                self.W1re.postSpike << self.z1e.s            # Recurrent post-synaptic spikes

                # Compile circuit operations for efficient execution
                reset_cmd, reset_args = self.circuit.compile_by_key(
                                            self.z0, self.z1e, self.z1i,
                                            self.tr0, self.tr1,
                                            self.W1, self.W1re,self.W1ie, self.W1ei,
                                            compile_key="reset")

                advance_cmd, advance_args = self.circuit.compile_by_key(
                                                self.W1, self.W1re,self.W1ie, self.W1ei,
                                                self.z0, self.z1e, self.z1i,
                                                self.tr0, self.tr1,
                                                compile_key="advance_state")
                
                evolve_cmd, evolve_args = self.circuit.compile_by_key(self.W1, self.W1re, compile_key="evolve")

                # Initialize dynamic commands
                self.dynamic()

    def dynamic(self):
        """
        Sets up dynamic commands for circuit operation including weight normalization,
        input clamping, and simulation processing.
        """
        # Get references to key components for dynamic operations
        W1, W1re, z0, z1e = self.circuit.get_components("W1", "W1re", "z0", "z1e")
        self.W1 = W1
        self.W1re = W1re
        self.z0 = z0
        self.z1e = z1e

        # Add JIT-compiled reset command for performance
        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")

        @Context.dynamicCommand
        def norm():
            """Normalize synaptic weights to maintain stability"""
            W1.weights.set(normalize_matrix(W1.weights.value, self.wNorm, order=1, axis=0))
            W1re.weights.set(normalize_matrix(W1re.weights.value, self.wNorm, order=1, axis=0))

        @Context.dynamicCommand
        def clamp(x):
            """Set input stimulus for the network"""
            z0.inputs.set(x)

        @scanner
        def process(compartment_values, args):
            """
            Single simulation step: advance neural states and evolve synapses
            
            Args:
                compartment_values: Current state of all neural compartments
                args: Time step arguments (current time, dt)
            
            Returns:
                Updated compartment values and excitatory spike outputs
            """
            _t, _dt = args
            # Advance neural dynamics for one time step
            compartment_values = self.circuit.advance_state(compartment_values, t=_t, dt=_dt, recurrent_weights=self.W1re.weights.value)
            # Update synaptic weights based on STDP rules
            compartment_values = self.circuit.evolve(compartment_values, t=_t, dt=_dt)
            return compartment_values, compartment_values[self.z1e.s.path]

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only == True:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.z1e.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, self.model_name)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        with Context("Circuit") as circuit:
            self.circuit = circuit
            self.circuit.load_from_dir(model_directory)
            # Reinitialize dynamic commands after loading
            self.dynamic()

    def get_synapse_stats(self):
        """
        Print basic statistics of W1 to string

        Returns:
            string containing min, max, mean, and L2 norm of W1
        """
        _W1 = self.W1.weights.value
        msg = "W1:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(_W1),
                                                                    jnp.amax(_W1),
                                                                    jnp.mean(_W1),
                                                                    jnp.linalg.norm(_W1))
        return msg


    def process(self, obs, adapt_synapses=True, collect_spike_train=False):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T. The observed pattern is converted to a Poisson 
        spike train with maximum frequency of 100 Hz.

        Note: This model assumes batch sizes of one (online learning).

        Args:
            obs: observed pattern to have spiking model process
            adapt_synapses: if True, synaptic efficacies will be adapted via STDP
            collect_spike_train: if True, stores T-length array of spike vectors

        Returns:
            Array containing spike vectors over time (empty if collect_spike_train is False)
        """
        batch_dim = obs.shape[0]
        assert batch_dim == 1  # Enforce single-sample processing

        # Reset circuit state and set input stimulus
        self.circuit.reset()
        self.circuit.clamp(obs)
        
        # Run simulation for T time steps
        out = self.circuit.process(jnp.array([[self.dt*i, self.dt] for i in range(self.T)]))
        
        # Apply weight normalizatrix(self.W1.weights.value, 78.4, order=1, axis=0))
        return out
