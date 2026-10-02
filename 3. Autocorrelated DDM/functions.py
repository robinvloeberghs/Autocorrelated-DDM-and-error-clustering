# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:59:13 2025

@author: Robin Vloeberghs
"""

import numpy as np
import logging


class ConfigurationError(Exception):
    """Class for error in model configuration, e.g. in meta dict"""

    pass

# From https://bayesflow.org/_modules/bayesflow/default_settings.html
DEFAULT_KEYS = {
    "prior_draws": "prior_draws",
    "obs_data": "obs_data",
    "sim_data": "sim_data",
    "batchable_context": "batchable_context",
    "non_batchable_context": "non_batchable_context",
    "prior_batchable_context": "prior_batchable_context",
    "prior_non_batchable_context": "prior_non_batchable_context",
    "prior_context": "prior_context",
    "hyper_prior_draws": "hyper_prior_draws",
    "shared_prior_draws": "shared_prior_draws",
    "local_prior_draws": "local_prior_draws",
    "sim_batchable_context": "sim_batchable_context",
    "sim_non_batchable_context": "sim_non_batchable_context",
    "summary_conditions": "summary_conditions",
    "direct_conditions": "direct_conditions",
    "parameters": "parameters",
    "hyper_parameters": "hyper_parameters",
    "shared_parameters": "shared_parameters",
    "local_parameters": "local_parameters",
    "observables": "observables",
    "targets": "targets",
    "conditions": "conditions",
    "posterior_inputs": "posterior_inputs",
    "likelihood_inputs": "likelihood_inputs",
    "model_outputs": "model_outputs",
    "model_indices": "model_indices",
}


# Custom function that enables local parameters to be constrained by shared parameters
# Original code: https://bayesflow.org/_modules/bayesflow/simulation.html#TwoLevelPrior
class _CustomTwoLevelPrior:
    """Basic interface for a simulation module responsible for generating random draws from a
    two-level prior distribution.

    The prior functions should return a np.array of simulation parameters which will be internally used
    by the TwoLevelGenerativeModel interface for simulations.

    An optional context generator (i.e., an instance of ContextGenerator) or a user-defined callable object
    implementing the following two methods can be provided:
    - ``context_generator.batchable_context(batch_size)``
    - ``context_generator.non_batchable_context()``
    """



    def __init__(
        self,
        hyper_prior_fun: callable,
        local_prior_fun: callable,
        shared_prior_fun: callable = None,
        local_context_generator: callable = None,
    ):
        """
        Instantiates a prior generator which will draw random parameter configurations from a joint prior
        having the general form:
    
        ``p(local | hyper) p(hyper) p(shared)``
    
        Such priors are often encountered in two-level hierarchical Bayesian models and allow for modeling
        nested data.
        No improper priors are allowed, as these may render the generative scope of a model undefined.
    
        Parameters
        ----------
        hyper_prior_fun         : callable
            A function (callbale object) which generates random draws from a hyperprior (unconditional)
        local_prior_fun         : callable
            A function (callable object) which generates random draws from a conditional prior
            given hyperparameters sampled from the hyperprior and optional context (e.g., variable number of groups)
        shared_prior_fun        : callable or None, optional, default: None
            A function (callable object) which generates random draws from an uncondtional prior.
            Represents optional shared parameters.
        local_context_generator : callable or None, optional, default: None
            An optional function (ideally an instance of ``ContextGenerator``) for generating control variables
            for the local_prior_fun.
    
        Examples
        --------
        Varying number of local factors (e.g., groups, participants) between 1 and 100:
    
            def draw_hyper():
                # Draw location for 2D conditional prior
                return np.random.normal(size=2)
    
            def draw_prior(means, num_groups, sigma=1.):
                # Draw parameter given location from hyperprior
                dim = means.shape[0]
                return np.random.normal(means, sigma, size=(num_groups, dim))
    
            context = ContextGenerator(non_batchable_context_fun=lambda : np.random.randint(1, 101))
            prior = TwoLevelPrior(draw_hyper, draw_prior, local_context_generator=context)
            prior_dict = prior(batch_size=32)
    
        """
    
        self.hyper_prior = hyper_prior_fun
        self.local_prior = local_prior_fun
        self.shared_prior = shared_prior_fun
        self.local_context_generator = local_context_generator
    
    
    
    def __call__(self, batch_size, **kwargs):
        """Generates ``batch_size`` draws from the hierarchical prior."""
        
        out_dict = {
            DEFAULT_KEYS["hyper_parameters"]: [None] * batch_size,
            DEFAULT_KEYS["local_parameters"]: [None] * batch_size,
        }
        if self.shared_prior is not None:
            out_dict[DEFAULT_KEYS["shared_parameters"]] = [None] * batch_size
        if self.local_context_generator is not None:
            local_context = self.local_context_generator(batch_size)
        else:
            local_context = {}
    
        for b in range(batch_size):
            # Draw hyper parameters
            hyper_params = self.draw_hyper_parameters(**kwargs.get("hyper_args", {}))
    
            # Take care of shared prior
            # MOVED THIS BEFORE THE LOCAL PARAMETERS
            if self.shared_prior is not None:
                shared_params = self.draw_shared_parameters(**kwargs.get("shared_args", {}))
                out_dict[DEFAULT_KEYS["shared_parameters"]][b] = shared_params
            
            # Determine context types for local parameters
            if local_context.get(DEFAULT_KEYS["batchable_context"]) is not None:
                local_batchable_context = local_context[DEFAULT_KEYS["batchable_context"]][b]
            else:
                local_batchable_context = None
            local_non_batchable_context = local_context.get(DEFAULT_KEYS["non_batchable_context"])
    
            # Draw local parameters
            # ADDED SHARED PARAMS AS ARGUMENT
            local_params = self.draw_local_parameters(
                shared_params, hyper_params, local_batchable_context, local_non_batchable_context, **kwargs.get("local_args", {})
            )
    
            out_dict[DEFAULT_KEYS["hyper_parameters"]][b] = hyper_params
            out_dict[DEFAULT_KEYS["local_parameters"]][b] = local_params
    
    
    
        # Array conversion must work or fail gently
        out_dict[DEFAULT_KEYS["hyper_parameters"]] = np.array(out_dict[DEFAULT_KEYS["hyper_parameters"]])
        out_dict[DEFAULT_KEYS["local_parameters"]] = np.array(out_dict[DEFAULT_KEYS["local_parameters"]])
        if self.shared_prior is not None:
            out_dict[DEFAULT_KEYS["shared_parameters"]] = np.array(out_dict[DEFAULT_KEYS["shared_parameters"]])
    
        # Add optional context entries
        out_dict[DEFAULT_KEYS["batchable_context"]] = local_context.get(DEFAULT_KEYS["batchable_context"])
        out_dict[DEFAULT_KEYS["non_batchable_context"]] = local_context.get(DEFAULT_KEYS["non_batchable_context"])
    
        return out_dict
    
    
    def draw_hyper_parameters(self, **kwargs):
        """TODO"""
    
        params = self.hyper_prior(**kwargs)
        return params
    
    
    # ADDED SHARED TO LOCAL PARAMETERS
    def draw_local_parameters(self, shared, hypers, batchable_context=None, non_batchable_context=None, **kwargs):
        """TODO"""
    
        # Case no context
        if batchable_context is None and non_batchable_context is None:
            return self.local_prior(shared, hypers, **kwargs)
        # Case only batchable context
        elif batchable_context is not None and non_batchable_context is None:
            return self.local_prior(shared, hypers, batchable_context, **kwargs)
        # Case only non batchable context
        elif batchable_context is None and non_batchable_context is not None:
            return self.local_prior(shared, hypers, non_batchable_context, **kwargs)
        # Case both context types present
        else:
            return self.local_prior(shared, hypers, batchable_context, non_batchable_context, **kwargs)
    
    
    def draw_shared_parameters(self, **kwargs):
        """TODO"""
    
        if self.shared_prior is None:
            raise Exception("No shared_prior_fun provided during initialization!")
        params = self.shared_prior(**kwargs)
        return params

    
class _Custom_Simulator:
    """Basic interface for a simulation module responsible for generating randomized simulations given a prior
    parameter distribution and optional context variables, given a user-provided simulation function.

    The user-provided simulator functions should return a np.array of synthetic data which will be used internally
    by the GenerativeModel interface for simulations.

    An optional context generator (i.e., an instance of ContextGenerator) or a user-defined callable object
    implementing the following two methods can be provided:
    - ``context_generator.batchable_context(batch_size)``
    - ``context_generator.non_batchable_context()``
    """


    def __init__(self, batch_simulator_fun=None, simulator_fun=None, context_generator=None):
        """Instantiates a data generator which will perform randomized simulations given a set of parameters and optional context.
        Either a ``batch_simulator_fun`` or ``simulator_fun``, but not both, should be provided to instantiate a ``Simulator`` object.

        If a ``batch_simulator_fun`` is provided, the interface will assume that the function operates on batches of parameter
        vectors and context variables and will pass the latter directly to the function. Power users should attempt to provide
        optimized batched simulators.

        If a ``simulator_fun`` is provided, the interface will assume that the function operates on single parameter vectors and
        context variables and will wrap the simulator internally to allow batched functionality.

        Parameters
        ----------
        batch_simulator_fun  : callable
            A function (callbale object) with optional control arguments responsible for generating a batch of simulations
            given a batch of parameters and optional context variables.
        simulator_fun       : callable
            A function (callable object) with optional control arguments responsible for generating a simulaiton given
            a single parameter vector and optional variables.
        context_generator   : callable (default None, recommended instance of ContextGenerator)
            An optional function (ideally an instance of ``ContextGenerator``) for generating prior context variables.
        """

        if (batch_simulator_fun is None) is (simulator_fun is None):
            raise ConfigurationError("Either batch_simulator_fun or simulator_fun should be provided, but not both!")

        self.is_batched = True if batch_simulator_fun is not None else False

        if self.is_batched:
            self.simulator = batch_simulator_fun
        else:
            self.simulator = simulator_fun
        self.context_gen = context_generator





    def __call__(self, params, condition_matrix, *args, **kwargs):
        """Generates simulated data given param draws and optional context variables generated internally.

        Parameters
        ----------
        params   :  np.ndarray of shape (n_sim, ...) - the parameter draws obtained from the prior.

        Returns
        -------

        out_dict : dictionary
            An output dictionary with randomly simulated variables, the following keys are mandatory, if default keys not modified:
            ``sim_data`` : value
            ``non_batchable_context`` : value
            ``batchable_context`` : value
        """

        # Always assume first dimension is batch dimension
        # Handle cases with multiple inputs to simulator
        if isinstance(params, tuple) or isinstance(params, list):
            batch_size = params[0].shape[0]
        # Handle all other cases or fail gently
        else:
            batch_size = params.shape[0]

        # Prepare placeholder dictionary
        out_dict = {
            DEFAULT_KEYS["sim_data"]: None,
            DEFAULT_KEYS["batchable_context"]: None,
            DEFAULT_KEYS["non_batchable_context"]: None,
        }

        # Populate dictionary with context or leave at None
        
        # MAKES THE SIMULATOR RECOGNIZE THAT THERE IS BATCHABLE_CONTEXT, condition_matrix (from prior) is given by _CustomTwoLevelGenerativeModel
        out_dict[DEFAULT_KEYS["batchable_context"]] = condition_matrix
        
        if self.context_gen is not None:
            context_dict = self.context_gen.generate_context(batch_size, *args, **kwargs)
            out_dict[DEFAULT_KEYS["non_batchable_context"]] = context_dict[DEFAULT_KEYS["non_batchable_context"]]
            out_dict[DEFAULT_KEYS["batchable_context"]] = context_dict[DEFAULT_KEYS["batchable_context"]]

        if self.is_batched:
            return self._simulate_batched(params, out_dict, *args, **kwargs)
        return self._simulate_non_batched(params, out_dict, *args, **kwargs)



    def _simulate_batched(self, params, out_dict, *args, **kwargs):
        """Assumes a batched simulator accepting batched contexts and priors."""

        # No context type
        if (
            out_dict[DEFAULT_KEYS["batchable_context"]] is None
            and out_dict[DEFAULT_KEYS["non_batchable_context"]] is None
        ):
            out_dict[DEFAULT_KEYS["sim_data"]] = self.simulator(params, *args, **kwargs)

        # Only batchable context
        elif out_dict["non_batchable_context"] is None:
            out_dict[DEFAULT_KEYS["sim_data"]] = self.simulator(
                params, out_dict[DEFAULT_KEYS["batchable_context"]], *args, **kwargs
            )

        # Only non-batchable context
        elif out_dict[DEFAULT_KEYS["batchable_context"]] is None:
            out_dict[DEFAULT_KEYS["sim_data"]] = self.simulator(
                params, out_dict[DEFAULT_KEYS["non_batchable_context"]], *args, **kwargs
            )

        # Both batchable and non-batchable context
        else:
            out_dict[DEFAULT_KEYS["sim_data"]] = self.simulator(
                params,
                out_dict[DEFAULT_KEYS["batchable_context"]],
                out_dict[DEFAULT_KEYS["non_batchable_context"]],
                *args,
                **kwargs,
            )

        return out_dict

    def _simulate_non_batched(self, params, out_dict, *args, **kwargs):
        """Assumes a non-batched simulator accepting batched contexts and priors."""

        # Extract batch size
        # Always assume first dimension is batch dimension
        # Handle cases with multiple inputs to simulator
        if isinstance(params, tuple) or isinstance(params, list):
            batch_size = params[0].shape[0]
            non_batched_params = [[params[i][b] for i in range(len(params))] for b in range(batch_size)]
        # Handle all other cases or fail gently
        else:
            # expand dimension by one to handle both cases in the same way
            batch_size = params.shape[0]
            non_batched_params = params

        # No context type
        if (
            out_dict[DEFAULT_KEYS["batchable_context"]] is None
            and out_dict[DEFAULT_KEYS["non_batchable_context"]] is None
        ):
            out_dict[DEFAULT_KEYS["sim_data"]] = np.array(
                [self.simulator(non_batched_params[b], *args, **kwargs) for b in range(batch_size)]
            )

        # Only batchable context
        elif out_dict["non_batchable_context"] is None:
            out_dict[DEFAULT_KEYS["sim_data"]] = np.array(
                [
                    self.simulator(
                        non_batched_params[b], out_dict[DEFAULT_KEYS["batchable_context"]][b], *args, **kwargs
                    )
                    for b in range(batch_size)
                ]
            )

        # Only non-batchable context
        elif out_dict[DEFAULT_KEYS["batchable_context"]] is None:
            out_dict[DEFAULT_KEYS["sim_data"]] = np.array(
                [
                    self.simulator(
                        non_batched_params[b], out_dict[DEFAULT_KEYS["non_batchable_context"]], *args, **kwargs
                    )
                    for b in range(batch_size)
                ]
            )

        # Both batchable and non_batchable context
        else:
            out_dict[DEFAULT_KEYS["sim_data"]] = np.array(
                [
                    self.simulator(
                        non_batched_params[b],
                        out_dict[DEFAULT_KEYS["batchable_context"]][b],
                        out_dict[DEFAULT_KEYS["non_batchable_context"]],
                        *args,
                        **kwargs,
                    )
                    for b in range(batch_size)
                ]
            )

        return out_dict



    
class _CustomTwoLevelGenerativeModel:
    """Basic interface for a generative model in a simulation-based context.

    Generally, a generative model consists of two mandatory components:
    - MultilevelPrior : A randomized function returning random parameter draws from a two-level prior distribution;
    - Simulator : A function which transforms the parameters into observables in a non-deterministic manner.
    """

    _N_SIM_TEST = 2



    def __init__(
        self,
        prior: callable,
        simulator: callable,
        skip_test: bool = False,
        simulator_is_batched: bool = None,
        name: str = "anonymous",
    ):
        """Instantiates a generative model responsible for generating parameters, data, and optional context.

        Parameters
        ----------
        prior                : callable
            A function returning random draws from the two-level prior parameter distribution. Should encode
            prior knowledge about plausible parameter ranges
        simulator            : callable or bayesflow.simulation.Simulator
            A function accepting parameter draws, shared parameters, optional context, and optional arguments as input
            and returning observable data
        skip_test            : bool, optional, default: False
            If True, a forward inference pass will be performed.
        simulator_is_batched : bool or None, optional, default: None
            Only relevant and mandatory if providing a custom simulator without the ``Simulator`` wrapper.
        name                 : str (default - "anonymous")
            An optional name for the generative model.

        Notes
        -----
        If you are not using the provided ``TwoLevelPrior`` and ``Simulator`` wrappers for your prior and data
        generator, only functions returning a ``np.ndarray`` in the correct format will be accepted, since these will be
        wrapped internally. In addition, you need to indicate whether your simulator operates on batched of
        parameters or on single parameter vectors via tha `simulator_is_batched` argument.
        """

        self.prior = prior
        self.simulator = simulator
        self.simulator_is_batched = self.simulator.is_batched

        if name is None:
            self.name = "anonymous"
        else:
            self.name = name

        if not skip_test:
            self._test()





    def __call__(self, batch_size, **kwargs):
        """Carries out forward inference ``batch_size`` times."""

        # Draw from prior batch_size times
        prior_out = self.prior(batch_size, **kwargs.pop("prior_args", {}))

        # Case no shared parameters - first input to simulator
        # is just the array of local prior draws
        if prior_out.get(DEFAULT_KEYS["shared_parameters"]) is None:
            sim_out = self.simulator(prior_out[DEFAULT_KEYS["local_parameters"]], **kwargs.pop("sim_args", {}))
        # Case shared parameters - first input to simulator
        # is a tuple (local_parameters, shared_parameters)
        # ALSO ADDED BATCHABLE CONTEXT AS INPUT FOR SIMULATOR
        else:
            sim_out = self.simulator(
                (prior_out[DEFAULT_KEYS["local_parameters"]], prior_out[DEFAULT_KEYS["shared_parameters"]]),
                 prior_out[DEFAULT_KEYS["batchable_context"]],
                 **kwargs.pop("sim_args", {}),
            )

        # Prepare and fill placeholder dict, starting from prior dict
        out_dict = {
            DEFAULT_KEYS["sim_data"]: sim_out[DEFAULT_KEYS["sim_data"]],
            DEFAULT_KEYS["hyper_prior_draws"]: prior_out[DEFAULT_KEYS["hyper_parameters"]],
            DEFAULT_KEYS["local_prior_draws"]: prior_out[DEFAULT_KEYS["local_parameters"]],
            DEFAULT_KEYS["shared_prior_draws"]: prior_out.get(DEFAULT_KEYS["shared_parameters"]),
            DEFAULT_KEYS["sim_batchable_context"]: sim_out.get(DEFAULT_KEYS["batchable_context"]),
            DEFAULT_KEYS["sim_non_batchable_context"]: sim_out.get(DEFAULT_KEYS["non_batchable_context"]),
            DEFAULT_KEYS["prior_batchable_context"]: prior_out.get(DEFAULT_KEYS["batchable_context"]),
            DEFAULT_KEYS["prior_non_batchable_context"]: prior_out.get(DEFAULT_KEYS["non_batchable_context"]),
        }

        return out_dict



    def _config_custom_simulator(self, sim_fun, is_batched):
        """Only called if user has provided a custom simulator not using the ``Simulator`` wrapper."""

        if is_batched is None:
            raise ConfigurationError(
                "Since you are not using the Simulator wrapper, please set "
                + "simulator_is_batched to True if your simulator operates on batches of parameters, "
                + "otherwise set it to False."
            )
        elif is_batched:
            return Simulator(batch_simulator_fun=sim_fun)
        else:
            return Simulator(simulator_fun=sim_fun)

    def _test(self):
        """Performs a sanity check on forward inference and some verbose information."""

        # Use minimal n_sim > 1
        _n_sim = _CustomTwoLevelGenerativeModel._N_SIM_TEST
        out = self(_n_sim)
        # Logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Attempt to log batch results or fail and warn user
        try:
            logger.info(f"Performing {_n_sim} pilot runs with the {self.name} model...")
            # Format strings
            p_shape_str = "(batch_size = {}, -{}".format(
                out[DEFAULT_KEYS["local_prior_draws"]].shape[0], out[DEFAULT_KEYS["local_prior_draws"]].shape[1:]
            )
            p_shape_str = p_shape_str.replace("-(", "").replace(",)", ")")
            d_shape_str = "(batch_size = {}, -{}".format(
                out[DEFAULT_KEYS["sim_data"]].shape[0], out[DEFAULT_KEYS["sim_data"]].shape[1:]
            )
            d_shape_str = d_shape_str.replace("-(", "").replace(",)", ")")

            # Log to default-config
            logger.info(f"Shape of parameter batch after {_n_sim} pilot simulations: {p_shape_str}")
            logger.info(f"Shape of simulation batch after {_n_sim} pilot simulations: {d_shape_str}")

            for k, v in out.items():
                if k.endswith("_prior_draws"):
                    if v is None:
                        logger.info(f"No {k} provided.")
                    else:
                        p_shape_str = "(batch_size = {}, -{}".format(v.shape[0], v.shape[1:])
                        p_shape_str = p_shape_str.replace("-(", "").replace(",)", ")")
                        logger.info(f"Shape of {k} batch after {_n_sim} pilot simulations: {p_shape_str}")
                if "context" in k:
                    name = k.replace("_", " ").replace("sim", "simulation").replace("non ", "non-")
                    if v is None:
                        logger.info(f"No optional {name} provided.")
                    else:
                        try:
                            logger.info(f"Shape of {name}: {v.shape}")
                        except Exception as _:
                            logger.info(
                                f"Could not determine shape of {name}. Type appears to be non-array: {type(v)},\
                                    so make sure your input configurator takes care of that!"
                            )
        except Exception as err:
            raise ConfigurationError(
                "Could not run forward inference with specified generative model..."
                + f"Please re-examine model components!\n {err}"
            )


