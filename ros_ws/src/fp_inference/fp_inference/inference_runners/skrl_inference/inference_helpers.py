# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import gymnasium as gym
import numpy as np
from typing import Any

from skrl import logger
from skrl.models.torch import Model

from skrl import logger
from skrl.models.torch import Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise  # noqa
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa


def _component(name: str) -> type:
    """Get skrl component (e.g.: agent, trainer, etc..) from string identifier

    :return: skrl component
    """
    component = None
    name = name.lower()
    # model
    if name == "gaussianmixin":
        from skrl.utils.model_instantiators.torch import gaussian_model as component
    elif name == "categoricalmixin":
        from skrl.utils.model_instantiators.torch import categorical_model as component
    elif name == "deterministicmixin":
        from skrl.utils.model_instantiators.torch import deterministic_model as component
    elif name == "multivariategaussianmixin":
        from skrl.utils.model_instantiators.torch import multivariate_gaussian_model as component
    elif name == "shared":
        from skrl.utils.model_instantiators.torch import shared_model as component
    # memory
    elif name == "randommemory":
        from skrl.memories.torch import RandomMemory as component
    # agent
    elif name in ["a2c", "a2c_default_config"]:
        from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG

        component = A2C_DEFAULT_CONFIG if "default_config" in name else A2C
    elif name in ["amp", "amp_default_config"]:
        from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG

        component = AMP_DEFAULT_CONFIG if "default_config" in name else AMP
    elif name in ["cem", "cem_default_config"]:
        from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG

        component = CEM_DEFAULT_CONFIG if "default_config" in name else CEM
    elif name in ["ddpg", "ddpg_default_config"]:
        from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG

        component = DDPG_DEFAULT_CONFIG if "default_config" in name else DDPG
    elif name in ["ddqn", "ddqn_default_config"]:
        from skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG

        component = DDQN_DEFAULT_CONFIG if "default_config" in name else DDQN
    elif name in ["dqn", "dqn_default_config"]:
        from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG

        component = DQN_DEFAULT_CONFIG if "default_config" in name else DQN
    elif name in ["ppo", "ppo_default_config"]:
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

        component = PPO_DEFAULT_CONFIG if "default_config" in name else PPO
    elif name in ["rpo", "rpo_default_config"]:
        from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG

        component = RPO_DEFAULT_CONFIG if "default_config" in name else RPO
    elif name in ["sac", "sac_default_config"]:
        from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG

        component = SAC_DEFAULT_CONFIG if "default_config" in name else SAC
    elif name in ["td3", "td3_default_config"]:
        from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG

        component = TD3_DEFAULT_CONFIG if "default_config" in name else TD3
    elif name in ["trpo", "trpo_default_config"]:
        from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG

        component = TRPO_DEFAULT_CONFIG if "default_config" in name else TRPO
    # multi-agent
    elif name in ["ippo", "ippo_default_config"]:
        from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG

        component = IPPO_DEFAULT_CONFIG if "default_config" in name else IPPO
    elif name in ["mappo", "mappo_default_config"]:
        from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG

        component = MAPPO_DEFAULT_CONFIG if "default_config" in name else MAPPO
    # trainer
    elif name == "sequentialtrainer":
        from skrl.trainers.torch import SequentialTrainer as component

    if component is None:
        raise ValueError(f"Unknown component '{name}' in runner cfg")
    return component


def load_cfg_from_yaml(path: str) -> dict:
    """Load a runner configuration from a yaml file

    :param path: File path

    :return: Loaded configuration, or an empty dict if an error has occurred
    """
    try:
        import yaml
    except Exception as e:
        logger.error(f"{e}. Install PyYAML with 'pip install pyyaml'")
        return {}

    try:
        with open(path) as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Loading yaml error: {e}")
        return {}


def process_cfg(cfg: dict) -> dict:
    """Convert simple types to skrl classes/components

    :param cfg: A configuration dictionary

    :return: Updated dictionary
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "shared_state_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
    ]

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    if type(d[key]) is str:
                        d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
        return d

    return update_dict(copy.deepcopy(cfg))


def generate_models(env: dict[str, Any], cfg: dict[str, Any], action_space: gym.spaces.Space, device: str) -> dict[str, Model]:
    """Generate model instances according to the environment specification and the given config

    :param env: Wrapped environment
    :param cfg: A configuration dictionary

    :return: Model instances
    """

    # Consider only single agent envs.
    print(env)
    observation_spaces = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env["observation_space"],), dtype=np.float32)
    action_spaces = action_space

    agent_class = cfg.get("agent", {}).get("class", "").lower()

    # instantiate models
    _cfg = copy.deepcopy(cfg)
    models_cfg = _cfg.get("models")
    models = {}
    if not models_cfg:
        raise ValueError("No 'models' are defined in cfg")
    # get separate (non-shared) configuration and remove 'separate' key
    try:
        separate = models_cfg["separate"]
        del models_cfg["separate"]
    except KeyError:
        separate = True
        logger.warning("No 'separate' field defined in 'models' cfg. Defining it as True by default")
    # non-shared models
    if separate:
        for role in models_cfg:
            # get instantiator function and remove 'class' key
            model_class = models_cfg[role].get("class")
            if not model_class:
                raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
            del models_cfg[role]["class"]
            model_class = _component(model_class)
            # get specific spaces according to agent/model cfg
            observation_space = observation_spaces
            if agent_class == "mappo" and role == "value":
                observation_space = state_spaces
            if agent_class == "amp" and role == "discriminator":
                try:
                    observation_space = env.amp_observation_space
                except Exception:
                    logger.warning(
                        "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                    )
            # print model source
            source = model_class(
                observation_space=observation_space,
                action_space=action_spaces,
                device=device,
                **process_cfg(models_cfg[role]),
                return_source=True,
            )
            print("==================================================")
            print(f"Model (role): {role}")
            print("==================================================\n")
            print(source)
            print("--------------------------------------------------")
            # instantiate model
            models[role] = model_class(
                observation_space=observation_space,
                action_space=action_spaces,
                device=device,
                **process_cfg(models_cfg[role]),
            )
    # shared models
    else:
        roles = list(models_cfg.keys())
        if len(roles) != 2:
            raise ValueError(
                "Runner currently only supports shared models, made up of exactly two models. "
                "Set 'separate' field to True to create non-shared models for the given cfg"
            )
        # get shared model structure and parameters
        structure = []
        parameters = []
        for role in roles:
            # get instantiator function and remove 'class' key
            model_structure = models_cfg[role].get("class")
            if not model_structure:
                raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
            del models_cfg[role]["class"]
            structure.append(model_structure)
            parameters.append(process_cfg(models_cfg[role]))
        model_class = _component("Shared")
        # print model source
        source = model_class(
            observation_space=observation_spaces,
            action_space=action_spaces,
            device=device,
            structure=structure,
            roles=roles,
            parameters=parameters,
            return_source=True,
        )
        print("==================================================")
        print(f"Shared model (roles): {roles}")
        print("==================================================\n")
        print(source)
        print("--------------------------------------------------")
        # instantiate model
        models[roles[0]] = model_class(
            observation_space=observation_spaces,
            action_space=action_spaces,
            device=device,
            structure=structure,
            roles=roles,
            parameters=parameters,
        )
        models[roles[1]] = models[roles[0]]

    # initialize lazy modules' parameters
    for role, model in models.items():
        model.init_state_dict(role)

    return models


def empty_preprocessor(x, *args, **kwargs):
    return x


def get_state_preprocessor(env_cfg: dict[str, Any], cfg: dict[str, Any], device: str):
    observation_spaces = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(env_cfg["observation_space"],), dtype=np.float32
    )

    agent_class = cfg.get("agent", {}).get("class", "").lower()
    if not agent_class:
        raise ValueError("No 'class' field defined in 'agent' cfg")

    if agent_class in ["amp"]:
        raise NotImplementedError(f"State preprocessor not implemented for agent class: {agent_class}")
    if agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3", "trpo"]:
        agent_cfg = _component(f"{agent_class}_DEFAULT_CONFIG").copy()
        agent_cfg.update(process_cfg(cfg["agent"]))
        agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": observation_spaces, "device": device})
    elif agent_class in ["ippo"] or agent_class in ["mappo"]:
        raise NotImplementedError(f"State preprocessor not implemented for agent class: {agent_class}")

    if agent_cfg["state_preprocessor"] is not None:
        state_preprocessor = agent_cfg["state_preprocessor"]
        state_preprocessor = state_preprocessor(**agent_cfg["state_preprocessor_kwargs"])
    else:
        state_preprocessor = empty_preprocessor

    return state_preprocessor
