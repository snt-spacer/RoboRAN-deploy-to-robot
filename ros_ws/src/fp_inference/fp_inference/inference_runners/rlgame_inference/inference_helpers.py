from rl_games.algos_torch import model_builder
import torch
import gym.spaces
import numpy as np
import copy
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch.players import rescale_actions


class BasicBasePlayer(object):
    def __init__(
        self, params, observation_space, action_space, clip_actions=False, deterministic=False, device_name="cuda"
    ):
        self.config = config = params["params"]["config"]
        self.load_networks(params["params"])
        self.clip_actions = clip_actions

        self.num_agents = 1
        self.value_size = 1
        self.action_space = action_space

        self.observation_space = observation_space
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get("central_value_config") is not None
        self.device_name = device_name
        self.is_deterministic = deterministic
        self.device = torch.device(self.device_name)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config["network"] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if "obs" in obs:
                obs = obs["obs"]
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert obs.dtype != np.int8
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def restore(self, fn):
        raise NotImplementedError("restore")

    def get_weights(self):
        weights = {}
        weights["model"] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights["model"])
        if self.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError("step")

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError("step")

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [
                torch.zeros((s.size()[0], self.batch_size, s.size()[2]), dtype=torch.float32).to(self.device)
                for s in rnn_states
            ]

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if "obs" in obses:
                obses = obses["obs"]
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if "observation" in obses:
                first_key = "observation"
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size


class BasicPpoPlayerContinuous(BasicBasePlayer):
    def __init__(
        self,
        params,
        observation_space,
        action_space,
        clip_actions=True,
        deterministic=False,
        device="cuda",
    ):
        BasicBasePlayer.__init__(
            self,
            params,
            observation_space,
            action_space,
            clip_actions,
            deterministic,
            device,
        )
        self.network = self.config["network"]
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input_keys = self.config.get("normalize_input_keys", [])
        self.normalize_input = self.config["normalize_input"]
        self.normalize_value = self.config.get("normalize_value", False)

        obs_shape = self.obs_shape
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.value_size,
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
            "normalize_input_keys": self.normalize_input_keys,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def restore(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def reset(self):
        self.init_rnn()


class BasicPpoPlayerDiscrete(BasicBasePlayer):
    def __init__(
        self,
        params,
        observation_space,
        action_space,
        clip_actions=True,
        deterministic=False,
        device="cuda",
    ):
        BasicBasePlayer.__init__(
            self,
            params,
            observation_space,
            action_space,
            clip_actions,
            deterministic,
            device,
        )

        self.network = self.config["network"]
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        self.mask = [False]
        self.normalize_input_keys = self.config.get("normalize_input_keys", [])
        self.normalize_input = self.config["normalize_input"]
        self.normalize_value = self.config.get("normalize_value", False)
        obs_shape = self.obs_shape
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.value_size,
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
            "normalize_input_keys": self.normalize_input_keys,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_masked_action(self, obs, action_masks, is_deterministic=True):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device).bool()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "action_masks": action_masks,
            "rnn_states": self.states,
        }
        self.model.eval()

        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict["logits"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits]
                return torch.stack(action, dim=-1)
            else:
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:
                return action.squeeze().detach()

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict["logits"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits]
                return torch.stack(action, dim=-1)
            else:
                return action.squeeze().detach()

        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:
                return action.squeeze().detach()

    def restore(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def reset(self):
        self.init_rnn()
