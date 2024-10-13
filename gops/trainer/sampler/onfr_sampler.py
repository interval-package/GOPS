from typing import List, NamedTuple

import numpy as np
import torch

from gops.trainer.sampler.base import BaseSampler, Experience

"""
The base sampler do not obtain the cost signal, so maybe we need to calc manually.

1. We gather the frt using the reverse process.
2. The fra are claced in the alg.
3. Constraint signal here denote as cost and can be get from the info.

For the implementation detail:

1. The target set is defined as the states which will give a positive reward.
2. The constraint violation is where the cost is negative.
3. The N is the shortest way to c or g.

"""

class OnfrSampler(BaseSampler):
    def __init__(
        self, 
        sample_batch_size,
        index=0, 
        noise_params=None,
        gae_gamma:float=None,
        fr_gamma:float=None,
        gae_lambda:float=None,
        **kwargs
    ):
        assert gae_gamma is not None # 0.99

        super().__init__(
            sample_batch_size,
            index, 
            noise_params,
            **kwargs
        )
        
        alg_name = kwargs["algorithm"]
        self.gae_gamma = gae_gamma
        if self._is_vector:
            self.obs_dim = self.env.single_observation_space.shape
            self.act_dim = self.env.single_action_space.shape
        else:
            self.obs_dim = self.env.observation_space.shape
            self.act_dim = self.env.action_space.shape

        self.mb_obs = np.zeros(
            (self.num_envs, self.horizon, *self.obs_dim), dtype=np.float32
        )
        self.mb_act = np.zeros(
            (self.num_envs, self.horizon, *self.act_dim), dtype=np.float32
        )
        self.mb_rew = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_cost = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_done = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_tlim = np.zeros((self.num_envs, self.horizon), dtype=np.bool_)
        self.mb_logp = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.need_value_flag = not (alg_name == "FHADP" or alg_name == "INFADP")
        if self.need_value_flag:
            assert gae_lambda is not None # 0.95
            assert fr_gamma  is not None
            self.gae_lambda = gae_lambda
            self.fr_gamma = fr_gamma
            self.mb_val = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_adv = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_ret = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
            self.mb_frt = np.zeros((self.num_envs, self.horizon), dtype=np.float32)
        self.mb_info = {}
        self.info_keys = kwargs["additional_info"].keys()
        for k, v in kwargs["additional_info"].items():
            self.mb_info[k] = np.zeros(
                (self.num_envs, self.horizon, *v["shape"]), dtype=v["dtype"]
            )
            self.mb_info["next_" + k] = np.zeros(
                (self.num_envs, self.horizon, *v["shape"]), dtype=v["dtype"]
            )

    def _sample(self) -> dict:
        self.ptr = np.zeros(self.num_envs, dtype=np.int32)
        self.last_ptr = np.zeros(self.num_envs, dtype=np.int32)
        for t in range(self.horizon):
            # batch_obs has shape (num_envs, obs_dim)
            if not self._is_vector:
                batch_obs = torch.from_numpy(
                    np.expand_dims(self.obs, axis=0).astype("float32")
                )
            else:
                batch_obs = torch.from_numpy(self.obs.astype("float32"))
            # interact with environment
            experiences = self._step()
            self._process_experiences(experiences, batch_obs, t)

        # wrap collected data into replay format
        mb_data = {
            "obs": torch.from_numpy(self.mb_obs.reshape(-1, *self.obs_dim)),
            "act": torch.from_numpy(self.mb_act.reshape(-1, *self.act_dim)),
            "rew": torch.from_numpy(self.mb_rew.reshape(-1)),
            "cost": torch.from_numpy(self.mb_cost.reshape(-1)),
            "done": torch.from_numpy(self.mb_done.reshape(-1)),
            "logp": torch.from_numpy(self.mb_logp.reshape(-1)),
            "time_limited": torch.from_numpy(self.mb_tlim.reshape(-1)),
        }
        if self.need_value_flag:
            mb_data.update({
                "ret": torch.from_numpy(self.mb_ret.reshape(-1)),   
                "adv": torch.from_numpy(self.mb_adv.reshape(-1)),
                "frt": torch.from_numpy(self.mb_frt.reshape(-1)),
            })
        for k, v in self.mb_info.items():
            mb_data[k] = torch.from_numpy(v.reshape(-1, *v.shape[2:]))
        return mb_data

    def sample_with_replay_format(self):
        return self.sample()

    def _process_experiences(
        self, 
        experiences: List[Experience],
        batch_obs: np.ndarray, 
        t: int
    ):
        if self.need_value_flag:
            value = self.networks.value(batch_obs).detach()
            self.mb_val[:, t] = value
        
        for i in np.arange(self.num_envs):
            (
                obs, 
                action, 
                reward, 
                done, 
                info, 
                next_obs, 
                next_info, 
                logp,
            ) = experiences[i]

            (
                self.mb_obs[i, t, ...],
                self.mb_act[i, t, ...],
                self.mb_rew[i, t],
                self.mb_cost[i, t],
                self.mb_done[i, t],
                self.mb_tlim[i, t],
                self.mb_logp[i, t],
            ) = (
                obs,
                action,
                reward,
                next_info["constraint"],
                done,
                next_info["TimeLimit.truncated"],
                logp,
            )

            for key in self.info_keys:
                self.mb_info[key][i, t] = info[key]
                self.mb_info["next_" + key][i, t] = next_info[key]

            # calculate value target (mb_ret) & gae (mb_adv)
            if (
                done
                or next_info["TimeLimit.truncated"]
                or t == self.horizon - 1
            ) and self.need_value_flag:
                last_obs_expand = torch.from_numpy(
                    np.expand_dims(next_obs, axis=0).astype("float32")
                )
                est_last_value = self.networks.value(
                    last_obs_expand
                ).detach().item() * (1 - done)
                est_last_fr = self.networks.frf(
                    last_obs_expand
                ).detach().item()
                self.ptr[i] = t
                self._finish_trajs(i, est_last_value, est_last_fr)
                self.last_ptr[i] = self.ptr[i]

    def _finish_trajs(self, env_index: int, est_last_val: float, est_last_fr: float):
        # calculate value target (mb_ret) & gae (mb_adv) whenever episode is finished
        path_slice = slice(self.last_ptr[env_index] + 1, self.ptr[env_index] + 1)
        value_preds_slice = np.append(self.mb_val[env_index, path_slice], est_last_val)
        rews_slice = self.mb_rew[env_index, path_slice]
        cost_slice = self.mb_cost[env_index, path_slice]
        length = len(rews_slice)
        ret = np.zeros(length)
        adv = np.zeros(length)
        frt = np.zeros(length)
        gae = 0.0

        # problem, only using the last one
        gfre = est_last_fr

        # # If larger than the length consider it to be inf
        # idx_g = length * 2
        # idx_c = length * 2
        # frf_flag = 1.0

        for i in reversed(range(length)):
            # v(s_{i+1}) + r - v(s_i)
            delta = (
                rews_slice[i]
                + self.gae_gamma * value_preds_slice[i + 1]
                - value_preds_slice[i]
            )
            gae = delta + self.gae_gamma * self.gae_lambda * gae
            ret[i] = gae + value_preds_slice[i]
            adv[i] = gae

            g_fr = int(rews_slice[i] > 0)
            c_fr = -int(cost_slice[i] < 0)
            gfre = g_fr + c_fr + (1-g_fr) * (1+c_fr) * self.fr_gamma * gfre
            frt[i] = gfre

            # # Stupid calc
            # # like the gae we need same mechanism to estimate. Should we perform this?
            # # update frt
            # idx_c = idx_c if cost_slice[i] >= 0 else i
            # idx_g = idx_g if rews_slice[i] <= 0 else i
            # if idx_c >= idx_g:
            #     frf_flag = -1.0
            #     idx_N = idx_g
            # else:
            #     frf_flag = 1.0
            #     idx_N = idx_c
            # frt[i] = np.power(self.fr_gamma, idx_N - i) * frf_flag if idx_N < length else 0

        self.mb_ret[env_index, path_slice] = ret
        self.mb_adv[env_index, path_slice] = adv
        self.mb_frt[env_index, path_slice] = frt
