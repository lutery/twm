import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

import utils


class ReplayBuffer:

    def __init__(self, config, env):
        self.config = config
        self.env = env

        device = config['buffer_device']
        self.device = torch.device(device)
        self.prev_seed = (config['seed'] + 1) * 7979
        initial_obs, _ = env.reset(seed=self.prev_seed)
        initial_obs = torch.as_tensor(np.array(initial_obs), device=device) # 初始的环境大小
        capacity = config['buffer_capacity'] # 缓冲区的容量

        self.obs = torch.zeros((capacity + 1,) + initial_obs.shape, dtype=initial_obs.dtype, device=device)
        self.obs[0] = initial_obs
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float, device=device)
        self.terminated = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.timesteps = torch.zeros(capacity + 1, dtype=torch.long, device=device)
        self.timesteps[0] = 0
        self.sample_visits = torch.zeros(capacity, dtype=torch.long, device='cpu')  # we sample indices on cpu

        self.capacity = capacity
        self.size = 0
        self.total_reward = 0
        self.score = 0
        self.episode_lengths = []
        self.scores = []
        # 这里看起来是在做奖励的缩放，比如使用tanh将奖励缩放到-1~1之间
        self.reward_transform = utils.create_reward_transform(config['env_reward_transform'])
        self.metrics_num_episodes = 0

    def sample_random_action(self):
        return self.env.action_space.sample()

    def _get(self, array, idx, device=None, prefix=0, return_next=False, repeat_fill_value=None, allow_last=False):
        '''
        array: 需要获取的数组
        idx: 索引，可以是整数、范围、元组、列表或numpy数组，可能为[wm_total_batch_size],可能为【1, replay_buffer.size】
        device: 设备，如果不为None，则将结果转换到指定设备

        return: 返回指定idx索引的数据，并且根据prefix和return_next参数进行前面N个数据的获取和后面一个数据的获取，如果索引越界，将越界的数据填充为repeat_fill_value
        '''

        assert prefix >= 0
        squeeze_seq = False # 这里表示是否需要将时序列压缩
        squeeze_batch = False # 这里表示是否需要将批次压缩
        # 最终将idx转换为torch张量
        if isinstance(idx, int):
            idx = torch.tensor([idx])
            squeeze_seq = True # 代表时序列
        if isinstance(idx, range):
            idx = tuple(idx)
        if isinstance(idx, (tuple, list, np.ndarray)):
            idx = torch.as_tensor(idx, device=self.device)
            # idx shape is 【wm_total_batch_size】

        assert torch.is_tensor(idx)
        assert torch.all(idx >= 0)
        assert torch.all((idx <= self.size) if allow_last else (idx < self.size))

        # 维度至少为2
        if idx.ndim == 1:
            idx = idx.unsqueeze(0) # shape is [1, wm_total_batch_size]
            squeeze_batch = True

        if prefix > 0 or return_next:
            idx_list = [idx]
            if prefix > 0:
                # 获取前缀索引
                # idx[:, 0] 是每个序列的第一个索引
                # unsqueeze(-1) 将其转换为列向量
                # torch.arange(-prefix, 0, device=idx.device) 生成一个从-prefix到-1的范围
                # prefix_idx < 0 是一个布尔掩码，表示前缀索引是否小于0
                # 如果前缀索引小于0，则将其设置为0
                # 这样可以确保前缀索引不会超出缓冲区的范围
                prefix_idx = idx[:, 0].unsqueeze(-1) + torch.arange(-prefix, 0, device=idx.device) # prefix_idx shape is [1, 1+prefix]
                prefix_mask = prefix_idx < 0
                # repeat first value, if prefix goes beyond the first value in the buffer
                prefix_idx[prefix_mask] = 0
                # 将生产的前缀索引插入到索引列表的开头
                idx_list.insert(0, prefix_idx)

            if return_next:
                # 这里就是获取下一个索引
                # idx[:, -1] 是每个序列的最后一个索引
                # unsqueeze(-1) 将其转换为列向量
                # suffix_idx = last_idx + 1 是下一个索引
                # suffix_mask 是一个布尔掩码，表示下一个索引是否超出了缓冲区的范围
                # 如果下一个索引超出了缓冲区的范围，则将其设置为最后一个索引
                # 这样可以确保下一个索引不会超出缓冲区的范围
                # allow_last=True 允许下一个索引等于缓冲区的大小
                # allow_last=False 则下一个索引必须小于缓冲区的大小
                # unsqueeze(1) 将其转换为列向量
                # 这样可以确保下一个索引的形状与其他索引一致
                # 最终将下一个索引添加到索引列表的末尾
                last_idx = idx[:, -1] # last_idx shape (1, 1)
                suffix_idx = last_idx + 1 # suffix_idx shape (1, 1)
                # repeat value, if next goes beyond the last value in the buffer
                suffix_mask = (suffix_idx > self.size) if allow_last else (suffix_idx >= self.size)
                suffix_idx = suffix_idx * (~suffix_mask) + last_idx * suffix_mask
                # 将下一个索引添加到索引列表的末尾
                idx_list.append(suffix_idx.unsqueeze(1))

            # 将所有索引列表中的索引沿着第二个维度（dim=1）连接起来
            # idx 0 shape is (1, 1 + prefix)
            # idx 1 shape is (1, wm_total_batch_size)
            # idx 2 shape is (1, 1)
            # cat later shape is (1, 1 + prefix + wm_total_batch_size + 1)
            idx = torch.cat(idx_list, dim=1)
            # 根据索引从数组中获取数据
            x = array[idx]

            if repeat_fill_value is not None:
                if prefix > 0:
                    # 对于超出前缀范围的索引，将其填充为repeat_fill_value
                    tmp = x[:, :prefix]
                    tmp[prefix_mask] = repeat_fill_value
                    x[:, :prefix] = tmp
                if return_next:
                    # 对于超出后缀范围的索引，将其填充为repeat_fill_value
                    x[suffix_mask, -1] = repeat_fill_value
        else:
            # shape is (1, wm_total_batch_size)
            x = array[idx]

        # 对输入的张量进行处理，应该主要针对的是传入的idx原本只是个整数的情况或者一维数组的情况
        if squeeze_seq:
            x = x.squeeze(1)
        if squeeze_batch:
            x = x.squeeze(0)
        # x shape [wm_total_batch_size, frame_stack, h, w, c] 或者 [1 + prefix + wm_total_batch_size + 1, frame_stack, h, w, c]
        # 如果传入的idx本身就是二维，那么x shape 可能 is [1, 1 + prefix + wm_total_batch_size, frame_stack, h, w, c]
        if device is not None and x.device != device:
            return x.to(device=device)
        return x

    def get_obs(self, idx, device=None, prefix=0, return_next=False):
        # return obs : [wm_total_batch_size, frame_stack, h, w, c] 或者 [1 + prefix + wm_total_batch_size + 1, frame_stack, h, w, c]
        # 或 return obs: [1, 1 + prefix + replay.size, frame_stack, h, w, c]
        obs = self._get(self.obs, idx, device, prefix, return_next=return_next, allow_last=True)
        return utils.preprocess_atari_obs(obs, device)

    def get_actions(self, idx, device=None, prefix=0):
        return self._get(self.actions, idx, device, prefix, repeat_fill_value=0)  # noop

    def get_rewards(self, idx, device=None, prefix=0):
        return self._get(self.rewards, idx, device, prefix, repeat_fill_value=0.)

    def get_terminated(self, idx, device=None, prefix=0):
        return self._get(self.terminated, idx, device, prefix)

    def get_truncated(self, idx, device=None, prefix=0):
        return self._get(self.truncated, idx, device, prefix)

    def get_timesteps(self, idx, device=None, prefix=None):
        return self._get(self.timesteps, idx, device, prefix, allow_last=True)

    def get_data(self, idx, device=None, prefix=None, return_next_obs=False):
        obs = self.get_obs(idx, device, prefix, return_next_obs)
        actions = self.get_actions(idx, device, prefix)
        rewards = self.get_rewards(idx, device, prefix)
        terminated = self.get_terminated(idx, device, prefix)
        truncated = self.get_truncated(idx, device, prefix)
        timesteps = self.get_timesteps(idx, device, prefix)
        return obs, actions, rewards, terminated, truncated, timesteps

    def step(self, policy_fn):
        '''
        policy_fn: 是一个lambda函数，输入当前的index，输出一个动作，实际上index没有使用
        '''
        config = self.config
        index = self.size
        if index >= config['buffer_capacity']:
            raise ValueError('Buffer overflow')
        
        # todo 这里是怎么选择动作的？
        action = policy_fn(index)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # throws away last obs
            # 如果游戏结束了，则重置环境，并且设置新的随机种子
            seed = self.prev_seed
            if seed is not None:
                seed = seed * 3 + 13
                self.prev_seed = seed
            next_obs, _ = self.env.reset(seed=seed)

        # 存储着连续数据
        self.obs[index + 1] = torch.as_tensor(np.array(next_obs), device=self.device)
        # 存储缩放后的奖励
        self.rewards[index] = self.reward_transform(reward)
        self.actions[index] = action
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        # 如果是终止或者截断，则将时间步数重置为0，否则加1
        # 应该可以用来计算每个episode的长度，区分不同的episode
        self.timesteps[index + 1] = 0 if (terminated or truncated) else (self.timesteps[index] + 1)

        # 增加存储的数据量
        self.size = index + 1
        # 统计总奖励和当前episode的分数
        self.total_reward += reward
        self.score += reward
        if terminated or truncated:
            # 记录着每次episode的长度和分数
            self.episode_lengths.append(self.timesteps[index] + 1)
            self.scores.append(self.score)
            # 重置当前episode的分数
            self.score = 0

    def _compute_visit_probs(self, n):
        temperature = self.config['buffer_temperature']
        if temperature == 'inf':
            visits = self.sample_visits[:n].float()
            visit_sum = visits.sum()
            if visit_sum == 0:
                probs = torch.full_like(visits, 1 / n)
            else:
                probs = 1 - visits / visit_sum
        else:
            logits = self.sample_visits[:n].float() / -temperature
            probs = F.softmax(logits, dim=0)
        assert probs.device.type == 'cpu'
        return probs

    def sample_indices(self, max_batch_size, sequence_length):
        n = self.size - sequence_length + 1
        batch_size = max_batch_size
        if batch_size * sequence_length > n:
            raise ValueError('Not enough data in buffer')

        probs = self._compute_visit_probs(n)
        start_idx = torch.multinomial(probs, batch_size, replacement=False)

        # stay on cpu
        flat_idx = start_idx.reshape(-1)
        flat_idx, counts = torch.unique(flat_idx, return_counts=True)
        self.sample_visits[flat_idx] += counts

        start_idx = start_idx.to(device=self.device)
        idx = start_idx.unsqueeze(-1) + torch.arange(sequence_length, device=self.device)
        return idx

    def generate_uniform_indices(self, batch_size, sequence_length, extra=0):
        '''
        extra: 2 for context + next

        return: 生成一个迭代器，每次返回一个batch_size个序列的起始索引，形状为(batch_size, sequence_length + extra)
        这里的sequence_length是指每个序列的长度，extra是指额外的上下文长度，比如2表示上下文和下一个状态
        这里的batch_size是指每次生成的序列的数量
        这里的start_offset是一个随机的偏移量，表示从哪个位置开始生成序列
        这里的start_idx是一个范围为[start_offset, size - sequence_length]的索引，并且其步长为sequence_length，这里应该是获取了多段squences的起始索引
        '''
        start_offset = random.randint(0, sequence_length - 1) # 获取一个随机的偏移量 n（小于sequence_length）
        start_idx = torch.arange(start_offset, self.size - sequence_length, sequence_length,
                                 dtype=torch.long, device=self.device) # 获取了一个范围为[start_offset, size - sequence_length]的索引，并且其步长为sequence_length，这里应该是获取了多段squences的起始索引， shape ((self.size - sequence_length - start_offset) // sequence_length)
        start_idx = start_idx[torch.randperm(start_idx.shape[0], device=self.device)] # 打乱这些索引
        while len(start_idx) > 0:
            idx = start_idx[:batch_size] # 获取batch_size个起始序列索引， shape is (batch_size,)
            idx = idx.unsqueeze(-1) + torch.arange(sequence_length + extra, device=self.device) # shape is (batch_size, sequence_length + extra)
            yield idx
            start_idx = start_idx[batch_size:]

    def compute_visit_entropy(self):
        if self.size <= 1:
            return 1.0
        # compute normalized entropy
        visits = self.sample_visits[:self.size]
        visit_sum = visits.sum()
        if visit_sum == 0:
            return 1.0
        max_entropy = math.log(self.size)
        entropy = D.Categorical(probs=visits / visit_sum).entropy().item()
        normalized_entropy = round(min(entropy / max_entropy, 1), 5)
        return normalized_entropy

    def _get_histogram(self, values, step):
        import wandb
        num_bins = int(math.ceil(self.size / step)) + 1
        bins = np.arange(num_bins) * step
        values = [v.sum().item() for v in torch.split(values, step)]
        return wandb.Histogram(np_histogram=[values, bins])

    def visit_histogram(self):
        visits = self.sample_visits[:self.size]
        return self._get_histogram(visits, step=500)

    def sample_probs_histogram(self):
        n = self.size
        visit_probs = self._compute_visit_probs(n)
        return self._get_histogram(visit_probs, step=500)

    def metrics(self):
        # 收集缓冲区的统计信息
        # 计算缓冲区的大小、总奖励、episode数量和访问熵
        # 如果episode数量超过了metrics_num_episodes，则计算最近episode的平均长度和分数
        num_episodes = len(self.episode_lengths)
        # 指标：缓冲区大小、总奖励、episode数量和访问熵
        # todo 访问熵如何计算来的
        metrics = {'size': self.size, 'total_reward': self.total_reward, 'num_episodes': num_episodes,
                   'visit_ent': self.compute_visit_entropy()}

        # 如果episode数量超过了metrics_num_episodes，则计算最近episode的平均长度步数和分数
        if num_episodes > self.metrics_num_episodes:
            new_episodes = num_episodes - self.metrics_num_episodes
            self.metrics_num_episodes = num_episodes
            metrics.update({'episode_len': utils.compute_mean(self.episode_lengths[-new_episodes:]),
                            'episode_score': np.mean(self.scores[-new_episodes:])})
        return metrics
