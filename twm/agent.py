import torch
from torch import nn

from actor_critic import ActorCritic
from world_model import WorldModel
import utils


class Agent(nn.Module):

    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.wm = WorldModel(config, num_actions)
        self.ac = ActorCritic(config, num_actions, self.wm.z_dim, self.wm.h_dim)


class Dreamer:
    # reset: s_t-1, a_t-1, r_t-1, d_t-1, s_t => s_t, h_t-1
    # step:  a_t => s_t+1, h_t, r_t, d_t

    def __init__(self, config, wm, mode, ac=None, store_data=False, start_z_sampler=None, always_compute_obs=False):
        '''
        config: 配置
        wm: WorldModel 实例
        mode: 'imagine' 或 'observe'
        ac: ActorCritic 实例 (仅在 mode='imagine' 时使用)
        store_data: 是否存储数据
        start_z_sampler: 用于生成初始状态 z 的采样器 (仅在 mode='imagine' 时使用)
        always_compute_obs: 是否在每一步都计算观察
        '''
        assert mode in ('imagine', 'observe')
        assert mode != 'imagine' or start_z_sampler is not None
        self.config = config
        self.wm = wm
        self.ac = ac
        self.mode = mode
        self.store_data = store_data
        self.start_z_sampler = start_z_sampler
        self.always_compute_obs = always_compute_obs

        self.cumulative_g = None  # cumulative discounts
        self.stop_mask = None  # history of dones, for transformer
        self.mems = None
        self.prev_z = None
        self.prev_o = None
        self.prev_h = None
        self.prev_r = None
        self.prev_g = None  # discounts
        self.prev_d = None  # episode ends

        if store_data:
            self.z_data = None
            self.o_data = None
            self.h_data = None
            self.a_data = None
            self.r_data = None
            self.g_data = None
            self.d_data = None
            self.weight_data = None

    @torch.no_grad()
    def get_data(self):
        assert self.store_data
        z = torch.cat(self.z_data, dim=1)
        o = torch.cat(self.o_data, dim=1) if len(self.o_data) > 0 else None
        h = torch.cat(self.h_data, dim=1)
        a = torch.cat(self.a_data, dim=1)
        r = torch.cat(self.r_data, dim=1)
        g = torch.cat(self.g_data, dim=1)
        d = torch.cat(self.d_data, dim=1)
        weights = torch.cat(self.weight_data, dim=1)
        return z, o, h, a, r, g, d, weights

    def _zero_h(self, batch_size, device):
        return torch.zeros(batch_size, 1, self.wm.h_dim, device=device)

    def _reset(self, start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data=False):
        '''
        start_z shape is (batch_size * num_windows, z_categoricals * z_categories)
        start_a shape is (batch_size * num_windows, num_actions)
        start_r shape is (batch_size * num_windows, 1)
        start_terminated shape is (batch_size * num_windows, 1)
        start_truncated shape is (batch_size * num_windows, 1)
        keep_start_data=False
        '''
        assert utils.same_batch_shape([start_a, start_r, start_terminated, start_truncated])
        assert utils.same_batch_shape_time_offset(start_z, start_r, 1)
        assert not (keep_start_data and not self.store_data)
        config = self.config
        # 设置为eval模式，也就是这里不训练wm模型
        wm = self.wm.eval()
        obs_model = wm.obs_model
        dyn_model = wm.dyn_model

        # 根据结束状态获取折扣矩阵
        start_g = wm.to_discounts(start_terminated) # start_g shape is (batch_size * num_windows, 1)
        start_d = torch.logical_or(start_terminated, start_truncated) # start_d shape is (batch_size * num_windows, 1)
        if self.mode == 'imagine' or (self.mode == 'observe' and config['ac_input_h']):
            if start_a.shape[1] == 0: # todo 
                h = self._zero_h(start_a.shape[0], start_a.device)
                mems = None
            else:
                # dyn_model 对 start_z, start_a, start_r, start_g, start_d 进行预测，可以看到这些都去除了最后一个时间步
                # 结果就是预测下一个时间步的状态 z, r, g, d得到隐藏状态（即每一层transformer的输出）h和记忆mems(即每一层transformer的输出和之前的历史记忆)
                _, h, mems = dyn_model.predict(
                    start_z[:, :-1], start_a, start_r[:, :-1], start_g[:, :-1], start_d[:, :-1], heads=[], tgt_length=1)
        else:
            h, mems = None, None

        # set cumulative_g to 1 for real data, start discounting after that
        start_weights = (~start_d).float()
        self.cumulative_g = torch.ones_like(start_g[:, -1:])
        self.stop_mask = start_d

        # 以下都是将初始状态的最后一个时间步的值取出来
        z = start_z[:, -1:]
        r = start_r[:, -1:]
        g = start_g[:, -1:]
        d = start_d[:, -1:]

        # 用于act方法中起始状态的初始化
        self.mems = mems
        self.prev_z = z
        self.prev_h = h
        self.prev_r = r
        self.prev_g = g
        self.prev_d = d

        if self.store_data:
            '''
            是否存储在想象或观察过程中收集和保存智能体的轨迹数据
            todo 后续作用
            '''
            self.h_data = [self._zero_h(start_z.shape[0], start_z.device) if h is None else h]

            if keep_start_data:
                # 是否保存起始状态（即全部轨迹信息）
                self.z_data = [start_z]
                self.a_data = [start_a]
                self.r_data = [start_r]
                self.g_data = [start_g]
                self.d_data = [start_d]
                self.weight_data = [start_weights]
            else:
                # 但是貌似默认时false，索引应该是只保存最后一个时间步的值
                self.z_data = [z]
                self.a_data = []
                self.r_data = []
                self.g_data = []
                self.d_data = []
                self.weight_data = []

        if self.always_compute_obs:
            # 对最后一个时间步的状态 z 进行解码，得到观察 o
            start_o = obs_model.decode(start_z)
            o = start_o[:, -1:] # 最后一个时间步的观察
            self.prev_o = o
            if self.store_data:
                if keep_start_data:
                    # 保存全部的观察轨迹信息
                    self.o_data = [start_o]
                else:
                    # 仅保存最后一个时间步的观察
                    self.o_data = [o]
        else:
            if self.store_data:
                self.o_data = []

        '''
        z shape is (batch_size * num_windows, z_categoricals * z_categories) 最后一个环境时间步的状态
        h shape is (batch_size * num_windows, n_layer, h_dim)  transformer每一层的输出
        start_g shape is (batch_size * num_windows, 1) 折扣矩阵
        start_d shape is (batch_size * num_windows, 1) 是否终止或截断
        但是貌似在训练过程中没有使用到 todo
        '''
        return z, h, start_g, start_d

    @torch.no_grad()
    def imagine_reset(self, start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data=False):
        '''
        start_z shape is (batch_size * num_windows, z_categoricals * z_categories)
        start_a shape is (batch_size * num_windows, num_actions)
        start_r shape is (batch_size * num_windows, 1)
        start_terminated shape is (batch_size * num_windows, 1)
        start_truncated shape is (batch_size * num_windows, 1)
        '''
        assert self.mode == 'imagine'
        # returns: z, h, start_g, start_d
        return self._reset(start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data)

    @torch.no_grad()
    def observe_reset(self, start_o, start_a, start_r, start_terminated, start_truncated, keep_start_data=False):
        assert self.mode == 'observe'
        obs_model = self.wm.obs_model.eval()
        start_z = obs_model.encode_sample(start_o, temperature=0)
        z, h, start_g, start_d = self._reset(
            start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data)
        return z, h, start_z, start_g, start_d

    @staticmethod
    def _create_single_data(batch_size, device):
        start_a = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        start_r = torch.zeros(batch_size, 0, device=device)
        start_terminated = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        start_truncated = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        return start_a, start_r, start_terminated, start_truncated

    @torch.no_grad()
    def imagine_reset_single(self, start_z, keep_start_data=False):
        assert start_z.shape[1] == 1
        start_a, start_r, start_terminated, start_truncated = self._create_single_data(start_z.shape[0], start_z.device)
        return self.imagine_reset(start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data)

    @torch.no_grad()
    def observe_reset_single(self, start_o, keep_start_data=False):
        assert start_o.shape[1] == 1
        start_a, start_r, start_terminated, start_truncated = self._create_single_data(start_o.shape[0], start_o.device)
        return self.observe_reset(start_o, start_a, start_r, start_terminated, start_truncated, keep_start_data)

    def _step(self, a, z, r, g, d, temperature, return_attention):
        '''
        a: 在正式训练时仅传入a，验证评估时传入a, z, r, g, d，用真实的观察数据替代想象的数据
        None
        None 
        None
        None
        temperature
        return_attention
        '''
        config = self.config
        imagine = self.mode == 'imagine' # 观察时这边为False
        assert a.shape[1] == 1
        assert all(x is None for x in (z, r, g, d)) if imagine else utils.same_batch_shape([a, z, r, g, d])
        wm = self.wm.eval()
        obs_model = wm.obs_model
        dyn_model = wm.dyn_model

        z_dist = None
        if imagine or self.config['ac_input_h']:
            # 在训练时imagine为true
            assert self.mems is not None or self.prev_r.shape[1] == 0
            assert self.mems is None or a.shape[0] == self.mems[0].shape[1]
            heads = ['z', 'r', 'g'] if imagine else []
            # 利用上一个状态和预测的动作，预测下一个状态 z, r, g
            outputs = dyn_model.predict(
                self.prev_z, a, self.prev_r, self.prev_g, self.stop_mask, tgt_length=1, heads=heads, mems=self.mems,
                return_attention=return_attention)
            preds, h, mems, attention = outputs if return_attention else (outputs + (None,))
            if imagine:
                # 在想象过程中，z_dist是一个分布，用于采样下一个状态 z
                z_dist = preds['z_dist']
                # 根据 z_dist 和温度参数采样下一个状态 z
                z = obs_model.sample_z(z_dist, temperature=temperature)
                # 如果是观察模式，则直接使用 z
                r = preds['r']
                # 如果是想象模式，则使用预测的奖励 r
                g = preds['g']
                # 如果是想象模式，则使用预测的折扣 g
        else:
            h, mems, attention = None, None, None

        # cumulative_g初始状态是一个全1的张量，表示从初始状态开始的折扣
        if self.cumulative_g.shape[1] == 0:
            # 表示当前是第一个时间步（或者说刚刚重置后的初始状态）
            weights = torch.ones_like(g)
            self.cumulative_g = g.clone()
        else:
            # 如果不是第一个时间步，则根据上一个状态的结束标志和当前状态的折扣计算权重
            done = self.prev_d.float()
            not_done = (~self.prev_d).float()
            # 如果prev_d为True，则表示当前状态是一个结束状态，当前是新状态，则权重采用全1矩阵
            # 如果prev_d为False，则表示当前状态不是一个结束状态，那么权重采用self.cumulative_g * not_done
            weights = self.cumulative_g * not_done + torch.ones_like(self.prev_g) * done
            # 更新 cumulative_g
            # 这行代码在 [Dreamer._step]agent.py ) 方法中用于更新累积折扣因子，是强化学习中实现跨多个时间步的回报计算的关键部分
            self.cumulative_g = (not_done * self.cumulative_g + done) * g

        if imagine:
            if config['wm_discount_threshold'] > 0:
                # 这行代码用于检测何时需要重置一个想象的轨迹，基于累积折扣值。让我详细解释这个机制
                # 如果低于阈值，则将该轨迹标记为"完成"(done)
                d = (self.cumulative_g < config['wm_discount_threshold'])
                num_done = d.sum()
                if num_done > 0:
                    # 为这些"完成"的轨迹重新采样新的起始状态
                    new_start_z = self.start_z_sampler(num_done)
                    z[d] = new_start_z
            else:
                d = torch.zeros(a.shape[0], 1, dtype=torch.bool, device=a.device)

        # 结合上一个状态的结束标志和当前状态的结束标志，更新 stop_mask
        stop_mask = torch.cat([self.stop_mask, d], dim=1)
        memory_length = config['wm_memory_length']
        if stop_mask.shape[1] > memory_length + 1:
            # 如果 stop_mask 的长度超过了 memory_length + 1，则截断
            stop_mask = stop_mask[:, -(memory_length + 1):]
        self.stop_mask = stop_mask

        # 走了一步后，更新记录的上一步信息 prev_z, prev_h, prev_r, prev_g, prev_d
        # 这边近存储一个时间步的值
        self.mems = mems
        self.prev_z, self.prev_h, self.prev_r, self.prev_g, self.prev_d = z, h, r, g, d

        if self.store_data:
            # 如果有存储数据，则将当前的 z, h, r, g, d, weights 存储到对应的列表中
            # 这边会存储每个时间步的值
            self.z_data.append(z)
            self.h_data.append(h)
            self.a_data.append(a)
            self.r_data.append(r)
            self.g_data.append(g)
            self.d_data.append(d)
            self.weight_data.append(weights)

        if self.always_compute_obs:
            # 如果 always_compute_obs 为 True，则每一步都计算观察，也就是将特征解码为观察
            o = obs_model.decode(z)
            # 并将观察存储起来
            self.prev_o = o
            if self.store_data:
                self.o_data.append(o)

        # 将当前的 z, h, z_dist, r, g, d, weights 打包成一个元组返回
        outputs = (z, h, z_dist, r, g, d, weights)
        if return_attention:
            outputs = outputs + (attention,)
        return outputs

    @torch.no_grad()
    def imagine_step(self, a, temperature=1, return_attention=False):
        assert self.mode == 'imagine'
        # returns: z, h, z_dist, r, g, d, weights, [attention]
        return self._step(a, None, None, None, None, temperature, return_attention)

    @torch.no_grad()
    def observe_step(self, a, o, r, terminated, truncated, return_attention=False):
        '''
        在观察模式下，传入的参数包括动作 a、观察 o、奖励 r、终止标志 terminated 和截断标志 truncated
        想象模型根据这些完成动作的预测和观察数据来更新状态
        '''
        assert self.mode == 'observe'
        wm = self.wm
        obs_model = wm.obs_model
        obs_model.eval()
        z = obs_model.encode_sample(o, temperature=0)
        g = wm.to_discounts(terminated)
        d = torch.logical_or(terminated, truncated)
        if return_attention:
            _, h, _, _, _, _, weights, attention = self._step(a, z, r, g, d, temperature=None, return_attention=True)
            return z, h, g, d, weights, attention
        else:
            _, h, _, _, _, _, weights = self._step(a, z, r, g, d, temperature=None, return_attention=False)
            return z, h, g, d, weights

    @torch.no_grad()
    def act(self, temperature=1, epsilon=0):
        z, h = self.prev_z, self.prev_h
        # 返回动作的索引
        a = self.ac.policy(z, h, temperature=temperature)
        if epsilon > 0:
            # 以下是在 epsilon-greedy 策略下进行随机动作选择
            # 获取动作的数量
            num_actions = self.ac.num_actions
            # 生成一个与 a 相同形状的随机掩码，掩码中小于 epsilon 的位置为 True
            epsilon_mask = torch.rand_like(a, dtype=torch.float) < epsilon
            # 在掩码为 True 的位置，随机选择动作，False的位置保持不变
            # 进一步的随机选择动作
            random_actions = torch.randint_like(a, num_actions)
            a[epsilon_mask] = random_actions[epsilon_mask]
        return a
