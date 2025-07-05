import math

import torch
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
from torch.distributions.utils import logits_to_probs

import nets
import utils


class WorldModel(nn.Module):

    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.num_actions = num_actions

        self.obs_model = ObservationModel(config)
        self.dyn_model = DynamicsModel(config, self.obs_model.z_dim, num_actions)

        self.obs_optimizer = utils.AdamOptim(
            self.obs_model.parameters(), lr=config['obs_lr'], eps=config['obs_eps'], weight_decay=config['obs_wd'],
            grad_clip=config['obs_grad_clip'])
        self.dyn_optimizer = utils.AdamOptim(
            self.dyn_model.parameters(), lr=config['dyn_lr'], eps=config['dyn_eps'], weight_decay=config['dyn_wd'],
            grad_clip=config['dyn_grad_clip'])

    @property
    def z_dim(self):
        return self.obs_model.z_dim

    @property
    def h_dim(self):
        return self.dyn_model.h_dim

    def optimize_pretrain_obs(self, o):
        '''
        o: 传入环境的观察数据，数值范围是[0, 1], 形状为(batch_size, time, frame_stack, height, width, channels)  [wm_total_batch_size, 1, frame_stack, h, w, channels] 或者 [1 + prefix + wm_total_batch_size + 1, 1, frame_stack, h, w, c]
        '''

        obs_model = self.obs_model
        obs_model.train()
        
        # 将观察数据进行特征提取，获取一个潜在分布
        z_dist = obs_model.encode(o)
        # z shape is (batch_size, time, z_categoricals * z_categories)
        z = obs_model.sample_z(z_dist, reparameterized=True) # # 这里返回的是对应的潜在分布的采样结果，z shape is (batch_size, time, z_categoricals * z_categories)
        # 通过潜在分布进行重构,recons shape is (batch_size, time, frame_stack, height, width, channels)
        recons = obs_model.decode(z)

        # no consistency loss required for pretraining
        dec_loss, dec_met = obs_model.compute_decoder_loss(recons, o) # 计算重构损失
        ent_loss, ent_met = obs_model.compute_entropy_loss(z_dist) # 计算熵损失和指标，这里的熵应该是让提取的特征不会过于确定，保持一定的随机性，能够抓住更多的关键特征点

        obs_loss = dec_loss + ent_loss # 汇总全部损失
        self.obs_optimizer.step(obs_loss) # 优化

        metrics = utils.combine_metrics([ent_met, dec_met]) # 合并指标
        metrics['obs_loss'] = obs_loss.detach()
        return metrics # 返回

    def optimize_pretrain_dyn(self, z, a, r, terminated, truncated, target_logits):
        '''
        z shape is （1，sequence_length + extra - 1， z_categoricals * z_categories）
        target_logits: （1，-2 + sequence_length + extra， z_categoricals * z_categories）
        a, r, terminated, truncated, shape is (1, sequence_length + extra - 2)
        '''
        assert utils.same_batch_shape([z, a, r, terminated, truncated]) 
        assert utils.same_batch_shape_time_offset(z, target_logits, 1)
        dyn_model = self.dyn_model
        dyn_model.train()

        d = torch.logical_or(terminated, truncated) # d shape is (1, sequence_length + extra - 2)
        g = self.to_discounts(terminated) # 获取一个折扣矩阵，并且如果terminated对应的位置为true，表示结束，折扣应为0 g shape is (batch_size(1), sequence_length + extra - 2)
        target_weights = (~d[:, 1:]).float() # target_weights shape is (1, -1 + sequence_length + extra - 2)，表示非结束状态的权重
        tgt_length = target_logits.shape[1] # 获取序列长度  tgt_length is sequence_length + extra - 2

        '''
        out/preds shape is {
            'z': (batch_size, tgt_length / num_modalities, z_dim),
            'r': (batch_size, tgt_length / num_modalities, 1),
            'g': (batch_size, tgt_length / num_modalities, 1)
        }
        hiddens/h shape is (batch_size, tgt_length / num_modalities, dim)
        mems shape is [num_layers + 1, mem_length, batch_size, dim]
        '''
        preds, h, mems = dyn_model.predict(z, a, r[:, :-1], g[:, :-1], d[:, :-1], tgt_length, compute_consistency=True)
        dyn_loss, metrics = dyn_model.compute_dynamics_loss(
            preds, h, target_logits=target_logits, target_r=r[:, 1:], target_g=g[:, 1:], target_weights=target_weights)
        # 优化动态环境模型损失，包含观察模型的潜在分布预测损失、奖励预测损失和折扣预测损失
        self.dyn_optimizer.step(dyn_loss)
        return metrics

    def optimize(self, o, a, r, terminated, truncated):
        '''
        obs shape is [wm_total_batch_size, wm_sequence_length, h, w, c]
        actions shape is [wm_total_batch_size, wm_sequence_length]
        rewards shape is [wm_total_batch_size, wm_sequence_length]
        terminated shape is [wm_total_batch_size, wm_sequence_length]
        truncated shape is [wm_total_batch_size, wm_sequence_length]
        '''
        assert utils.same_batch_shape([a, r, terminated, truncated])
        assert utils.same_batch_shape_time_offset(o, r, 1)

        obs_model = self.obs_model
        dyn_model = self.dyn_model

        self.eval() # 将模型切换为评估模式，避免在编码和解码过程中进行梯度计算
        with torch.no_grad():
            context_z_dist = obs_model.encode(o[:, :1]) # 这里仅编码了第一帧，为动态模型提供初始状态，不参与观察模型的重建损失计算
            context_z = obs_model.sample_z(context_z_dist) # 根据前缀观察的潜在分布进行采样，context_z shape is  (batch_size, 1, z_categoricals * z_categories)
            next_z_dist = obs_model.encode(o[:, -1:]) # 获取后续观察的潜在分布，也就是最后一个序列的观察，仅编码了最后一帧
            next_logits = next_z_dist.base_dist.logits # 获取后续观察的潜在分布的logits，next_logits shape is (batch_size, 1, z_categoricals, z_categories)

        self.train() # 将模型切换为训练模式，开始进行梯度计算和优化

        # observation model
        o = o[:, 1:-1] # 这里应该是去除了第一帧和最后一帧
        z_dist = obs_model.encode(o)
        z = obs_model.sample_z(z_dist, reparameterized=True) # 这里返回的是对应的潜在分布的采样结果，z shape is (batch_size, wm_sequence_length - 2, z_categoricals * z_categories)
        recons = obs_model.decode(z)

        dec_loss, dec_met = obs_model.compute_decoder_loss(recons, o) # 计算重构损失和指标
        ent_loss, ent_met = obs_model.compute_entropy_loss(z_dist) # 计算熵损失和指标，这里的熵应该是让提取的特征不会过于确定，保持一定的随机性，能够抓住更多的关键特征点

        # dynamics model
        z = z.detach()
        z = torch.cat([context_z, z], dim=1) # 将第一帧的潜在分布和后续帧的潜在分布拼接起来，z shape is (batch_size, wm_sequence_length - 1, z_categoricals * z_categories)
        z_logits = z_dist.base_dist.logits # z_logits shape is (batch_size, wm_sequence_length - 2, z_categoricals, z_categories)
        target_logits = torch.cat([z_logits[:, 1:].detach(), next_logits.detach()], dim=1) # 这里是将中间帧和最后一帧的潜在分布的logits拼接起来，target_logits shape is (batch_size, wm_sequence_length - 2 + 1, z_categoricals, z_categories)
        d = torch.logical_or(terminated, truncated) # d shape is (batch_size, wm_sequence_length)，表示是否结束的标志
        g = self.to_discounts(terminated) # g shape is (batch_size, wm_sequence_length)，获取一个折扣矩阵，并且如果terminated对应的位置为true，表示结束，折扣应为0
        target_weights = (~d[:, 1:]).float() # target_weights shape is (batch_size, wm_sequence_length - 2)，表示非结束状态的权重
        tgt_length = target_logits.shape[1] # wm_sequence_length - 2 + 1

        preds, h, mems = dyn_model.predict(z, a, r[:, :-1], g[:, :-1], d[:, :-1], tgt_length, compute_consistency=True)
        dyn_loss, dyn_met = dyn_model.compute_dynamics_loss(
            preds, h, target_logits=target_logits, target_r=r[:, 1:], target_g=g[:, 1:], target_weights=target_weights)
        self.dyn_optimizer.step(dyn_loss)

        z_hat_probs = preds['z_hat_probs'].detach()
        con_loss, con_met = obs_model.compute_consistency_loss(z_logits, z_hat_probs)

        obs_loss = dec_loss + ent_loss + con_loss
        self.obs_optimizer.step(obs_loss)

        metrics = utils.combine_metrics([dec_met, ent_met, con_met, dyn_met])
        metrics['obs_loss'] = obs_loss.detach()

        return z, h, metrics

    @torch.no_grad()
    def to_discounts(self, mask):
        assert utils.check_no_grad(mask)
        discount_factor = self.config['env_discount_factor']
        g = torch.full(mask.shape, discount_factor, device=mask.device) # 获取折扣因子，创建和mask相同shape的折扣矩阵
        g = g * (~mask).float() # 如果mask对应的位置为True（表示结束），那么对应位置去反为false，float后为0，给折扣矩阵对应位置的折扣设置为0
        return g


class ObservationModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # todo 看起来是将观察编码、解码的维度
        self.z_dim = config['z_categoricals'] * config['z_categories']

        h = config['obs_channels']
        activation = config['obs_act']
        norm = config['obs_norm']
        dropout_p = config['obs_dropout']

        num_channels = config['env_frame_stack']
        if not config['env_grayscale']:
            num_channels *= 3

        self.encoder = nn.Sequential(
            nets.CNN(num_channels, [h, h * 2, h * 4], h * 8,
                     [4, 4, 4, 4], [2, 2, 2, 2], [0, 0, 0, 0], activation, norm=norm, post_activation=True),
            nn.Flatten(),
            nets.MLP((h * 8) * 2 * 2, [512, 512], self.z_dim, activation, norm=norm, dropout_p=dropout_p)
        )

        # no norm here
        self.decoder = nn.Sequential(
            nets.MLP(self.z_dim, [], (h * 16) * 1 * 1, activation, dropout_p=dropout_p, post_activation=True),
            nn.Unflatten(1, (h * 16, 1, 1)),
            nets.TransposeCNN(h * 16, [h * 4, h * 2, h], num_channels, [5, 5, 6, 6], [2, 2, 2, 2], [0, 0, 0, 0],
                              activation, final_bias_init=0.5)
        )

    @staticmethod
    def create_z_dist(logits, temperature=1):
        assert temperature > 0 # 这里的温度是为了控制采样的平滑度，temperature越大，采样越平滑
        # OneHotCategoricalStraightThrough
        # 用于将观察编码为离散的潜在表示
        # temperature 参数控制采样的"软硬程度"
        # 通过 Straight-Through 技巧实现离散变量的可导采样

        # independent 的作用
        '''
        reinterpreted_batch_ndims=1 表示将最后一个维度视为事件维度
        对于形状 (batch, time, z_categoricals, z_categories) 的数据：
        将 z_categoricals 个类别分布视为独立分布

        联合概率计算:
        自动处理多个独立分布的联合概率
        在计算 log probability 时自动对事件维度求和

        学习离散的潜在表示
        保持端到端的可训练性
        正确处理多个独立分布的概率计算
        todo 调试了解
        '''
        return D.Independent(D.OneHotCategoricalStraightThrough(logits=logits / temperature), 1)

    def encode(self, o):
        '''
        o: 形状为(batch_size, time, frame_satck, height, width, channels)  [wm_total_batch_size, 1, frame_stack, h, w, channels] 或者 [1 + prefix + wm_total_batch_size + 1, 1, frame_stack, h, w, channels]

        return: 返回的分布对象
        '''
        assert utils.check_no_grad(o)
        config = self.config
        shape = o.shape[:2] # 获取batch_size, timestep
        o = o.flatten(0, 1) # 仅展平 前两个维度，o shape is (batch_size * time, [frame_stack], height, width, channels)

        if not config['env_grayscale']: # 如果是彩色图，那么存在channels维度，需要将channels维度展平
            o = o.permute(0, 1, 4, 2, 3) # o shape is (batch_size * time, [frame_stack], height, width, channels) -> (batch_size * time, [frame_stack], channels, height, width)
            o = o.flatten(1, 2) # 展品第1个维度和第2个维度， shape is (batch_size * time, [frame_stack * channels], height, width)

        # z_logits shape is (batch_size * time, z_categoricals * z_categories(z_dim))
        z_logits = self.encoder(o)
        z_logits = z_logits.unflatten(0, shape) # 将展平的维度恢复为原来的维度，shape is (batch_size, time, z_categoricals * z_categories(z_dim))
        z_logits = z_logits.unflatten(-1, (config['z_categoricals'], config['z_categories'])) # shape is (batch_size, time, z_categoricals, z_categories)
        z_dist = ObservationModel.create_z_dist(z_logits) # 调用观察模型的create_z_dist方法，创建一个独立的OneHotCategorical分布
        return z_dist

    def sample_z(self, z_dist, reparameterized=False, temperature=1, idx=None, return_logits=False):
        '''
        当 reparameterized=True 时：
        使用 rsample() 方法进行采样
        实现可导的采样过程
        允许梯度通过采样操作反向传播
        当 reparameterized=False 时：

        使用 sample() 方法进行采样
        采样过程不可导
        阻止梯度通过采样操作反向传播


        return_logits: 是否返回 logits
        '''
        logits = z_dist.base_dist.logits # shape is (batch_size, time, z_categoricals, z_categories)
        # 这里应该是为了确保参数可导性是否和 reparameterized 参数一致
        assert (not reparameterized) == utils.check_no_grad(logits)
        if temperature == 0:
            # 这里是参数不可导的流程
            assert not reparameterized
            with torch.no_grad():
                if idx is not None:
                    logits = logits[idx] # 这里的idx需要保证小于 logits 的第一个维度大小 所以这里之后 logits shape (idx batch_size, time, z_categoricals, z_categories)
                indices = torch.argmax(logits, dim=-1) # shape is (batch_size, time, z_categoricals)
                # one_hot = F.one_hot(indices, num_classes=self.config['z_categories'])
                # shape: (batch_size, time, z_categoricals, z_categories)
                # flattened = one_hot.flatten(2, 3)
                # shape: (batch_size, time, z_categoricals * z_categories)
                z = F.one_hot(indices, num_classes=self.config['z_categories']).flatten(2, 3).float()
            if return_logits:
                # 不但返回预测的最大可能预测值，还会返回原始的logits
                return z, logits  # actually wrong logits for temperature = 0
            # 这里是不可导的采样流程，直接返回
            return z

        if temperature != 1 or idx is not None:
            if idx is not None:
                logits = logits[idx] # 这里的idx需要保证小于 logits 的第一个维度大小 所以这里之后 logits shape (idx batch_size, time, z_categoricals, z_categories)
            # 这里是可导的采样流程，使用温度参数进行采样，又将logits包装为预测分布的对象
            z_dist = ObservationModel.create_z_dist(logits, temperature)
            if return_logits:
                # 获取新的归一化的logits
                logits = z_dist.base_dist.logits  # return new normalized logits

        # 这里是可导的采样流程，使用 reparameterized 参数决定是否使用 rsample() 方法进行采样
        '''
        `rsample()` 和 `sample()` 是 PyTorch 分布类中两种不同的采样方法，主要区别在于梯度传播：

        ### `sample()`
        - **不可导的采样**
        - 直接从分布中采样
        - **阻止梯度反向传播**
        - 用于推理阶段

        ```python
        # 示例
        dist = D.Normal(mean, std)
        z = dist.sample()  # 梯度在此处停止
        ```

        ### `rsample()`
        - **可重参数化的采样**
        - 使用重参数化技巧进行采样
        - **允许梯度反向传播**
        - 用于训练阶段

        ```python
        # 示例
        dist = D.Normal(mean, std)
        z = dist.rsample()  # 梯度可以通过采样传播
        ```

        ### 在代码中的使用
        ```python
        # 在 sample_z 函数中
        z = z_dist.rsample() if reparameterized else z_dist.sample()
        ```
        - 当 `reparameterized=True` 时使用 `rsample()` 用于训练
        - 当 `reparameterized=False` 时使用 `sample()` 用于推理

        ### 重参数化技巧的工作原理
        1. 将随机性从分布参数中分离
        2. 使用确定性函数将标准分布样本转换为目标分布样本
        3. 保持计算图的可导性

        这种设计对于变分自编码器（VAE）等需要通过随机变量反向传播梯度的模型特别重要。
        '''
        z = z_dist.rsample() if reparameterized else z_dist.sample()
        # z shape is (batch_size, time, z_categoricals, z_categories)
        z = z.flatten(2, 3) # 将最后两个维度展平，shape is (batch_size, time, z_categoricals * z_categories)
        if return_logits:
            return z, logits
        return z

    def encode_sample(self, o, reparameterized=False, temperature=1, idx=None, return_logits=False):
        z_dist = self.encode(o)
        return self.sample_z(z_dist, reparameterized, temperature, idx, return_logits)

    def decode(self, z):
        '''
        z: (batch_size, time, z_categoricals * z_categories)
        '''
        config = self.config
        shape = z.shape[:2]
        z = z.flatten(0, 1) # 将前两个维度展平，shape is (batch_size * time, z_categoricals * z_categorie(z_dim)))
        recons = self.decoder(z) # 解码潜在表示，shape is (batch_size * time, num_channels * frame_stack, height, width)
        if not config['env_grayscale']:
            # 如果是彩色图片 shape (batch_size * time, frame_stack, num_channels, height, width)
            recons = recons.unflatten(1, (config['env_frame_stack'], 3))
            # recons shape is (batch_size * time, frame_stack, height, width, num_channels)
            recons = recons.permute(0, 1, 3, 4, 2)
        recons = recons.unflatten(0, shape) # recons shape is (batch_size, time, frame_stack, height, width, num_channels)
        return recons

    def compute_decoder_loss(self, recons, o):
        '''
        recons: 重构观察后的数据，形状为(batch_size, time, frame_stack, height, width, channels) [wm_total_batch_size, 1, frame_stack, h, w, channels] 或者 [1 + prefix + wm_total_batch_size + 1, 1, frame_stack, h, w, channels]    
        o: 真实的观察数据 形状为(batch_size, time, frame_stack, height, width, channels) [wm_total_batch_size, 1, frame_stack, h, w, channels] 或者 [1 + prefix + wm_total_batch_size + 1, 1, frame_stack, h, w, channels]
        如果是灰度图，那么channels应该不存在
        return： loss: 重构损失, metrics: 一个字典，包含重构损失和重构均方误差
        '''
        assert utils.check_no_grad(o)
        config = self.config
        metrics = {}
        # recons.flatten(0, 1):  (batch_size * time, frame_stack, height, width, channels)
        # recon_mean： .permute(0, 2, 3, 1)： (batch_size * time, height, width, frame_stack，channels) todo 验证这里的shape变化
        recon_mean = recons.flatten(0, 1).permute(0, 2, 3, 1) 
        coef = config['obs_decoder_coef'] # 这里的obs_decoder_coef可以为0吗？
        # coef权重系数，用于控制重构损失在总损失中的比重
        if coef != 0:
            if config['env_grayscale']: 
                # 灰度图像的处理
                # o shape is (batch_size * time, height, width, frame_stack)
                o = o.flatten(0, 1).permute(0, 2, 3, 1)
            else:
                # 彩色图像的处理
                # o.flatten(0, 1): (batch_size * time, frame_stack， height, width， channels)
                # o = o.permute(0, 1, 4, 2, 3) # 将channels维度移动到最后，o shape is (batch_size * time, height, width, frame_stack， channels)
                # o = o.flatten(1, 2) # 展平frame_stack和channels维度，o shape is (batch_size * time, height, width, frame_stack * channels)
                o = o.flatten(0, 1).permute(0, 2, 3, 1, 4).flatten(-2, -1)
            # D.Independent： base_distribution: 基础分布（如 Normal、Categorical 等）；reinterpreted_batch_ndims: 要重新解释为事件维度的批次维度数量，这里是将最后一个维度视为采样分数维度
            # D.Normal(loc, scale)： loc: 均值（μ），决定分布的中心位置；scale: 标准差（σ），决定分布的分散程；
            recon_dist = D.Independent(D.Normal(recon_mean, torch.ones_like(recon_mean)), 3)
            # 在计算重构分布和原始观察之间的对数似然（log-likelihood），越接近真实观察值，对数概率越大，负号表示我们要最小化负对数似然
            loss = -coef * recon_dist.log_prob(o).mean()
            metrics['dec_loss'] = loss.detach() # 记录重构损失
        else:
            # 如果重构损失系数为0，则不计算重构损失 为啥？
            loss = torch.zeros(1, device=recons.device, requires_grad=False)
        # 计算重构均方误差
        metrics['recon_mae'] = torch.abs(o - recon_mean.detach()).mean()
        return loss, metrics

    def compute_entropy_loss(self, z_dist):
        '''
        z_dist: 观察编码后的潜在分布对象，采样的形状为(batch_size, time, z_categoricals, z_categories)
        '''
        config = self.config
        metrics = {}

        entropy = z_dist.entropy().mean() # 计算潜在分布的熵
        '''
        单个类别分布的最大熵:
        对于一个有 k 个类别的离散分布
        当各类别概率相等时（均匀分布）熵最大
        此时每个类别概率为 1/k
        最大熵值为 log(k)
        独立分布的熵:
        z_categoricals 个独立的类别分布
        每个分布有 z_categories 个类别
        独立分布的总熵是各个分布熵的和

        max_entropy = n * log(k)
        其中：
        - n = z_categoricals (独立分布的数量)
        - k = z_categories (每个分布的类别数)


        '''
        max_entropy = config['z_categoricals'] * math.log(config['z_categories']) # 最大熵
        normalized_entropy = entropy / max_entropy # 归一化熵
        metrics['z_ent'] = entropy.detach() # 记录熵值和归一化熵值
        metrics['z_norm_ent'] = normalized_entropy.detach()

        coef = config['obs_entropy_coef'] # 获取熵损失权重
        if coef != 0:
            # 归一化熵损失阈值
            if config['obs_entropy_threshold'] < 1:
                # 如果有阈值且为1，则阈值生效，当normalized_entropy小于config['obs_entropy_threshold']，则产生损失
                # 当 normalized_entropy 大于等于 config['obs_entropy_threshold'] 时，则损失为0，因为relu
                # 过滤掉熵过大的部分，保留熵小的部分，这样可以避免本身过大的熵继续增加，而让过小的熵能够增加从而保持稳定的不确定性
                # hinge loss, inspired by https://openreview.net/pdf?id=HkCjNI5ex
                loss = coef * torch.relu(config['obs_entropy_threshold'] - normalized_entropy)
            else:
                # 得到熵的负值作为损失，求最小，则求最大化熵
                loss = -coef * normalized_entropy
            metrics['z_entropy_loss'] = loss.detach()
        else:
            # 同样如果权重为0，则熵损失为0
            loss = torch.zeros(1, device=z_dist.base_dist.logits.device, requires_grad=False)

        return loss, metrics

    def compute_consistency_loss(self, z_logits, z_hat_probs):
        assert utils.check_no_grad(z_hat_probs)
        config = self.config
        metrics = {}
        coef = config['obs_consistency_coef']
        if coef > 0:
            cross_entropy = -((z_hat_probs.detach() * z_logits).sum(-1))
            cross_entropy = cross_entropy.sum(-1)  # independent
            loss = coef * cross_entropy.mean()
            metrics['enc_prior_ce'] = cross_entropy.detach().mean()
            metrics['enc_prior_loss'] = loss.detach()
        else:
            loss = torch.zeros(1, device=z_logits.device, requires_grad=False)
        return loss, metrics


class DynamicsModel(nn.Module):

    def __init__(self, config, z_dim, num_actions):
        super().__init__()
        self.config = config

        embeds = {
            'z': {'in_dim': z_dim, 'categorical': False}, # 环境的特征维度
            'a': {'in_dim': num_actions, 'categorical': True} # 动作的特征维度
        }
        modality_order = ['z', 'a']
        num_current = 2

        # todo 
        if config['dyn_input_rewards']:
            embeds['r'] = {'in_dim': 0, 'categorical': False}
            modality_order.append('r')

        if config['dyn_input_discounts']:
            embeds['g'] = {'in_dim': 0, 'categorical': False}
            modality_order.append('g')

        self.modality_order = modality_order

        # todo
        out_heads = {
            'z': {'hidden_dims': config['dyn_z_dims'], 'out_dim': z_dim},
            'r': {'hidden_dims': config['dyn_reward_dims'], 'out_dim': 1, 'final_bias_init': 0.0},
            'g': {'hidden_dims': config['dyn_discount_dims'], 'out_dim': 1,
                  'final_bias_init': config['env_discount_factor']}
        }

        # 这里可能是环境的记忆维度
        memory_length = config['wm_memory_length']
        max_length = 1 + config['wm_sequence_length']  # 1 for context
        # todo 预测网络
        self.prediction_net = nets.PredictionNet(
            modality_order, num_current, embeds, out_heads, embed_dim=config['dyn_embed_dim'],
            activation=config['dyn_act'], norm=config['dyn_norm'], dropout_p=config['dyn_dropout'],
            feedforward_dim=config['dyn_feedforward_dim'], head_dim=config['dyn_head_dim'],
            num_heads=config['dyn_num_heads'], num_layers=config['dyn_num_layers'],
            memory_length=memory_length, max_length=max_length)

    @property
    def h_dim(self):
        return self.prediction_net.embed_dim

    def predict(self, z, a, r, g, d, tgt_length, heads=None, mems=None, return_attention=False,
                compute_consistency=False):
        '''
        z shape is （1，sequence_length + extra - 1， z_categoricals * z_categories） 是环境特征提取后，采样的环境潜在状态
        a  shape is (1, sequence_length + extra - 2)
        r shape is (1, sequence_length + extra - 3)
        g shape is (batch_size(1), sequence_length + extra - 3) 是根据终止状态计算的折扣矩阵
        tgt_length: -2 + sequence_length + extra，是序列长度
        d shape is (1, sequence_length + extra - 2 - 1) 是一个结束标志，表示当前状态是否是终止状态
        compute_consistency is True 参数在 TWM 世界模型中用于控制是否计算一致性损失，这是一个重要的训练机制
        heads=None
        mems=None
        return_attention=False
        '''
        assert utils.check_no_grad(z, a, r, g, d)
        assert mems is None or utils.check_no_grad(*mems)
        config = self.config

        if compute_consistency:
            # todo 确认这里加1后，对后续的影响是什么？
            tgt_length += 1  # add 1 timestep for context

        inputs = {'z': z, 'a': a, 'r': r, 'g': g}
        heads = tuple(heads) if heads is not None else ('z', 'r', 'g')

        '''
        out shape is {
            'z': (batch_size, tgt_length / num_modalities, z_dim),
            'r': (batch_size, tgt_length / num_modalities, 1),
            'g': (batch_size, tgt_length / num_modalities, 1)
        }
        hiddens shape is (batch_size, tgt_length / num_modalities, dim)
        mems shape is [num_layers + 1, mem_length, batch_size, dim]
        这里的tgt_length / num_modalities 是因为每个模态的输出都是在最后一个位置进行预测的，所以需要除以模态数量

        这里的输出应该是预测下一个状态的潜在分布、奖励和折扣
        '''
        outputs = self.prediction_net(
            inputs, tgt_length, stop_mask=d, heads=heads, mems=mems, return_attention=return_attention)
        out, h, mems, attention = outputs if return_attention else (outputs + (None,))

        preds = {}

        if 'z' in heads:  # latent states
            z_categoricals = config['z_categoricals']
            z_categories = config['z_categories']
            z_logits = out['z'].unflatten(-1, (z_categoricals, z_categories)) # z_logits shape is (batch_size, tgt_length / num_modalities, z_categoricals, z_categories)

            if compute_consistency:
                # used for consistency loss
                preds['z_hat_probs'] = ObservationModel.create_z_dist(z_logits[:, :-1].detach()).base_dist.probs 
                z_logits = z_logits[:, 1:]  # remove context 移除第一帧 todo  z_lgoits shape is (batch_size, tgt_length / num_modalities - 1, z_categoricals, z_categories)

            z_dist = ObservationModel.create_z_dist(z_logits) # 创建中间的潜在分布
            preds['z_dist'] = z_dist

        if 'r' in heads:  # rewards todo 调试观察这边的shape
            r_params = out['r']
            if compute_consistency:
                r_params = r_params[:, 1:]  # remove context 移除第一帧 r_params shape is (batch_size, tgt_length / num_modalities - 1, 1)
            r_mean = r_params.squeeze(-1) # r_mean shape is (batch_size, tgt_length / num_modalities - 1)
            r_dist = D.Normal(r_mean, torch.ones_like(r_mean)) 

            r_pred = r_dist.mean
            preds['r_dist'] = r_dist  # used for dynamics loss 中间状态的奖励分布，用于计算动态损失
            preds['r'] = r_pred # 

        if 'g' in heads:  # discounts
            g_params = out['g']
            if compute_consistency:
                g_params = g_params[:, 1:]  # remove context 只保留第一帧的折扣
            g_mean = g_params.squeeze(-1) # g_mean shape is (batch_size, tgt_length / num_modalities, 1)
            g_dist = D.Bernoulli(logits=g_mean)

            g_pred = torch.clip(g_dist.mean, 0, 1)
            preds['g_dist'] = g_dist  # used for dynamics loss
            preds['g'] = g_pred

        return (preds, h, mems, attention) if return_attention else (preds, h, mems)

    def compute_dynamics_loss(self, preds, h, target_logits, target_r, target_g, target_weights):
        '''
        preds: 预测的结果，包含潜在状态、奖励和折扣 shape
        {
            'z': (batch_size, tgt_length / num_modalities, z_dim),
            'r': (batch_size, tgt_length / num_modalities, 1),
            'g': (batch_size, tgt_length / num_modalities, 1)
        }
        h: 隐藏状态 shape is (batch_size, tgt_length / num_modalities, dim)
        target_logits: 目标的潜在状态 logits，shape is (batch_size, tgt_length, z_categoricals * z_categories)
        target_r: 目标的奖励，shape is (batch_size, tgt_length - 1, 1)
        target_g: 目标的折扣，shape is (batch_size, tgt_length - 1, 1)
        target_weights: 目标的权重，shape is (batch_size, tgt_length - 1) 是中断状态的权重，通常是1或0
        '''
        assert utils.check_no_grad(target_logits, target_r, target_g, target_weights)
        config = self.config
        losses = []
        metrics = {}

        '''
        h.norm(dim=-1,  # 在最后一个维度上计算范数
            p=2      # L2范数，也称为欧几里德范数
        )

        batch_size: 批次大小（通常为1）
        tgt_length / num_modalities: 目标序列长度除以模态数量
        dim: 隐藏状态维度（embed_dim）
        '''
        metrics['h_norm'] = h.norm(dim=-1, p=2).mean().detach() # shape is (batch_size, tgt_length / num_modalities, dim) 这里仅仅只是记录起来

        if 'z' in preds:
            # 如果环境观察特征包含在预测中
            z_dist = preds['z_dist']
            z_logits = z_dist.base_dist.logits  # use normalized logits

            # doesn't check for q == 0
            target_probs = logits_to_probs(target_logits)
            cross_entropy = -((target_probs * z_logits).sum(-1)) # 目标的潜在状态 logits 和预测的潜在状态 logits 之间的交叉熵损失
            cross_entropy = cross_entropy.sum(-1)  # independent
            weighted_cross_entropy = target_weights * cross_entropy # 如果是中断状态则权重为0，否则为1
            weighted_cross_entropy = weighted_cross_entropy.sum() / target_weights.sum() # 计算加权交叉熵损失的平均值

            coef = config['dyn_z_coef']
            if coef != 0:
                transition_loss = coef * weighted_cross_entropy
                losses.append(transition_loss)

                metrics['z_pred_loss'] = transition_loss.detach()
                metrics['z_pred_ent'] = z_dist.entropy().detach().mean()
                metrics['z_pred_ce'] = weighted_cross_entropy.detach()

            # doesn't check for q == 0
            # 这边应该还是老一套，预测经过的潜在状态 logits （transformers）和目标的潜在状态 logits 之间的KL散度
            kl = (target_probs * (target_logits - z_logits.detach())).mean()
            kl = F.relu(kl.mean())
            metrics['z_kl'] = kl

        if 'r' in preds:
            r_dist = preds['r_dist']
            r_pred = preds['r']
            coef = config['dyn_reward_coef']
            if coef != 0:
                r_loss = -coef * r_dist.log_prob(target_r).mean() # 计算奖励的负对数似然损失，目标是为了使得r_dist和target_r尽可能接近
                losses.append(r_loss)
                metrics['reward_loss'] = r_loss.detach()
                metrics['reward_mae'] = torch.abs(target_r - r_pred.detach()).mean()
            metrics['reward'] = r_pred.mean().detach()

        if 'g' in preds:
            g_dist = preds['g_dist']
            g_pred = preds['g']
            coef = config['dyn_discount_coef']
            if coef != 0:
                g_dist._validate_args = False
                g_loss = -coef * g_dist.log_prob(target_g).mean()
                losses.append(g_loss)
                metrics['discount_loss'] = g_loss.detach()
                metrics['discount_mae'] = torch.abs(target_g - g_pred.detach()).mean()
            metrics['discount'] = g_pred.detach().mean()

        if len(losses) == 0:
            loss = torch.zeros(1, device=z.device, requires_grad=False)
        else:
            loss = sum(losses)
            metrics['dyn_loss'] = loss.detach()
        # 这里的loss是所有损失的总和
        # metrics是一个字典，包含了所有的指标
        # 这里的loss是所有损失的总和，包含了潜在状态的交叉熵损失、奖励的负对数似然损失和折扣的负对数似然损失
        return loss, metrics
