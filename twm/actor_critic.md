# 为什么 `_compute_critic_loss` 使用负对数似然而非 MSE

在 `_compute_critic_loss` 方法中，代码使用了负对数似然（Negative Log Likelihood, NLL）损失而非均方误差（Mean Squared Error, MSE）：

```python
value_dist = D.Normal(values, torch.ones_like(values))
loss = -(weights * value_dist.log_prob(returns)).mean()
```

## 技术上可以使用 MSE

从技术上讲，是可以使用 MSE 计算 Critic 损失的：
```python
# 替代方案
loss = weights * (returns - values).pow(2).mean()
```

实际上，许多强化学习实现中确实使用 MSE 作为值函数损失。

## 使用负对数似然的优势

然而，使用负对数似然（NLL）有几个重要优势：

1. **概率建模**：
   - NLL 将值函数视为概率分布的参数（均值）
   - 这与强化学习中的随机性更为一致

2. **对异常值的鲁棒性**：
   - MSE 对异常值非常敏感（平方项放大大误差）
   - NLL 在某些情况下更稳定，特别是在有噪声的环境中

3. **一致的框架**：
   - 在概率模型中使用负对数似然保持了概率建模的一致性
   - Actor 使用对数概率，Critic 也使用对数概率

4. **潜在的不确定性建模**：
   - 虽然此处的标准差固定为1，但这种方法可以扩展为预测不确定性
   - `D.Normal(values, std)` 可以建模预测的不确定性

## 为什么不使用可学习的标准差？

代码使用固定标准差 `torch.ones_like(values)` 而非可学习标准差：

```python
value_dist = D.Normal(values, torch.ones_like(values))
```

这意味着实际上只是学习均值，标准差固定为1。当标准差固定时，最小化 NLL 实际上等价于最小化 MSE 的某种变形。

## 总结

1. **技术上可行**：使用 MSE 替代 NLL 是技术上可行的
2. **实践选择**：选择 NLL 可能是为了与概率框架保持一致性
3. **效果相似**：使用固定标准差的 NLL 和 MSE 效果可能相当接近
4. **扩展潜力**：NLL 方法为将来扩展到不确定性估计提供了可能性

这种选择反映了设计者对概率模型一致性的重视，尽管在这种特定实现中，两种方法的差异可能不太显著。