# 实验方案：
## 网络框架
- 替换掉MaxPool,增加深度可分离卷积
## 蒸馏方式
- 基于发版大模型进行蒸馏
## 蒸馏loss
- 增加Cosine loss
## 训练参数
- 调整num_batch_per_epoch: 1000,max_epoch: 500