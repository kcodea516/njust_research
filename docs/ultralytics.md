模型前向传播：YOLO 模型结构 / Ultralytics 模块 / PyTorch
loss 计算：Ultralytics 的 loss 类，当前我们 patch 了 cls loss
反向传播：PyTorch autograd，loss.backward()
梯度下降：Ultralytics 创建 optimizer，然后 optimizer.step()
数据读取增强：Ultralytics dataset / dataloader
验证评估：Ultralytics validator

我们的脚本可控的是：
用哪个权重初始化：--weights
用哪个数据集：--data
训练多少轮：--epochs
输入尺寸：--imgsz
batch 大小：--batch
单卡还是双卡：--device
学习率和调度：--lr0 / --cos-lr
增强相关参数：close_mosaic 等
实验目录名：--name
是否启用 focal loss：--loss focal
focal 参数：--focal-gamma / --focal-alpha