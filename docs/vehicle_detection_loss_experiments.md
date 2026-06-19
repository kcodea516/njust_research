# 车辆检测损失函数改进实验计划

## 1. 当前实验结论

下面的结果来自 `src/models/runs/detect/*/results.csv`，取的是验证集 `mAP50-95` 最高的 epoch，不是最后一轮结果。

| 实验目录 | best epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vehicle5_YOLO11m_2x3090 | 70 | 0.8815 | 0.9126 | 0.9279 | 0.8391 |
| Vehicle5_YOLO26l_640_2x3090-2 | 50 | 0.8746 | 0.9322 | 0.9312 | 0.8366 |
| Vehicle5_YOLO26l_768_1x3090-2 | 75 | 0.8876 | 0.9235 | 0.9277 | 0.8315 |
| Vehicle5_YOLO26l_960_1x3090-2 | 79 | 0.8921 | 0.9257 | 0.9384 | 0.8485 |
| Vehicle5_YOLO26l_960_focal_g15 | 70 | 0.9077 | 0.9018 | 0.9381 | 0.8352 |
| Vehicle5_YOLO26l_960_focal_g15_alpha_auto | 67 | 0.8900 | 0.9212 | 0.9410 | 0.8363 |
| Vehicle5_YOLO26x_640_bce | 69 | 0.9092 | 0.8709 | 0.9263 | 0.8270 |
| Vehicle5_YOLO26x_960_bce | 75 | 0.9230 | 0.8975 | 0.9327 | 0.8342 |

当前最优单模型仍然是 `YOLO26l 960 + 默认 BCE`，也就是 `Vehicle5_YOLO26l_960_1x3090-2`。

第一批 Focal 消融结果显示，`gamma=1.5` 的 Focal BCE 并未优于默认 BCE；加入 `alpha auto` 后 mAP50 略有提升，但 mAP50-95 仍明显低于 baseline。因此当前阶段不继续扩展 YOLO26l Focal 实验，下一步优先测试 YOLO26x 默认 BCE 的模型容量上限。

第二批 YOLO26x 默认 BCE 结果显示，更大的模型容量没有带来收益。`YOLO26x 960` 的 Precision 提升到 0.9230，但 Recall 降到 0.8975，mAP50-95 只有 0.8342，比 `YOLO26l 960` 低 0.0143；`YOLO26x 640` 的 mAP50-95 进一步降到 0.8270，且混淆矩阵中 `Zhatu` 对角线只有约 0.73，弱类别表现不稳定。因此当前不继续跑 YOLO26x Focal，车辆检测单模型优先保留 `YOLO26l 960 + 默认 BCE`。

注意：本地 `runs` 目录里目前没有 `weights/best.pt`。如果要做推理、复验或集成，需要把服务器上的 `weights/` 目录同步回来。

## 2. YOLO 检测损失函数由哪几部分组成

Ultralytics YOLO 检测训练的主要损失可以简化理解为三项：

```text
总损失 = box loss + cls loss + dfl loss
```

### 2.1 box loss

`box loss` 是框位置损失，用来优化预测框和真实标注框的位置重合程度。

它主要关注：

```text
预测框中心是否准
宽高是否准
预测框和真实框的 IoU 是否高
```

这个部分决定模型能不能把车框准。

### 2.2 cls loss

`cls loss` 是类别损失，用来优化预测框属于哪个类别。

在当前五类货车任务中，它负责区分：

```text
Kong
Sanhuo
Jizhuangxiang
Caoguan
Zhatu
```

类别不均衡主要影响这一项。比如 `Sanhuo` 样本多，`Zhatu` 和 `Caoguan` 样本少，普通分类损失可能更容易被多数类和简单样本主导。

### 2.3 dfl loss

`dfl loss` 是 Distribution Focal Loss，用在框回归里面。

它不是我们这次说的分类 Focal Loss。它的作用是让模型更细致地学习边界框四个边的位置分布，从而提高定位精度。

可以这样理解：

```text
box loss 负责整体框准不准
dfl loss 负责框边界位置更细粒度地准不准
cls loss 负责类别分得准不准
```

本次改进只替换 `cls loss`，不动 `box loss` 和 `dfl loss`。这样实验更干净，结论更容易解释。

## 3. Focal BCE 是什么

当前 YOLO 检测分类损失默认用的是：

```text
BCEWithLogitsLoss(pred_scores, target_scores)
```

这里的 BCE 是 Binary Cross Entropy，中文一般叫二元交叉熵。YOLO 检测头对每个候选位置、每个类别都会输出一个 logit，(logit 是模型还没经过 sigmoid 的原始输出分数) 然后用 BCE 来判断这个类别应该是 1 还是 0。

`Focal BCE` 可以理解为：

```text
在 BCE 的基础上加一个 focal 权重
```

公式可以简化写成：

```text
Focal Loss = alpha * BCE * (1 - p_t)^gamma
Focal BCE = BCE * (1 - p_t) ^ gamma
```

其中：

```text
p_t：模型对正确目标的置信程度
gamma：聚焦参数，越大越压低简单样本
alpha：可选类别权重，用来加强少数类

gamma 解决：简单样本太多，训练被容易样本主导
alpha 解决：类别数量不均衡，少数类影响太弱
```

如果一个样本很容易，模型已经预测得很准，那么 `p_t` 接近 1：

```text
(1 - p_t) ^ gamma 接近 0
这个样本的损失会被压低
```

如果一个样本很难，模型预测得不好，那么 `p_t` 比较低：

```text
(1 - p_t) ^ gamma 仍然比较大
这个样本会继续贡献较多梯度
```

所以 Focal BCE 的目的不是让 loss 变复杂，而是让训练少被大量简单样本牵着走，多关注难样本、少数类和容易混淆的类别。

## 4. 我们替换的是哪一部分

这里说的 `patch`，意思是“运行时打补丁”。

不是修改 `.pt` 权重文件，也不是永久改掉服务器里安装的 Ultralytics 包，而是在本次 Python 训练进程启动后，把 Ultralytics 默认检测损失类里的分类损失对象换成我们自己的版本。

可以这样理解：

```text
Ultralytics 默认训练器
  原本使用：普通 BCE 分类损失

我们的 train_chexing.py 启动时
  如果检测到 --loss focal
  就调用 loss_patches.py
  在当前训练进程里把 cls loss 换成 Focal BCE
```

这个过程就叫 patch。它的优点是不用手工改 site-packages 里的源码，也不用改 `.pt` 文件；缺点是要固定 Ultralytics 版本，并且双卡 DDP 时要额外确认每个训练子进程都加载到了同一个 patch。

Ultralytics 8.3.228 的检测损失里，分类损失原本是这一类逻辑：

```python
loss[1] = BCE(pred_scores, target_scores).sum() / target_scores_sum
```

现在代码只把这里的 `BCE` 换成 `FocalBCEWithLogitsLoss`：

```python
loss[1] = FocalBCE(pred_scores, target_scores).sum() / target_scores_sum
```

所以替换位置是：

```text
src/models/train_chexing.py
  -> 调用 enable_focal_loss()
src/models/loss_patches.py
  -> 把 Ultralytics 的 v8DetectionLoss 里的 self.bce 替换成 FocalBCEWithLogitsLoss
```

训练时如果使用：

```bash
--loss focal
```

就会启用新的分类损失。

训练时如果不加，或者使用：

```bash
--loss bce
```

就仍然使用 YOLO 默认分类损失。

## 5. 损失函数是不是在 pt 文件里

不是。

`.pt` 文件主要保存的是模型结构信息和模型权重，也就是网络已经学到的参数。训练时用什么损失函数，主要由当前 Python 代码和训练器决定。

可以这样区分：

```text
pt 文件：保存模型权重，用来初始化模型
训练代码：定义前向传播、loss 计算、反向传播和参数更新
```

因此我们不需要修改 `yolo26l.pt` 文件。训练开始时，代码先加载 `yolo26l.pt` 的权重，然后每个 batch 前向传播，再用当前训练代码里的 loss 计算损失，最后反向传播更新权重。

这就是为什么可以做到“只替换分类损失，不修改 pt 文件”。

## 6. 什么是消融实验

消融实验就是一次只改一个因素，看这个因素到底有没有用。

如果同时改很多东西，比如同时换模型、换尺寸、改增强、改 loss，那么结果变好了也不知道是谁带来的提升。

本项目里正确的消融方式是：

```text
实验 A：YOLO26l 960 + 默认 BCE
实验 B：YOLO26l 960 + Focal BCE，其他全部不变
```

这样 B 比 A 好，才能说明 Focal BCE 可能有效。

再进一步：

```text
实验 C：YOLO26l 960 + Focal BCE + alpha auto，其他全部不变
```

这样 C 比 B 好，才能说明类别权重 `alpha` 可能有效。

所以消融实验的核心是：

```text
每次只动一个变量
其余变量保持一致
用同一套指标比较
```

这对论文、项目答辩和后续复现都很重要。

## 7. 已实现的训练参数

`src/models/train_chexing.py` 现在支持：

```bash
--loss bce
--loss focal
--focal-gamma 1.5
--focal-alpha none
--focal-alpha auto
--focal-alpha 0.35,0.25,0.45,0.70,0.75
```

含义如下：

| 参数 | 含义 |
| --- | --- |
| `--loss bce` | 使用 YOLO 默认分类损失 |
| `--loss focal` | 使用 Focal BCE 分类损失 |
| `--focal-gamma 1.5` | 控制难样本聚焦强度 |
| `--focal-alpha none` | 不加类别权重，只测试 gamma |
| `--focal-alpha auto` | 根据训练集每类数量自动给少数类更高权重 |
| `--focal-alpha 0.35,0.25,0.45,0.70,0.75` | 手动指定五个类别的 alpha |

## 8. 后续实验执行手册

后续实验按“先消融、再扩展、最后集成”的顺序进行。你有两张 3090，所以每个时间段采用“两张卡各跑一组单卡实验”的方式：

```text
终端 A：使用 GPU0，命令里写 --device 0
终端 B：使用 GPU1，命令里写 --device 1
```

这样不使用 DDP，也不会触发双卡验证阶段的 `dist.gather_object` 通信问题；同时两张卡都能利用起来。

### 8.0 第零步：更新代码并清理刚才试跑目录

先在服务器上执行一次，删除刚才失败或试跑产生的 Focal 目录，避免后续目录名冲突。

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

git pull

rm -rf src/models/runs/detect/Vehicle5_YOLO26l_960_focal_g15
rm -rf src/models/runs/detect/Vehicle5_YOLO26l_960_focal_g15_alpha_auto
```

如果 `git pull` 提示本地 `loss_patches.py` 有改动，执行：

```bash
git restore src/models/loss_patches.py
git pull
```

### 8.1 第一批并行实验：YOLO26l 960 的 Focal 消融

第一批同时跑第一组和第二组。两组都固定模型和尺寸，只比较是否加入类别权重。

```text
共同固定：
模型 = YOLO26l
imgsz = 960
loss = focal
gamma = 1.5
batch = 4
```

#### 第一组：YOLO26l 960 测试 Focal BCE

这组实验只改分类损失，不加入类别权重。

固定：

```text
模型 = YOLO26l
imgsz = 960
loss = focal
gamma = 1.5
alpha = none
device = 单卡
```

这组实验回答的问题是：

```text
只加入 Focal 的难样本聚焦机制，是否比 YOLO26l 960 默认 BCE 更好？
```

终端 A，使用 GPU0：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights weights/pretrained/yolo26l.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 960 \
  --device 0 \
  --workers 4 \
  --cos-lr \
  --name Vehicle5_YOLO26l_960_focal_g15 \
  --loss focal \
  --focal-gamma 1.5 \
  --focal-alpha none
```

完成后对比：

```text
对比 baseline：Vehicle5_YOLO26l_960_1x3090-2
重点指标：mAP50-95、Recall、Kong/Caoguan/Zhatu 每类 AP
判断标准：弱类别提升且整体 mAP 不明显下降，才认为 Focal BCE 有价值
```

#### 第二组：YOLO26l 960 测试 Focal BCE 加类别权重

这组实验在 Focal BCE 基础上加入自动类别权重。

固定：

```text
模型 = YOLO26l
imgsz = 960
loss = focal
gamma = 1.5
alpha = auto
device = 单卡
```

这组实验回答的问题是：

```text
在 Focal 难样本聚焦之外，再给少数类更高权重，是否进一步提升弱类别？
```

终端 B，使用 GPU1：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights weights/pretrained/yolo26l.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 960 \
  --device 1 \
  --workers 4 \
  --cos-lr \
  --name Vehicle5_YOLO26l_960_focal_g15_alpha_auto \
  --loss focal \
  --focal-gamma 1.5 \
  --focal-alpha auto
```

完成后对比：

```text
对比 1：Vehicle5_YOLO26l_960_1x3090-2
对比 2：Vehicle5_YOLO26l_960_focal_g15
重点指标：弱类别 AP 是否提升，整体 Precision 是否明显下降
判断标准：如果弱类别提升但整体 mAP 或 Precision 明显下降，说明 alpha 权重可能过强
```

### 8.2 第二批并行实验：YOLO26x 默认 BCE 上限测试

第二批同时跑第三组和第四组。两组都不改损失函数，只测试更大的 YOLO26x 在不同尺寸下的表现。

```text
共同固定：
模型 = YOLO26x
loss = bce
```

#### 第三组：YOLO26x 960 默认 BCE

这组实验不改损失函数，只换更大的模型。

固定：

```text
模型 = YOLO26x
imgsz = 960
loss = bce
device = 单卡
```

这组实验回答的问题是：

```text
在相同 960 尺寸和默认损失下，更大的 YOLO26x 是否比 YOLO26l 有更高上限？
```

终端 A，使用 GPU0：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights weights/pretrained/yolo26x.pt \
  --epochs 100 \
  --batch 2 \
  --imgsz 960 \
  --device 0 \
  --workers 4 \
  --cos-lr \
  --name Vehicle5_YOLO26x_960_bce \
  --loss bce
```

如果显存不够，先把 `--batch 2` 改成：

```bash
  --batch 1
```

完成后对比：

```text
对比 baseline：Vehicle5_YOLO26l_960_1x3090-2
重点指标：mAP50-95 是否明显提升，训练时间是否可接受
判断标准：如果提升很小但耗时显著增加，YOLO26x 不一定值得作为最终单模型
```

#### 第四组：YOLO26x 640 默认 BCE

这组实验测试大模型在较低分辨率下的性价比。

固定：

```text
模型 = YOLO26x
imgsz = 640
loss = bce
device = 优先单卡；如果单卡太慢且默认 BCE 双卡稳定，可以再考虑双卡
```

这组实验回答的问题是：

```text
YOLO26x 在 640 尺寸下是否能接近或超过 YOLO26l 960，同时获得更快推理速度？
```

终端 B，使用 GPU1：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights weights/pretrained/yolo26x.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 640 \
  --device 1 \
  --workers 4 \
  --cos-lr \
  --name Vehicle5_YOLO26x_640_bce \
  --loss bce
```

如果单卡显存很宽裕，可以下次把 `--batch 4` 提到：

```bash
  --batch 8
```

完成后对比：

```text
对比 1：Vehicle5_YOLO26l_640_2x3090-2
对比 2：Vehicle5_YOLO26l_960_1x3090-2
重点指标：mAP50-95、推理速度、弱类别 AP
判断标准：如果 YOLO26x 640 精度接近 YOLO26l 960 且速度更好，可以作为部署候选
```

### 8.3 第三批条件并行实验：Focal gamma 扫描

只有第一组 `YOLO26l_960_focal_g15` 比默认 BCE 更好时，才跑这一组。

当前第一批结果中，`YOLO26l_960_focal_g15` 和 `YOLO26l_960_focal_g15_alpha_auto` 的 mAP50-95 均低于默认 BCE baseline，因此暂时不跑 gamma 扫描。

固定：

```text
模型 = YOLO26l
imgsz = 960
loss = focal
alpha = none
gamma = 1.0 / 2.0
```

这组实验回答的问题是：

```text
Focal 的聚焦强度 gamma 取多少更合适？
```

终端 A，使用 GPU0，gamma = 1.0：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights weights/pretrained/yolo26l.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 960 \
  --device 0 \
  --workers 4 \
  --cos-lr \
  --name Vehicle5_YOLO26l_960_focal_g10 \
  --loss focal \
  --focal-gamma 1.0 \
  --focal-alpha none
```

终端 B，使用 GPU1，gamma = 2.0：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights weights/pretrained/yolo26l.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 960 \
  --device 1 \
  --workers 4 \
  --cos-lr \
  --name Vehicle5_YOLO26l_960_focal_g20 \
  --loss focal \
  --focal-gamma 2.0 \
  --focal-alpha none
```

完成后对比：

```text
对比对象：g10 / g15 / g20
重点指标：mAP50-95、弱类别 AP、Precision 是否下降
判断标准：选择整体和弱类别最平衡的 gamma，不一定选最高整体 mAP
```

### 8.4 第四批条件并行实验：YOLO26x 960 的 Focal 消融

只有同时满足下面条件时，才跑这一批：

```text
1. YOLO26l 960 上 Focal BCE 明确有效
2. YOLO26x 960 BCE 比 YOLO26l 960 BCE 有明显提升或有潜力
```

如果 YOLO26l 的第二组 `alpha auto` 没有收益，那么这一批只跑第六组，不跑第七组。

当前 YOLO26l Focal 消融没有超过默认 BCE，YOLO26x 默认 BCE 也没有超过 YOLO26l 960 baseline，因此暂时不跑 YOLO26x Focal。继续给 YOLO26x 加 Focal 只会扩大实验成本，且缺少明确收益依据。

#### 第六组：YOLO26x 960 测试 Focal BCE

固定：

```text
模型 = YOLO26x
imgsz = 960
loss = focal
gamma = 1.5
alpha = none
```

这组实验回答的问题是：

```text
更大的 YOLO26x 是否也能从 Focal BCE 中受益？
```

终端 A，使用 GPU0：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights weights/pretrained/yolo26x.pt \
  --epochs 100 \
  --batch 2 \
  --imgsz 960 \
  --device 0 \
  --workers 4 \
  --cos-lr \
  --name Vehicle5_YOLO26x_960_focal_g15 \
  --loss focal \
  --focal-gamma 1.5 \
  --focal-alpha none
```

完成后对比：

```text
对比 1：Vehicle5_YOLO26x_960_bce
对比 2：Vehicle5_YOLO26l_960_focal_g15
重点指标：YOLO26x 是否在弱类别和整体 mAP 上同时提升
判断标准：如果提升不明显，就不继续给 YOLO26x 跑 alpha auto
```

#### 第七组：YOLO26x 960 测试 Focal BCE 加类别权重

只有当第二组 `YOLO26l 960 + Focal + alpha auto` 明确优于第一组，并且第六组 YOLO26x Focal 也有效时，才跑这一组。

固定：

```text
模型 = YOLO26x
imgsz = 960
loss = focal
gamma = 1.5
alpha = auto
```

这组实验回答的问题是：

```text
YOLO26x 是否也需要类别权重来提升少数类？
```

终端 B，使用 GPU1：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights weights/pretrained/yolo26x.pt \
  --epochs 100 \
  --batch 2 \
  --imgsz 960 \
  --device 1 \
  --workers 4 \
  --cos-lr \
  --name Vehicle5_YOLO26x_960_focal_g15_alpha_auto \
  --loss focal \
  --focal-gamma 1.5 \
  --focal-alpha auto
```

完成后对比：

```text
对比对象：YOLO26x_960_bce / YOLO26x_960_focal_g15 / YOLO26x_960_focal_g15_alpha_auto
重点指标：弱类别 AP 是否提升，整体 Precision 是否下降
判断标准：只有弱类别提升且整体指标稳定时，才保留 alpha auto 方案
```

## 9. 评价指标记录表

每次实验完成后都记录：

```text
实验目录
模型
imgsz
batch
loss 类型
gamma
alpha
best epoch
Precision
Recall
mAP50
mAP50-95
每类 AP50
每类 AP50-95
训练时间
显存占用
```

尤其要记录弱类别：

```text
Kong
Caoguan
Zhatu
```

不能只看整体 mAP。如果整体 mAP 提升，但 `Zhatu` 仍然很低，那么类别不均衡问题没有真正解决。

## 10. 实验优先级总表

| 优先级 | 实验组 | 模型 | 尺寸 | 损失 | 是否必须跑 |
| --- | --- | --- | ---: | --- | --- |
| 1 | 第一组 | YOLO26l | 960 | Focal，gamma=1.5，alpha=none | 必跑 |
| 2 | 第二组 | YOLO26l | 960 | Focal，gamma=1.5，alpha=auto | 必跑 |
| 3 | 第三组 | YOLO26x | 960 | BCE | 必跑 |
| 4 | 第四组 | YOLO26x | 640 | BCE | 必跑 |
| 5 | 第五组 | YOLO26l | 960 | Focal gamma 扫描 | 条件跑 |
| 6 | 第六组 | YOLO26x | 960 | Focal，gamma=1.5，alpha=none | 条件跑 |
| 7 | 第七组 | YOLO26x | 960 | Focal，gamma=1.5，alpha=auto | 条件跑 |

暂时不建议跑：

```text
YOLO26l 640 Focal
YOLO26l 768 Focal
所有尺寸全部跑 Focal + alpha
一开始就 YOLO26x + Focal + alpha
```

## 11. 最终模型选择标准

车辆检测和分类模块最终至少保留两个结果：

```text
1. 单模型最优权重
2. 集成候选权重组合
```

单模型选择优先级：

```text
1. mAP50-95 高
2. 弱类别 Kong / Caoguan / Zhatu 不明显拖后腿
3. Precision 和 Recall 平衡
4. best epoch 和 last epoch 差距不过大
5. 推理速度和显存成本可接受
```

如果某个模型整体 mAP 最高，但 `Zhatu` 或 `Caoguan` 很差，需要谨慎作为最终模型。车辆模块不是只追求平均分，也要保证五类货车都能稳定识别。

## 12. 集成实验安排

集成不是越多模型越好，而是要选择错误互补的模型。

候选组合：

```text
YOLO26l 960 best
YOLO26x 960 best
一个低分辨率但混淆矩阵互补的模型
```

集成前先比较每个模型的：

```text
confusion_matrix_normalized.png
每类 AP
每类 Recall
```

如果两个模型错的是同一批类别，集成收益通常有限。只有当一个模型能补另一个模型的弱类时，集成才有价值。

最终车辆检测模块建议保留：

```text
单模型最优方案
集成最优方案
二者速度和精度对比
```

如果集成只提升很小但推理速度下降明显，实际系统可以优先用单模型。

完成上述实验、选出单模型最优和集成候选后，可以认为车辆检测和五类车型分类模块阶段性完成。之后再进入车牌检测识别模块和超载等级模块。
