# 车辆检测鲁棒性实验计划

## 1. 为什么要做鲁棒性实验

当前车辆检测与五类车型分类模块已经完成了：

```text
YOLO11m / YOLO26l / YOLO26x 对比
640 / 768 / 960 多尺度对比
默认 BCE / Focal BCE / Focal + alpha auto 消融
最终模型选择
```

最终结论是：

```text
YOLO26l 960 + 默认 BCE 是当前最优方案
```

但如果简历和答辩只写“训练了一个 YOLO 检测模型”，深度不够。更好的表达方式是：

```text
面向高速复杂场景的重载车辆检测与鲁棒性分析
```

高速路货车图像的真实难点包括：

```text
远距离小目标
夜间低光照
车灯和车牌反光
监控画面压缩
高速运动模糊
不同车型之间外观相似
```

因此后续不再盲目训练新模型，而是在最终模型上做鲁棒性评估，分析模型在困难场景下的性能变化。

## 2. 本次只做两个高性价比实验

### 2.1 小目标鲁棒性

目标：

```text
分析车辆目标在图像中占比变小时，检测性能是否下降
```

做法：

```text
读取验证集 YOLO 标签
计算每张图中最大车辆框面积占整图比例
按面积分成 small / medium / large 三组
分别调用 YOLO val 评估
比较三组 mAP、Precision、Recall
```

默认使用三等分分组：

```text
最小 1/3：small
中间 1/3：medium
最大 1/3：large
```

这样即使数据集中没有特别小的车辆，也能分析“相对小目标”和“相对大目标”的性能差异。

回答的问题是：

```text
模型是否主要依赖大目标图像
远距离或画面占比较小的货车是否更容易漏检
哪一类目标尺寸下 Recall 下降最明显
```

### 2.2 类别鲁棒性

目标：

```text
分析不同车型在复杂图像退化条件下的性能下降规律
```

做法：

```text
在验证集上生成不同退化版本
clean：原始图像
low_light：亮度降低，模拟夜间
overexposure：亮度增强，模拟强反光/车灯过曝
gaussian_blur：高斯模糊，模拟虚焦或运动模糊
jpeg：低质量 JPEG 压缩，模拟监控视频压缩
```

每个版本都保留原始标签，然后用最终模型重新评估，输出每个类别的 AP50 和 AP50-95。

回答的问题是：

```text
哪一类车最稳定
哪一类车在低光照下下降最大
哪一类车在模糊或压缩后最容易混淆
Focal loss 没有提升时，问题是否来自类别本身还是图像退化
```

## 3. 已新增脚本

### 3.1 小目标鲁棒性脚本

```text
src/models/robustness_size_eval.py
```

输出目录默认：

```text
src/models/runs/robustness/Vehicle5_size_robustness
```

关键输出：

```text
size_robustness_summary.csv
small/
medium/
large/
```

其中 `size_robustness_summary.csv` 会记录：

```text
group
images
instances
area_min
area_max
precision
recall
map50
map50_95
class_instance_counts
```

### 3.2 类别鲁棒性脚本

```text
src/models/robustness_class_eval.py
```

输出目录默认：

```text
src/models/runs/robustness/Vehicle5_class_robustness
```

关键输出：

```text
class_robustness_summary.csv
vals/clean/
vals/low_light/
vals/overexposure/
vals/gaussian_blur/
vals/jpeg/
```

其中 `class_robustness_summary.csv` 会记录：

```text
variant
class_id
class_name
instances
ap50
ap50_95
mean_precision
mean_recall
mean_map50
mean_map50_95
```

## 4. 服务器运行命令

先确保服务器已经拉到最新代码：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research

git pull
```

### 4.1 第一组：小目标鲁棒性

```bash
python src/models/robustness_size_eval.py \
  --weights src/models/runs/detect/Vehicle5_YOLO26l_960_1x3090-2/weights/best.pt \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --name Vehicle5_size_robustness
```

如果显存紧张：

```bash
  --batch 4
```

### 4.2 第二组：类别鲁棒性

这组会生成退化图像，时间会比普通 val 更长。

```bash
python src/models/robustness_class_eval.py \
  --weights src/models/runs/detect/Vehicle5_YOLO26l_960_1x3090-2/weights/best.pt \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --name Vehicle5_class_robustness \
  --variants clean,low_light,overexposure,gaussian_blur,jpeg
```

如果先想快速测试脚本是否能跑通，可以加：

```bash
  --max-images 100
```

正式实验不要加 `--max-images`。

## 5. 结果拉回本地

在本地 Windows PowerShell 执行：

```powershell
cd D:\Research\njust_research

Remove-Item -Recurse -Force .\src\models\runs\robustness\Vehicle5_size_robustness -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\src\models\runs\robustness\Vehicle5_class_robustness -ErrorAction SilentlyContinue

scp -r 3090:/home/kangsiyuan/research/njust_research/src/models/runs/robustness/Vehicle5_size_robustness .\src\models\runs\robustness\
scp -r 3090:/home/kangsiyuan/research/njust_research/src/models/runs/robustness/Vehicle5_class_robustness .\src\models\runs\robustness\
```

检查不要拉入权重：

```powershell
git status --short | Select-String "weights|\.pt|\.pth|\.onnx|\.engine"
```

这两个鲁棒性实验不会生成 `best.pt`，一般不会有权重文件。

## 6. 结果分析重点

### 6.1 小目标鲁棒性

重点看：

```text
small 的 mAP50-95 是否明显低于 medium / large
small 的 Recall 是否明显下降
小目标组是否集中在某些类别
```

如果 small 下降明显，可以写：

```text
模型在远距离小目标场景下仍存在召回不足，后续可通过更高分辨率输入、小目标增强或切片推理优化。
```

如果 small 没有明显下降，可以写：

```text
模型在不同目标尺度下表现较稳定，说明 960 输入尺寸对高速货车场景的小目标具有较好的适应性。
```

### 6.2 类别鲁棒性

重点看：

```text
low_light 下哪个类别 AP 下降最大
overexposure 下哪个类别 AP 下降最大
gaussian_blur 下哪个类别最不稳定
jpeg 压缩是否显著影响整体 mAP
```

如果某类下降明显，可以结合混淆矩阵解释：

```text
该类别可能依赖车厢边缘、货物形态或车头细节，图像退化后可辨识特征减少。
```

## 7. 关于超载四分类数据集

当前超载数据集信息如下：

```text
class 0: 2989 train / 528 val
class 1: 708 train / 126 val
class 2: 4992 train / 882 val
class 3: 694 train / 123 val
```

它来自图像级分类文件夹，并被转换成 YOLO 检测格式：

```text
class_id 0.5 0.5 1.0 1.0
```

这说明它本质上是：

```text
图像级超载状态分类数据
```

而不是严格的目标检测数据。

因此可以尝试训练一个视觉超载 baseline，但要谨慎表述：

```text
可以说：视觉超载风险分类 baseline
不要说：准确超载检测或称重级判断
```

如果时间允许，可以后续单独做一组：

```text
YOLO 分类式 baseline 或全图框检测 baseline
```

但当前优先级不如车辆检测鲁棒性实验。因为鲁棒性实验直接服务于已经完成的车辆模块，也更容易写进简历并经得起追问。

## 8. 简历可用表述

完成鲁棒性实验后，可以把车辆模块写成：

```text
面向高速重载车辆图像，构建五类货车检测与分类模型，系统比较 YOLO11m、YOLO26l、YOLO26x 及多尺度输入；针对类别不均衡引入 Focal BCE 并完成消融实验，进一步设计小目标、低光照、过曝、模糊和压缩等鲁棒性评估，分析模型在复杂监控场景下的性能退化规律，最终选取 YOLO26l-960 作为最优方案。
```

这比只写“训练 YOLO 目标检测模型”更有深度。
