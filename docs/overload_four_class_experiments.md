# 超载四分类实验执行方案

## 1. 模块定位

超载模块当前按“高速卡口图像中的车辆重量区间识别与疑似超载风险分类”来做。

这不是地磅称重，也不承诺单张图片能得到真实重量；它的作用是从车辆外观、载货状态、车厢状态等视觉线索中学习一个四分类 baseline，为系统输出疑似超载风险提供依据。

四个类别暂定为：

| 类别编号 | 类别名 | 重量区间 |
| ---: | --- | --- |
| 0 | under_25 | 25 吨以下 |
| 1 | weight_25_40 | 25-40 吨 |
| 2 | weight_40_52 | 40-52 吨 |
| 3 | over_52 | 52 吨以上 |

当前脚本默认边界为：

```text
class 0: weight <= 25
class 1: 25 < weight <= 40
class 2: 40 < weight <= 52
class 3: weight > 52
```

如果后续师兄给的表格里明确规定了边界，例如 `52` 归入 `52 以上`，再按表格规则调整。

## 2. 数据来源优先级

数据来源按优先级处理：

```text
1. 原始称重表格中的真实重量字段
2. labels.txt 中图片文件名末尾的重量字段
3. 图片文件名末尾的重量字段
```

例如：

```text
xxx_56.jpg -> 56 吨 -> class 3
```

注意：`超载.zip` 顶层包含 `labels.txt`、`images.rar` 和多个内层压缩包。当前已在内层压缩包中解出 `data_1.4-2.17_modify.xlsx`，表格字段包括：

```text
车号
检测时间
总重
车道
```

该表格能够证明原始数据中存在称重记录，但当前图片与表格的稳定关联字段还需要进一步核对。因此第一版 baseline 先按图片文件名末尾的重量字段构造四分类数据集；后续如果能确认图片与表格的车号、时间戳对应关系，再优先按表格中的 `总重` 字段重建标签。

当前已整理出的干净数据集统计为：

| 类别 | 数量 |
| --- | ---: |
| under_25 | 696 |
| weight_25_40 | 3770 |
| weight_40_52 | 18560 |
| over_52 | 9114 |
| 总计 | 32140 |

训练集 / 验证集划分：

```text
train: 27319
val: 4821
```

## 3. 本地整理干净数据集

先把 `超载.zip` 在本地解压到一个临时目录，不提交这个目录。

需要忽略：

```text
__MACOSX/
.DS_Store
._*
```

然后执行：

```powershell
cd D:\Research\njust_research

python src\models\prepare_overload_weight_dataset.py `
  --source "D:\Research\njust_research\超载解压目录" `
  --output data\overload_weight_dataset `
  --val-ratio 0.15 `
  --bins 25,40,52 `
  --overwrite `
  --allow-empty-class
```

如果已经拿到原始表格，但脚本无法自动识别列名，可以显式指定：

```powershell
python src\models\prepare_overload_weight_dataset.py `
  --source "D:\Research\njust_research\超载解压目录" `
  --output data\overload_weight_dataset `
  --image-col "图片名" `
  --weight-col "重量" `
  --val-ratio 0.15 `
  --bins 25,40,52 `
  --overwrite
```

整理后目录结构为：

```text
data/overload_weight_dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  overload.yaml
  manifest.csv
  dataset_summary.json
```

其中 YOLO 标签是全图框：

```text
class_id 0.5 0.5 1.0 1.0
```

这表示当前任务虽然使用 YOLO detection-compatible 格式，但本质是图像级重量区间分类。

为了避免中文车牌号和 macOS 压缩包编码问题影响服务器训练，干净数据集中的图片已经统一重命名为英文编号：

```text
overload_00000000.jpg
overload_00000001.jpg
...
```

原始文件名、解析重量、类别和来源路径保存在：

```text
data/overload_weight_dataset/manifest.csv
```

## 4. 上传服务器

本地整理完成后上传服务器：

```powershell
cd D:\Research\njust_research

scp -r data\overload_weight_dataset 3090:/home/kangsiyuan/research/njust_research/data/
```

服务器只负责训练，不直接提交 Git。

## 5. 第一组实验：YOLO26l 640 默认 BCE

这组是超载四分类 baseline。

固定：

```text
模型 = YOLO26l
imgsz = 640
loss = BCE
任务 = 视觉重量区间四分类
```

服务器终端：

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research
git pull

python src/models/train_overload.py \
  --data data/overload_weight_dataset/overload.yaml \
  --weights weights/pretrained/yolo26l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0 \
  --workers 4 \
  --cos-lr \
  --name Overload4_YOLO26l_640_bce \
  --loss bce
```

如果显存很宽裕，可以把 `--batch 8` 提到 `12` 或 `16`。先以跑通和稳定为主。

## 6. 第二组实验：YOLO26l 960 默认 BCE

如果 640 baseline 能正常收敛，再跑 960，看更高分辨率是否有助于识别货物状态和车厢细节。

```bash
conda activate ksy
cd /home/kangsiyuan/research/njust_research
git pull

python src/models/train_overload.py \
  --data data/overload_weight_dataset/overload.yaml \
  --weights weights/pretrained/yolo26l.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 960 \
  --device 1 \
  --workers 4 \
  --cos-lr \
  --name Overload4_YOLO26l_960_bce \
  --loss bce
```

## 7. 条件实验：Focal BCE 或类别权重

只有当四类样本极度不均衡，且弱类 AP 明显偏低时，才考虑 Focal BCE。

```bash
python src/models/train_overload.py \
  --data data/overload_weight_dataset/overload.yaml \
  --weights weights/pretrained/yolo26l.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 960 \
  --device 0 \
  --workers 4 \
  --cos-lr \
  --name Overload4_YOLO26l_960_focal_g15_alpha_auto \
  --loss focal \
  --focal-gamma 1.5 \
  --focal-alpha auto
```

这组不是优先实验。先看默认 BCE 是否已经能形成可解释 baseline。

## 8. 分析重点

每次实验完成后重点看：

```text
1. overall mAP50 和 mAP50-95
2. 每个重量区间的 AP
3. confusion_matrix.png
4. 40-52 与 52+ 是否大量混淆
5. 25 以下与 25-40 是否样本不足
6. 高分辨率是否明显改善重载区间
```

如果 `40-52` 和 `52+` 混淆严重，这是合理现象，因为纯视觉很难稳定区分接近重量段。文档和答辩中应说明：

```text
该模块是视觉风险辅助判断，不替代真实称重。
```

## 9. Git 规则

不提交：

```text
超载.zip
超载原始解压目录
data/overload_weight_dataset/
原始图片
weights/
*.pt
车辆超限智能识别系统-软著申报材料/
```

提交：

```text
src/models/prepare_overload_weight_dataset.py
src/models/train_overload.py
docs/overload_four_class_experiments.md
训练完成后的必要 results.csv / png / jpg
```
