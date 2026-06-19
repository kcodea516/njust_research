# 服务器实验结果同步流程

## 目标

从服务器把 YOLO 实验结果同步到 Windows 本地时，只同步分析需要的实验产物：

```text
results.csv
args.yaml
results.png
confusion_matrix.png
confusion_matrix_normalized.png
Box*.png
val_batch*.jpg
train_batch*.jpg
labels.jpg
```

默认不把训练权重加入 Git：

```text
weights/
*.pt
*.pth
*.onnx
*.engine
```

如果后续需要做推理或集成，可以单独手动同步权重，但不要提交到 Git。

## 推荐方式

如果服务器生成了 `server_results.patch`，优先只应用 `runs` 实验产物，排除容易冲突的代码和配置文件。

在 Windows PowerShell 中执行：

```powershell
cd D:\Research\njust_research
git am --abort
git status
```

如果当前不在 `git am` 状态，`git am --abort` 可能提示没有进行中的 am，可以忽略。

## 检查 `.gitignore`

打开：

```powershell
notepad .gitignore
```

确保不要有这些会屏蔽实验结果的规则：

```text
runs/
results/*/*.png
src/models/runs/
```

保留下面这段，用来忽略权重但保留 csv/png/jpg/yaml：

```gitignore
# Keep experiment artifacts under src/models/runs,
# but ignore trained checkpoint weights
src/models/runs/**/weights/
src/models/runs/**/*.pt
src/models/runs/**/*.pth
src/models/runs/**/*.onnx
src/models/runs/**/*.engine
```

## 应用结果 patch

排除 `.gitignore` 和训练脚本，只应用实验产物：

```powershell
git apply --binary --reject --exclude=.gitignore --exclude=src/models/train_chexing.py D:\server_results.patch
```

这样做的原因是：服务器生成 patch 时，`.gitignore` 和 `train_chexing.py` 可能与 Windows 本地版本不一致，容易冲突；实验分析只需要 `src/models/runs/detect` 下的结果文件。

## 检查同步结果

```powershell
git status
```

正确情况应该看到新增或修改类似这些文件：

```text
src/models/runs/detect/.../results.csv
src/models/runs/detect/.../args.yaml
src/models/runs/detect/.../*.png
src/models/runs/detect/.../*.jpg
```

检查是否误加入权重或数据：

```powershell
git status --short | Select-String "\.pt|weights|data"
```

如果没有输出，或者只出现 `.gitignore`，说明没有误加入权重。

## 提交实验产物

如果 `git add src/models/runs/detect` 正常工作：

```powershell
git add .gitignore
git add src/models/runs/detect
git status
git commit -m "add vehicle detection experiment artifacts"
git push origin main
```

如果提示 `src/models/runs/detect` 被 ignore，则强制添加非权重结果：

```powershell
git add -f src/models/runs/detect/**/*.csv
git add -f src/models/runs/detect/**/*.png
git add -f src/models/runs/detect/**/*.jpg
git add -f src/models/runs/detect/**/*.yaml
```

不要执行：

```powershell
git add -f src/models/runs/detect/**/*.pt
```

## 后续约定

以后从服务器拉实验结果时，默认采用这个流程：

```text
只同步 runs 实验产物
不通过 patch 覆盖 .gitignore
不通过 patch 覆盖 train_chexing.py
不提交权重文件
```
