# Fruit Detection & Counting with YOLO11

基于 YOLO11 的水果目标检测与计数系统 —— 从数据集构建、模型训练、对比实验到应用系统部署的完整流程。

识别类别：**apple（苹果） / orange（橙子） / banana（香蕉）**

Web 界面（Streamlit）与桌面界面（Tkinter）双实现，支持上传图片、绘制检测框、自动计数。

---

## 📊 实验结果

**最佳模型（`weights/best.pt`）表现：**

| 指标 | 数值 |
| :--- | :--- |
| **mAP@0.5** | **0.7665** |
| mAP@0.5:0.95 | 0.7012 |
| Precision | 0.6491 |
| Recall | 0.9260 |
| 最佳轮次 | epoch 106 / 150 |

**测试集端到端实测：21/21 张全部识别正确**（apple 10/10、orange 6/6、banana 5/5）。

---

训练曲线与混淆矩阵见 [`results/`](results/)：

- `results.png` —— 损失与 mAP 随轮次变化曲线
- `confusion_matrix.png` / `confusion_matrix_normalized.png` —— 混淆矩阵
- `BoxF1_curve.png` / `BoxPR_curve.png` —— F1 与 PR 曲线
- `val_batch*_labels.jpg` / `val_batch*_pred.jpg` —— 验证集真值 vs 预测对照

---

## 🔬 对比实验与数据修复历程

项目共进行了 11 次训练。**这个过程中最关键的转折不是调参，而是发现了标注数据的错误并修复。**

### 阶段一：数据错误导致的"假高分"

早期训练（`train`、`train-2`）的 mAP 高达 0.99，但模型实际上**只会识别 apple**。
原因是当时标注文件里 banana 的类别 id 全部错标为 `0`（apple），orange 也有约 20 个框错标为 `0`。

> 训练时模型看到的标注分布几乎全是 apple，于是它学到的最优策略就是"无脑输出 apple"，
> 从而在错误标注上拿到了虚假的 0.99 高分。

| 实验 | 模型 | mAP@0.5 | 真实表现（test 集） | 说明 |
| :--- | :--- | :---: | :--- | :--- |
| train | yolo11n | 0.9917 | 只认 apple（48%） | ❌ 数据未修复，假高分 |
| train-2 | yolo11s | 0.9871 | 只认 apple（48%） | ❌ 数据未修复，假高分 |

### 阶段二：修复标注后重训

修复了 banana 的类别 id、校正了 orange 中错标为 apple 的框后，重训得到：

| 实验 | 模型 | epochs | mAP@0.5 | 说明 |
| :--- | :--- | ---: | :---: | :--- |
| **train-9** | yolo11s | 150 | **0.7665** | ✅ **最终选用**，测试集 21/21 全对 |
| train-7 / train-8 | yolo11s | 150 | 0.7152 | 修复后数据，次优 |
| train-11 | yolo11s | 150 | 0.8716 | 部分轮次仍受脏数据影响 |

### 主要结论

1. **指标高低不能脱离数据质量看。** 0.9917 的"高分"其实是数据标注错误造成的假象，
   修复数据后真实 mAP 只有 0.7665 —— 这才是模型在三分类任务上的真实水平。
2. **标注质量是目标检测项目成败的关键。** 一个类别 id 标错，就能让模型彻底失效。
   本项目通过逐类核对文件名与类别 id 的对应关系，发现并修复了 banana/orange 的错标问题。
3. **判断模型好坏要用"端到端实测"，不能只看 mAP。** 最终选用 train-9 的依据是：
   它在独立测试集上 21/21 张全部识别正确（apple 10/10、orange 6/6、banana 5/5），
   而 mAP 更高的 train/train-2 反而只能识别 apple。

> 💡 最终交付选用 `train-9`（yolo11s，150 epochs，开数据增强），
> 权衡了 mAP（0.7665）与测试集端到端实测（21/21 全对）。

---

## 📁 项目结构

```
.
├── app.py                        # Web 界面（Streamlit）
├── src/
│   ├── train.py                  # 训练脚本（自动生成 dataset.yaml）
│   ├── split_dataset.py          # 数据集 7:2:1 划分脚本
│   └── gui.py                    # 桌面界面（Tkinter）
├── weights/
│   └── best.pt                   # 训练好的模型权重（19.2 MB）
├── results/                      # 训练曲线、混淆矩阵、预测对照图
├── examples/                     # 示例图片，可直接测试
├── data.yaml                     # 数据集配置
├── requirements.txt              # 依赖清单
├── LICENSE                       # MIT 协议
└── .gitignore
```

> 📌 **数据集（`data/`）未包含在仓库中**，见下方"数据集"说明。

---

## 📦 数据集

> ⚠️ **数据集（`data/`）不在本仓库中。** 出于精简仓库的考虑，203 张水果图片与标注未上传。
> 如需复现训练，请自行采集图片并按下方格式组织，或联系作者获取数据集。

| 项目 | 内容 |
| :--- | :--- |
| 图片总数 | 203 张（自建） |
| 类别数 | 3（apple / orange / banana） |
| 划分比例 | 训练 142 / 验证 40 / 测试 21（7:2:1） |
| 标注格式 | YOLO（`class x_center y_center width height`，归一化） |
| 标注框总数（训练集） | class 0: 198，class 1: 164，class 2: 158 |

**数据集目录结构（需自行准备）：**

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**类别 id 映射（固定，不可随意更改）：**

| class id | 名称 |
| :---: | :--- |
| 0 | apple |
| 1 | orange |
| 2 | banana |

> ⚠️ `data.yaml`、标注 `.txt` 文件与模型权重三者的类别顺序必须一致，
> 否则会出现“检测框正确但类别名错误”的问题。

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 建议 Python 3.9+
pip install -r requirements.txt
```

依赖：`ultralytics`、`streamlit`、`opencv-python`、`pillow`、`pyyaml`、`numpy`

### 2. 运行 Web 界面

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，上传 `examples/` 中的图片即可看到检测框与计数结果。

### 3. 运行桌面界面

```bash
python src/gui.py
```

### 4. 重新训练

> ⚠️ 数据集（`data/`）不在仓库中。重训前请先按上方"数据集"章节准备数据，
> 放入 `data/raw_images/` 与 `data/raw_labels/`，再执行下面的划分与训练。

```bash
python src/split_dataset.py     # 按 7:2:1 重新划分数据集
python src/train.py             # 训练并自动输出 mAP
```

> 若有 NVIDIA 显卡，把 `src/train.py` 中的 `DEVICE = 'cpu'` 改为 `DEVICE = 0` 可大幅加速。
> 首次运行会自动下载 `yolo11n.pt` 预训练权重。

### 5. 直接推理（命令行）

```bash
yolo predict model=weights/best.pt source=examples save=True
```

---

## 🛠 技术栈

| 环节 | 工具 |
| :--- | :--- |
| 检测框架 | Ultralytics YOLO11 |
| 深度学习 | PyTorch |
| 图像处理 | OpenCV、Pillow |
| Web 界面 | Streamlit |
| 桌面界面 | Tkinter |
| 数据标注 | LabelImg（YOLO 格式） |
| 数据划分 | 自写脚本（固定随机种子 42，保证可复现） |

---

## 📌 实现要点

- **完整的项目闭环**：数据采集与标注 → 数据集划分 → 模型训练 → 对比实验 → 评估选优 → 系统部署
- **可复现性**：数据集划分使用固定随机种子（`seed=42`），任何人拉取代码后划分结果一致
- **双界面交付**：Web 版便于演示与远程访问，桌面版可打包为独立 EXE 离线运行
- **对比实验驱动选型**：通过多组训练对比，用数据说明“小模型 + 适度轮次”在小数据集上更优，
  而非盲目追求更大的模型

---

## 📄 License

本项目代码采用 [MIT License](LICENSE)。

> ⚠️ **模型权重协议说明**：仓库中的 `weights/best.pt` 基于
> [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 微调，
> 该框架遵循 **AGPL-3.0** 协议。数据集图片来源于公开网络，版权归原作者所有。
> 本项目仅用于学习与教学用途。
