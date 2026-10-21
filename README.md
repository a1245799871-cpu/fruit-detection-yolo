# Fruit Detection & Counting with YOLO11

基于 YOLO11 的水果目标检测与计数系统，识别 **apple（苹果）/ orange（橙子）/ banana（香蕉）** 三类水果。

支持 Web 界面（Streamlit）与桌面界面（Tkinter）两种交互方式，可上传图片、绘制检测框并自动统计数量。

## 项目简介

这是一个完整的目标检测项目，覆盖从数据采集、标注、训练、评估到应用部署的完整流程。使用 Ultralytics YOLO11 框架，在自建的 203 张水果图片数据集上训练，最终模型在独立测试集上 21/21 张全部识别正确。

## 实验结果

**最佳模型（`weights/best.pt`）在验证集上的表现：**

| 指标 | 数值 |
| :--- | :--- |
| mAP@0.5 | 0.7665 |
| mAP@0.5:0.95 | 0.7012 |
| Precision | 0.6491 |
| Recall | 0.9260 |
| 最佳轮次 | epoch 106 / 150 |

**测试集端到端实测：21/21 张全部识别正确**（apple 10/10、orange 6/6、banana 5/5）。

训练曲线、混淆矩阵、验证集预测对照图见 [`results/`](results/)。

## 执行流程

1. **数据准备**：采集水果图片，使用 LabelImg 标注，输出 YOLO 格式（`class x_center y_center width height`，归一化）
2. **数据集划分**：按 7:2:1 划分为训练集（142 张）、验证集（40 张）、测试集（21 张）
3. **模型训练**：使用 `yolo11s.pt` 预训练权重迁移学习，150 epochs，batch=8，imgsz=640，开启数据增强
4. **评估选优**：在验证集上比较 mAP、Precision、Recall，选取最佳轮次权重
5. **应用部署**：用 Streamlit 开发 Web 界面，用 Tkinter 开发桌面界面，均可直接推理

## 项目结构

```
.
├── app.py                        # Web 界面（Streamlit）
├── src/
│   ├── train.py                  # 训练脚本
│   ├── split_dataset.py          # 数据集划分脚本
│   └── gui.py                    # 桌面界面（Tkinter）
├── weights/
│   └── best.pt                   # 训练好的模型权重
├── results/                      # 训练曲线、混淆矩阵、预测对照图
├── examples/                     # 示例图片
├── data.yaml                     # 数据集配置
├── requirements.txt              # 依赖清单
├── LICENSE                       # MIT 协议
└── .gitignore
```

## 数据集

| 项目 | 内容 |
| :--- | :--- |
| 图片总数 | 203 张 |
| 类别数 | 3（apple / orange / banana） |
| 划分比例 | 训练 142 / 验证 40 / 测试 21（7:2:1） |
| 标注格式 | YOLO（`class x_center y_center width height`，归一化） |
| 类别 id 映射 | 0=apple，1=orange，2=banana |

数据集未包含在仓库中。如需复现训练，请自行准备数据并按下述结构组织：

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

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖：`ultralytics`、`streamlit`、`opencv-python`、`pillow`、`pyyaml`、`numpy`

### 2. 运行 Web 界面

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，上传图片即可查看检测框与计数结果。

### 3. 运行桌面界面

```bash
python src/gui.py
```

### 4. 直接推理（命令行）

```bash
yolo predict model=weights/best.pt source=examples save=True
```

### 5. 重新训练

```bash
python src/split_dataset.py     # 按 7:2:1 重新划分数据集
python src/train.py             # 训练并自动输出 mAP
```

首次运行 `train.py` 会自动下载 `yolo11n.pt` 预训练权重。若有 NVIDIA 显卡，把 `src/train.py` 中的 `DEVICE = 'cpu'` 改为 `DEVICE = 0` 可大幅加速。

## 技术栈

| 环节 | 工具 |
| :--- | :--- |
| 检测框架 | Ultralytics YOLO11 |
| 深度学习 | PyTorch |
| 图像处理 | OpenCV、Pillow |
| Web 界面 | Streamlit |
| 桌面界面 | Tkinter |
| 数据标注 | LabelImg（YOLO 格式） |
| 数据划分 | 自写脚本（固定随机种子，保证可复现） |

## License

本项目代码采用 [MIT License](LICENSE)。

仓库中的 `weights/best.pt` 基于 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 微调，该框架遵循 AGPL-3.0 协议。数据集图片来源于公开网络，版权归原作者所有。本项目仅用于学习与教学用途。
