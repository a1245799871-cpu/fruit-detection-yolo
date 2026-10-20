# -*- coding: utf-8 -*-
"""
训练脚本（CPU 可跑）
用法:  python train.py
"""
import os
import yaml
from ultralytics import YOLO

# ======================= 配置区 =======================
# 数据集根目录（相对仓库根 data/ 目录，由 split_dataset.py 生成）
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
IMG_SIZE   = 640
BATCH_SIZE = 16
EPOCHS     = 50
DEVICE     = 'cpu'          # 有显卡改成 0
WORKERS    = 2
SEED       = 42

# 类别顺序必须与标注文件中的 class id 一致
CLASSES = ['apple', 'orange', 'banana']     # 0=apple 1=orange 2=banana

RUN_NAME = 'fruit_detector'
YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset.yaml')


def make_yaml():
    """生成 Ultralytics 数据集配置文件"""
    cfg = {
        'path': os.path.abspath(DATA_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASSES),
        'names': CLASSES,
    }
    with open(YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    print(f'[OK] 已生成 {YAML_PATH}')
    return cfg


if __name__ == '__main__':
    make_yaml()

    model = YOLO('yolo11n.pt')      # 首次运行会自动下载预训练权重

    model.train(
        data=YAML_PATH,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        optimizer='AdamW',
        lr0=0.01,
        cos_lr=True,
        warmup_epochs=2,
        patience=20,
        seed=SEED,
        name=RUN_NAME,
        exist_ok=True,
        amp=False,                  # CPU 不支持混合精度
    )

    best = f'runs/detect/{RUN_NAME}/weights/best.pt'
    print(f'[OK] 训练完成，最佳权重: {best}')

    metrics = YOLO(best).val()
    print(f'验证集 mAP@0.5      : {metrics.box.map50:.4f}')
    print(f'验证集 mAP@0.5:0.95 : {metrics.box.map:.4f}')
