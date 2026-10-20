# -*- coding: utf-8 -*-
"""
数据集划分脚本
================
把 data/ 下已按类别名存放的原始图片与标签，重新按 7:2:1 划分。

注意：本脚本的"原始数据"来源是 data/raw_images/ 与 data/raw_labels/
（如果你自己采集了新图片，请把图片放进 data/raw_images/、对应 YOLO 标签放进
data/raw_labels/，然后运行本脚本，它会重新生成 images/{train,val,test}）。

用法:  python src/split_dataset.py
"""
import os
import random
import shutil

SEED = 42
RATIO = (0.7, 0.2, 0.1)
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# 仓库根目录 = src/ 的上一级
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_IMG = os.path.join(DATA_DIR, 'raw_images')
RAW_LBL = os.path.join(DATA_DIR, 'raw_labels')

# 如果不存在 raw 目录，回退：把当前已划分的 train/val/test 合并作为原始数据
if not os.path.isdir(RAW_IMG):
    RAW_IMG = os.path.join(DATA_DIR, 'images')
    RAW_LBL = os.path.join(DATA_DIR, 'labels')

random.seed(SEED)

# 收集所有图片（递归，兼容 raw_images 平铺 或 images/{split} 结构）
img_files = []
for dp, dn, fn in os.walk(RAW_IMG):
    for f in fn:
        if f.lower().endswith(EXTS):
            img_files.append(os.path.join(dp, f))

names = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
random.shuffle(names)
total = len(names)
n_train = int(total * RATIO[0])
n_val = int(total * (RATIO[0] + RATIO[1]))
splits = {
    'train': names[:n_train],
    'val': names[n_train:n_val],
    'test': names[n_val:],
}
print(f'总数 {total} -> train {len(splits["train"])} / val {len(splits["val"])} / test {len(splits["test"])}')

OUT_IMG = os.path.join(DATA_DIR, 'images')
OUT_LBL = os.path.join(DATA_DIR, 'labels')

# 清空并重建输出目录
for d in (OUT_IMG, OUT_LBL):
    if os.path.exists(d):
        shutil.rmtree(d)
for split in splits:
    os.makedirs(os.path.join(OUT_IMG, split), exist_ok=True)
    os.makedirs(os.path.join(OUT_LBL, split), exist_ok=True)

# 建立 文件名 -> 图片路径 的映射（用文件名去匹配）
img_map = {}
for p in img_files:
    img_map[os.path.splitext(os.path.basename(p))[0]] = p

# 建立 文件名 -> 标签路径 的映射
lbl_map = {}
for dp, dn, fn in os.walk(RAW_LBL):
    for f in fn:
        if f.lower().endswith('.txt'):
            lbl_map[os.path.splitext(f)[0]] = os.path.join(dp, f)

missing = 0
for split, items in splits.items():
    for name in items:
        src_img = img_map.get(name)
        if src_img:
            ext = os.path.splitext(src_img)[1]
            shutil.copy2(src_img, os.path.join(OUT_IMG, split, name + ext))
        src_lbl = lbl_map.get(name)
        if src_lbl:
            shutil.copy2(src_lbl, os.path.join(OUT_LBL, split, name + '.txt'))
        else:
            missing += 1

print(f'[OK] 划分完成，缺失标签 {missing} 个')
print(f'[提示] 图片在 data/images/{{train,val,test}}，标签在 data/labels/{{train,val,test}}')
