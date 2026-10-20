# -*- coding: utf-8 -*-
"""
水果识别与计数系统 —— Streamlit 网页版
启动:  streamlit run app.py
"""
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'best.pt')
DISPLAY_WIDTH = 1000            # 展示宽度(px)
CONF = 0.5                      # 置信度阈值

st.set_page_config(page_title='水果识别计数系统', layout='wide')
st.title('水果识别与计数系统')
st.caption('YOLO 目标检测 | 类别：apple / orange / banana')


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f'未找到模型文件 {MODEL_PATH}，请确认它和 app.py 在同一目录下。')
        st.stop()
    return YOLO(MODEL_PATH)


model = load_model()


def resize_to_width(img, width):
    """等比缩放到指定宽度"""
    w, h = img.size
    if w == width:
        return img
    return img.resize((width, int(h * width / w)), Image.Resampling.LANCZOS)


uploaded_file = st.file_uploader('上传一张水果图片', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    results = model(image, conf=CONF)[0]

    plotted_bgr = results.plot(line_width=1)
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(plotted_rgb)

    col1, col2 = st.columns(2)
    with col1:
        st.image(resize_to_width(image, DISPLAY_WIDTH), caption='原图')
    with col2:
        st.image(resize_to_width(result_img, DISPLAY_WIDTH), caption='检测结果')

    counts = {name: 0 for name in results.names.values()}
    for cls in results.boxes.cls:
        counts[results.names[int(cls)]] += 1

    st.subheader('计数结果')
    total = sum(counts.values())
    if total == 0:
        st.write('未检测到水果')
    else:
        for fruit, cnt in counts.items():
            if cnt > 0:
                st.write(f'- {fruit}: {cnt} 个')
        st.write(f'**合计：{total} 个**')

    with st.expander('查看每个检测框的置信度'):
        for i, box in enumerate(results.boxes, 1):
            st.write(f'{i}. {results.names[int(box.cls)]}: 置信度 {box.conf.item():.3f}')
