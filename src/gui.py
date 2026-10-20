# -*- coding: utf-8 -*-
"""
水果识别计数系统 —— 桌面版 (Tkinter)
用法:  python fruit_detector_gui.py
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights', 'best.pt')
CONF = 0.5
MAX_W, MAX_H = 700, 500


class FruitDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('水果识别计数系统')
        self.root.geometry('800x700')

        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror('模型加载失败', f'无法加载模型: {e}')
            sys.exit(1)

        self._build_ui()

    def _build_ui(self):
        tk.Button(self.root, text='打开图片', command=self.open_image).pack(pady=10)

        self.frame_image = tk.Frame(self.root, bg='#f0f0f0')
        self.frame_image.pack(expand=True, fill='both', padx=10, pady=5)

        self.label_result = tk.Label(
            self.frame_image, text='检测结果将显示在这里',
            bg='#f0f0f0', font=('Microsoft YaHei', 14))
        self.label_result.pack(expand=True, fill='both')

        tk.Frame(self.root, height=2, bd=1, relief='sunken').pack(fill='x', padx=10, pady=5)

        self.text_result = tk.Text(self.root, height=8, font=('Consolas', 10))
        self.text_result.pack(fill='x', padx=10, pady=10)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('图片文件', '*.jpg *.jpeg *.png')])
        if path:
            self.detect(path)

    @staticmethod
    def _fit(img, max_w, max_h):
        w, h = img.size
        ratio = min(max_w / w, max_h / h, 1.0)
        return img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

    def detect(self, path):
        try:
            result = self.model(path, conf=CONF)[0]

            plotted = cv2.cvtColor(result.plot(line_width=1), cv2.COLOR_BGR2RGB)
            img = self._fit(Image.fromarray(plotted), MAX_W, MAX_H)

            photo = ImageTk.PhotoImage(img)
            self.label_result.config(image=photo, text='')
            self.label_result.image = photo

            counts = {name: 0 for name in result.names.values()}
            for cls in result.boxes.cls:
                counts[result.names[int(cls)]] += 1

            self.text_result.delete('1.0', tk.END)
            self.text_result.insert(tk.END, '计数结果:\n')
            for fruit, cnt in counts.items():
                if cnt > 0:
                    self.text_result.insert(tk.END, f'{fruit}: {cnt} 个\n')
            if sum(counts.values()) == 0:
                self.text_result.insert(tk.END, '未检测到水果\n')
        except Exception as e:
            messagebox.showerror('错误', f'检测失败:\n{e}')


if __name__ == '__main__':
    root = tk.Tk()
    FruitDetectorApp(root)
    root.mainloop()
