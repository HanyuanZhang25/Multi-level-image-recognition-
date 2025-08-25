import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os

from train_test import predict_image  # 假设你的主文件叫 main.py，包含predict_image等函数

# 加载类别映射
with open('class_map.json', 'r', encoding='utf-8') as f:
    class_map = json.load(f)

class_names = list(class_map.keys())  # ['class1', 'class2', 'class3', ...]
real_names = [class_map[c] for c in class_names]  # ['狗', '猫', '鸟', ...]

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("图像分类识别（Few-Shot）")
        self.root.geometry("400x500")

        self.label = tk.Label(root, text="请选择一张图片进行识别", font=("Arial", 14))
        self.label.pack(pady=10)

        self.img_label = tk.Label(root)
        self.img_label.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 16), fg="blue")
        self.result_label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="上传图片", command=self.upload_image)
        self.upload_btn.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        # 显示图片
        img = Image.open(file_path).resize((128, 128))
        self.tk_image = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_image)

        try:
            # 预测类别
            # pred_class = predict_image(file_path, class_names=class_names)
            pred_class = predict_image(
                file_path,
                encoder_path='best_prototypical_encoder.pth',
                class_names=class_names
            )
            pred_label = class_map.get(pred_class, "未知类别")
            self.result_label.config(text=f"识别结果：{pred_label}")
        except Exception as e:
            messagebox.showerror("错误", f"识别失败：{e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
